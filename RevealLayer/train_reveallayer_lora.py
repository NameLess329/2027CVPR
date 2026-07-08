from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DIFFUSERS_SRC = PROJECT_ROOT / "diffusers" / "src"
if LOCAL_DIFFUSERS_SRC.exists() and str(LOCAL_DIFFUSERS_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIFFUSERS_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        elif value is not None:
            base[key] = value
    return base


def _flatten_for_tracker(value: Any, prefix: str = "") -> dict[str, Any]:
    flat = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_key = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_for_tracker(child, child_key))
    elif isinstance(value, (list, tuple)):
        flat[prefix] = ",".join(str(item) for item in value)
    elif value is None:
        flat[prefix] = "null"
    elif isinstance(value, (int, float, str, bool)):
        flat[prefix] = value
    else:
        flat[prefix] = str(value)
    return flat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RevealLayer FLUX LoRA.")
    parser.add_argument("--config", type=str, default="configs/train_reveallayer_lora.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max_layers", type=int, default=None)
    parser.add_argument("--layer_order", type=str, default=None, choices=("as_is", "reverse"))
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=("no", "fp16", "bf16"))
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=("adamw", "prodigy"))
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=None)
    parser.add_argument("--preview_steps", type=int, default=None)
    parser.add_argument("--preview_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_checkpoint", type=str, default=None)
    parser.add_argument("--transparent_decoder_ckpt", type=str, default=None)
    parser.add_argument("--find_unused_parameters", action="store_true", default=None)
    return parser.parse_args()


def _config_with_cli(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _load_config(args.config)
    overrides: dict[str, Any] = {}
    for key in ("model_path", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    data = {}
    for key in ("manifest_path", "data_root", "resolution", "max_layers", "layer_order"):
        value = getattr(args, key)
        if value is not None:
            data[key] = value
    if data:
        overrides["data"] = data
    training = {}
    for key in (
        "train_batch_size",
        "gradient_accumulation_steps",
        "max_train_steps",
        "mixed_precision",
        "resume_from_checkpoint",
        "checkpointing_steps",
        "preview_steps",
        "preview_dir",
        "seed",
        "find_unused_parameters",
    ):
        value = getattr(args, key)
        if value is not None:
            training[key] = value
    if training:
        overrides["training"] = training
    optimizer = {}
    if args.optimizer is not None:
        optimizer["name"] = args.optimizer
    if args.learning_rate is not None:
        optimizer["learning_rate"] = args.learning_rate
    if optimizer:
        overrides["optimizer"] = optimizer
    lora = {}
    if args.lora_rank is not None:
        lora["rank"] = args.lora_rank
    if args.lora_checkpoint is not None:
        lora["checkpoint"] = args.lora_checkpoint
    if lora:
        overrides["lora"] = lora
    if args.transparent_decoder_ckpt is not None:
        overrides["transparent_decoder"] = {"checkpoint": args.transparent_decoder_ckpt}
    return _deep_update(cfg, overrides)


def _dtype_from_mixed_precision(value: str):
    import torch

    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    return torch.float32


def _build_pipeline(cfg: dict[str, Any], device, dtype):
    import torch
    from diffusers import FluxPipeline
    from diffusers.configuration_utils import FrozenDict
    from models.custom_model_mmdit import CustomFluxTransformer2DModel

    pipe = FluxPipeline.from_pretrained(cfg["model_path"], torch_dtype=dtype)
    original_transformer = pipe.transformer
    mmdit_config = dict(original_transformer.config)
    mmdit_config["_class_name"] = "CustomSD3Transformer2DModel"
    mmdit_config["max_layer_num"] = int(cfg.get("data", {}).get("max_layers", 12))
    custom_transformer = CustomFluxTransformer2DModel.from_config(FrozenDict(mmdit_config)).to(dtype=dtype)
    custom_transformer.load_state_dict(original_transformer.state_dict(), strict=False)
    pipe.transformer = custom_transformer
    pipe.to(device)
    ckpt_dir = cfg.get("ckpt_dir")
    if ckpt_dir:
        _load_layer_pe(pipe.transformer, ckpt_dir)
        _load_refiner(pipe.transformer, ckpt_dir)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    return pipe


def _load_layer_pe(transformer, ckpt_dir: str) -> None:
    import torch

    corrected_path = Path(ckpt_dir) / "layer_pe_corrected.pt"
    fallback_path = Path(ckpt_dir) / "layer_pe.pt"
    layer_pe_path = corrected_path if corrected_path.exists() else fallback_path
    if layer_pe_path.exists():
        transformer.layer_pe.data.copy_(torch.load(layer_pe_path, map_location="cpu").to(transformer.layer_pe.device))


def _load_refiner(transformer, ckpt_dir: str) -> None:
    import torch

    module_path = Path(ckpt_dir) / "Refiner.pt"
    if module_path.exists():
        state_dict = torch.load(module_path, map_location="cpu")
        corrected_state_dict = {f"refiner.{key}": value for key, value in state_dict.items()}
        transformer.load_state_dict(corrected_state_dict, strict=False)


def _build_transparent_decoder(cfg: dict[str, Any], device, dtype):
    import torch
    from models.custom_model_xvae import AutoencoderKLTransformerTraining as CustomVAE

    decoder_cfg = cfg.get("transparent_decoder", {})
    ckpt = decoder_cfg.get("checkpoint")
    if not ckpt:
        return None
    decoder = CustomVAE()
    decoder.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    decoder.to(device=device, dtype=dtype)
    train_decoder = bool(decoder_cfg.get("train", False))
    decoder.train(train_decoder)
    for param in decoder.parameters():
        param.requires_grad_(train_decoder)
    return decoder


def _enable_extra_trainables(transformer, cfg: dict[str, Any]) -> list[str]:
    enabled = []
    if cfg["lora"].get("train_layer_pe", True):
        transformer.layer_pe.requires_grad_(True)
        transformer.layer_pe.data = transformer.layer_pe.data.float()
        enabled.append("layer_pe")
    if cfg["lora"].get("train_refiner", True):
        for param in transformer.refiner.parameters():
            param.requires_grad_(True)
            param.data = param.data.float()
        enabled.append("refiner")
    return enabled


def _set_gradient_checkpointing(transformer, enabled: bool) -> None:
    module = transformer.module if hasattr(transformer, "module") else transformer
    if hasattr(module, "gradient_checkpointing"):
        module.gradient_checkpointing = enabled


def _build_optimizer(params, cfg: dict[str, Any]):
    name = cfg["optimizer"].get("name", "prodigy").lower()
    lr = float(cfg["optimizer"].get("learning_rate", 1.0))
    weight_decay = float(cfg["optimizer"].get("weight_decay", 0.01))
    betas = tuple(cfg["optimizer"].get("betas", (0.9, 0.99)))
    if name == "prodigy":
        try:
            from prodigyopt import Prodigy
        except ImportError as exc:
            raise ImportError("optimizer.name=prodigy requires `pip install prodigyopt`.") from exc
        return Prodigy(params, lr=lr, weight_decay=weight_decay, betas=betas)
    import torch

    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def _build_lr_scheduler(optimizer, cfg: dict[str, Any], max_train_steps: int):
    scheduler_name = cfg["optimizer"].get("lr_scheduler", "constant")
    warmup_steps = int(cfg["optimizer"].get("lr_warmup_steps", 0))
    try:
        from diffusers.optimization import get_scheduler
    except ModuleNotFoundError:
        try:
            from transformers.optimization import get_scheduler
        except ModuleNotFoundError:
            get_scheduler = None

    if get_scheduler is not None:
        return get_scheduler(
            scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps,
        )

    import torch

    if scheduler_name != "constant":
        raise ImportError(
            f"lr_scheduler={scheduler_name!r} requires diffusers.optimization or transformers.optimization. "
            "Install a compatible diffusers/transformers version or set optimizer.lr_scheduler=constant."
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def _current_lr(optimizer, lr_scheduler) -> float:
    if hasattr(lr_scheduler, "get_last_lr"):
        values = lr_scheduler.get_last_lr()
        if values:
            return float(values[0])
    if optimizer.param_groups:
        return float(optimizer.param_groups[0].get("lr", 0.0))
    return 0.0


def _save_checkpoint(accelerator, pipe, transparent_decoder, cfg: dict[str, Any], output_dir: Path, step: int) -> None:
    from training.lora import DIFFUSERS_LORA_NAME, LEGACY_LORA_NAME, save_diffusers_lora_state, save_lora_state

    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(checkpoint_dir))
    transformer = accelerator.unwrap_model(pipe.transformer)
    save_diffusers_lora_state(transformer, checkpoint_dir / DIFFUSERS_LORA_NAME)
    save_lora_state(transformer, checkpoint_dir / LEGACY_LORA_NAME)
    if cfg["lora"].get("train_layer_pe", True):
        import torch

        torch.save(transformer.layer_pe.detach().cpu(), checkpoint_dir / "layer_pe.pt")
    if cfg["lora"].get("train_refiner", True):
        import torch

        torch.save({key: value.detach().cpu() for key, value in transformer.refiner.state_dict().items()}, checkpoint_dir / "Refiner.pt")
    if transparent_decoder is not None and cfg.get("transparent_decoder", {}).get("train", False):
        import torch

        decoder = accelerator.unwrap_model(transparent_decoder)
        torch.save({key: value.detach().cpu() for key, value in decoder.state_dict().items()}, checkpoint_dir / "transparent_decoder_ckpt.pth")


def _rgb_tensor_to_uint8(tensor):
    import numpy as np
    import torch

    array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round()
    array = array.to(dtype=torch.uint8).permute(1, 2, 0).numpy()
    return np.ascontiguousarray(array)


def _alpha_tensor_to_uint8(tensor, already_01: bool = False):
    import numpy as np
    import torch

    alpha = tensor.detach().float().cpu()
    if not already_01:
        alpha = (alpha + 1.0) / 2.0
    array = (alpha.clamp(0, 1) * 255.0).round().to(dtype=torch.uint8).squeeze(0).numpy()
    return np.ascontiguousarray(array)


def _make_rgba_image(rgb, alpha, alpha_already_01: bool = False):
    from PIL import Image
    import numpy as np

    rgb_array = _rgb_tensor_to_uint8(rgb)
    alpha_array = _alpha_tensor_to_uint8(alpha, already_01=alpha_already_01)
    rgba = np.concatenate([rgb_array, alpha_array[..., None]], axis=-1)
    return Image.fromarray(rgba, mode="RGBA")


def _save_visual_previews(output_dir: Path, batch, pred_rgb, pred_alpha, validation_boxes, step: int, preview_dir: str) -> None:
    from PIL import Image

    step_dir = output_dir / preview_dir / f"step-{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    source = _rgb_tensor_to_uint8(batch.pixel_values[0])
    Image.fromarray(source, mode="RGB").save(step_dir / "source_full_image.png")

    pred_rgba_images = []
    for layer_idx, box in enumerate(validation_boxes):
        if box is None or layer_idx >= pred_rgb.shape[0]:
            continue
        image = _make_rgba_image(pred_rgb[layer_idx], pred_alpha[layer_idx], alpha_already_01=False)
        if layer_idx == 0:
            name = "pred_layer_00_full_rgba.png"
        elif layer_idx == 1:
            name = "pred_layer_01_background_rgba.png"
        else:
            name = f"pred_layer_{layer_idx:02d}_foreground_rgba.png"
        image.save(step_dir / name)
        pred_rgba_images.append((layer_idx, image))

    target_rgba_images = []
    for target_idx in range(batch.target_rgb.shape[1]):
        layer_idx = target_idx + 1
        if layer_idx >= len(validation_boxes) or validation_boxes[layer_idx] is None:
            continue
        image = _make_rgba_image(batch.target_rgb[0, target_idx], batch.target_alpha[0, target_idx], alpha_already_01=True)
        if layer_idx == 1:
            name = "target_layer_01_background_rgba.png"
        else:
            name = f"target_layer_{layer_idx:02d}_foreground_rgba.png"
        image.save(step_dir / name)
        target_rgba_images.append((layer_idx, image))

    pred_by_layer = {layer_idx: image for layer_idx, image in pred_rgba_images}
    if 1 in pred_by_layer:
        merged = pred_by_layer[1].copy()
        for layer_idx in sorted((idx for idx in pred_by_layer if idx >= 2), reverse=True):
            merged = Image.alpha_composite(merged, pred_by_layer[layer_idx])
        merged.save(step_dir / "pred_merged_rgba.png")


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    cfg = _config_with_cli(args)

    import torch
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.logging import get_logger
    from accelerate.utils import set_seed
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from training.dataset import RevealLayerDataset, reveal_layer_collate
    from training.lora import add_lora_to_transformer, load_lora_state
    from training.losses import flow_matching_loss, hard_constraint_alpha_loss, soft_orthogonality_loss
    from training.train_step import decode_predicted_rgba, forward_flow_matching

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]

    if int(train_cfg.get("train_batch_size", 1)) != 1:
        raise ValueError("RevealLayer list_layer_box is currently unbatched; set train_batch_size=1.")
    needs_transparent_decoder = bool(loss_cfg.get("enable_alpha_loss") or loss_cfg.get("enable_orth_loss"))

    ddp = DistributedDataParallelKwargs(find_unused_parameters=bool(train_cfg.get("find_unused_parameters", True)))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        mixed_precision=train_cfg.get("mixed_precision", "bf16"),
        log_with=train_cfg.get("report_to", "tensorboard"),
        project_dir=str(Path(cfg["output_dir"]) / train_cfg.get("logging_dir", "logs")),
        kwargs_handlers=[ddp],
    )
    logger = get_logger(__name__)
    if train_cfg.get("seed") is not None:
        set_seed(int(train_cfg["seed"]))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    dataset = RevealLayerDataset(
        manifest_path=data_cfg["manifest_path"],
        data_root=data_cfg.get("data_root"),
        resolution=int(data_cfg.get("resolution", 1024)),
        max_layers=int(data_cfg.get("max_layers", 12)),
        layer_order=data_cfg.get("layer_order", "as_is"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 4)),
        collate_fn=reveal_layer_collate,
        pin_memory=True,
    )

    dtype = _dtype_from_mixed_precision(train_cfg.get("mixed_precision", "bf16"))
    pipe = _build_pipeline(cfg, accelerator.device, dtype)
    transparent_decoder = _build_transparent_decoder(cfg, accelerator.device, dtype)
    if needs_transparent_decoder and transparent_decoder is None:
        raise ValueError("Alpha/orth losses require transparent_decoder.checkpoint in the config.")
    pipe.transformer, targets = add_lora_to_transformer(
        pipe.transformer,
        rank=int(cfg["lora"].get("rank", 64)),
        alpha=cfg["lora"].get("alpha"),
        dropout=float(cfg["lora"].get("dropout", 0.0)),
        target_modules=cfg["lora"].get("target_modules"),
    )
    lora_checkpoint = cfg["lora"].get("checkpoint")
    if lora_checkpoint is None and cfg.get("ckpt_dir"):
        candidate = Path(cfg["ckpt_dir"]) / "pytorch_lora_weights.safetensors"
        if candidate.exists():
            lora_checkpoint = str(candidate)
    if lora_checkpoint:
        load_lora_state(pipe.transformer, lora_checkpoint)
    extra_trainables = _enable_extra_trainables(pipe.transformer, cfg)
    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", False))
    _set_gradient_checkpointing(pipe.transformer, gradient_checkpointing)

    trainable_params = [param for param in pipe.transformer.parameters() if param.requires_grad]
    if transparent_decoder is not None and cfg.get("transparent_decoder", {}).get("train", False):
        trainable_params.extend(param for param in transparent_decoder.parameters() if param.requires_grad)
    optimizer = _build_optimizer(trainable_params, cfg)
    max_train_steps = int(train_cfg.get("max_train_steps", 50000))
    lr_scheduler = _build_lr_scheduler(optimizer, cfg, max_train_steps)

    if transparent_decoder is not None and cfg.get("transparent_decoder", {}).get("train", False):
        pipe.transformer, transparent_decoder, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            pipe.transformer,
            transparent_decoder,
            optimizer,
            dataloader,
            lr_scheduler,
        )
    else:
        pipe.transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            pipe.transformer,
            optimizer,
            dataloader,
            lr_scheduler,
        )
    pipe.transformer.train()
    _set_gradient_checkpointing(pipe.transformer, gradient_checkpointing)
    if transparent_decoder is not None:
        transparent_decoder.train(bool(cfg.get("transparent_decoder", {}).get("train", False)))

    resume = train_cfg.get("resume_from_checkpoint")
    global_step = 0
    if resume:
        if resume == "latest":
            checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            resume = str(checkpoints[-1]) if checkpoints else None
        if resume:
            accelerator.load_state(str(resume))
            global_step = int(Path(resume).name.split("-")[-1])
            logger.info("Resumed from %s", resume)

    accelerator.init_trackers("reveallayer_lora", config=_flatten_for_tracker(cfg))
    logger.info("LoRA targets: %s", targets)
    logger.info("Extra trainables: %s", extra_trainables)
    logger.info("Gradient checkpointing: %s", gradient_checkpointing)
    preview_steps = int(train_cfg.get("preview_steps", train_cfg.get("checkpointing_steps", 1000)))
    preview_dir = str(train_cfg.get("preview_dir", "previews"))
    progress_bar = tqdm(
        total=max_train_steps,
        initial=global_step,
        desc="training",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    )

    while global_step < max_train_steps:
        for batch in dataloader:
            batch.pixel_values = batch.pixel_values.to(accelerator.device)
            batch.target_rgb = batch.target_rgb.to(accelerator.device)
            batch.target_alpha = batch.target_alpha.to(accelerator.device)
            with accelerator.accumulate(pipe.transformer):
                result = forward_flow_matching(
                    pipe,
                    batch,
                    num_train_timesteps=1000,
                    min_timestep_boundary=float(train_cfg.get("min_timestep_boundary", 0.0)),
                    max_timestep_boundary=float(train_cfg.get("max_timestep_boundary", 1.0)),
                    max_sequence_length=int(train_cfg.get("max_sequence_length", 512)),
                    guidance_scale=float(train_cfg.get("guidance_scale", 1.0)),
                )
                fm_loss = flow_matching_loss(
                    result.prediction,
                    result.target,
                    result.token_mask,
                )
                loss = float(loss_cfg.get("lambda_fm", 1.0)) * fm_loss
                alpha_loss = result.prediction.sum() * 0.0
                orth_loss = result.prediction.sum() * 0.0
                pred_rgb = None
                pred_alpha = None
                if needs_transparent_decoder:
                    pred_rgb, pred_alpha = decode_predicted_rgba(pipe, transparent_decoder, result)
                    if loss_cfg.get("enable_alpha_loss"):
                        alpha_loss = hard_constraint_alpha_loss(
                            pred_alpha,
                            batch.target_alpha[0],
                            result.validation_boxes,
                            tau=float(loss_cfg.get("alpha_tau", 0.95)),
                            gamma=float(loss_cfg.get("alpha_gamma", 1.5)),
                        )
                        loss = loss + float(loss_cfg.get("lambda_alpha", 1.0)) * alpha_loss
                    if loss_cfg.get("enable_orth_loss"):
                        orth_loss = soft_orthogonality_loss(
                            pred_rgb,
                            batch.target_rgb[0],
                            batch.target_alpha[0],
                            result.validation_boxes,
                        )
                        loss = loss + float(loss_cfg.get("lambda_orth", 1.0)) * orth_loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                logs = {
                    "train/loss": loss.detach().float().item(),
                    "train/loss_fm": fm_loss.detach().float().item(),
                    "train/loss_alpha": alpha_loss.detach().float().item(),
                    "train/loss_orth": orth_loss.detach().float().item(),
                    "train/sigma": result.sigma.detach().float().mean().item(),
                    "train/lr": _current_lr(optimizer, lr_scheduler),
                }
                accelerator.log(logs, step=global_step)
                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "loss": f"{logs['train/loss']:.4f}",
                        "fm": f"{logs['train/loss_fm']:.4f}",
                        "alpha": f"{logs['train/loss_alpha']:.4f}",
                        "orth": f"{logs['train/loss_orth']:.4f}",
                        "lr": f"{logs['train/lr']:.3e}",
                    },
                    refresh=True,
                )
                if accelerator.is_main_process:
                    logger.info(
                        "step=%d/%d loss=%.6f loss_fm=%.6f loss_alpha=%.6f loss_orth=%.6f sigma=%.6f lr=%.6e",
                        global_step,
                        max_train_steps,
                        logs["train/loss"],
                        logs["train/loss_fm"],
                        logs["train/loss_alpha"],
                        logs["train/loss_orth"],
                        logs["train/sigma"],
                        logs["train/lr"],
                    )
                if global_step % int(train_cfg.get("checkpointing_steps", 1000)) == 0:
                    _save_checkpoint(accelerator, pipe, transparent_decoder, cfg, output_dir, global_step)
                if (
                    needs_transparent_decoder
                    and pred_rgb is not None
                    and preview_steps > 0
                    and global_step % preview_steps == 0
                    and accelerator.is_main_process
                ):
                    _save_visual_previews(output_dir, batch, pred_rgb, pred_alpha, result.validation_boxes, global_step, preview_dir)
                if global_step >= max_train_steps:
                    break

    progress_bar.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_checkpoint(accelerator, pipe, transparent_decoder, cfg, output_dir, global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
