from __future__ import annotations

import argparse
import json
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


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "None", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        import yaml

        with config_path.open("r", encoding="utf-8-sig") as f:
            return yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        return _load_simple_yaml(config_path)


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        elif value is not None:
            base[key] = value
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage-1 RCM probe only.")
    parser.add_argument("--config", type=str, default="configs/train_rcm_stage1.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max_layers", type=int, default=None)
    parser.add_argument("--layer_order", choices=("as_is", "reverse"), default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=None)
    parser.add_argument("--preview_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include_background", action="store_true", default=None)
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
    for key in ("max_train_steps", "mixed_precision", "resume_from_checkpoint", "checkpointing_steps", "preview_steps", "seed"):
        value = getattr(args, key)
        if value is not None:
            training[key] = value
    if training:
        overrides["training"] = training
    optimizer = {}
    if args.learning_rate is not None:
        optimizer["learning_rate"] = args.learning_rate
    if optimizer:
        overrides["optimizer"] = optimizer
    if args.include_background is not None:
        overrides["rcm"] = {"include_background": args.include_background}
    return _deep_update(cfg, overrides)


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


def _dtype_from_mixed_precision(value: str):
    import torch

    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    return torch.float32


def _current_lr(optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _build_optimizer(params, cfg: dict[str, Any]):
    import torch

    opt_cfg = cfg.get("optimizer", {})
    lr = float(opt_cfg.get("learning_rate", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    betas = tuple(float(v) for v in opt_cfg.get("betas", (0.9, 0.999)))
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def _save_checkpoint(accelerator, transformer, output_dir: Path, step: int) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    module = accelerator.unwrap_model(transformer)
    state = {key: value.detach().cpu() for key, value in module.rcm_head.state_dict().items()}
    accelerator.save(state, checkpoint_dir / "rcm_head.pt")


def _rgb_tensor_to_pil(tensor):
    import numpy as np
    from PIL import Image

    tensor = ((tensor.detach().float().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _colorize(values):
    import numpy as np
    from PIL import Image

    values = values.detach().float().cpu().numpy()
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    red = (255.0 * values).astype(np.uint8)
    green = (255.0 * (1.0 - np.abs(values - 0.5) * 2.0)).astype(np.uint8)
    blue = (255.0 * (1.0 - values)).astype(np.uint8)
    return Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")


def _selected_logits_by_layer(rcm_output, targets):
    logits_by_index = {
        int(layer_idx): logits
        for layer_idx, logits in zip(rcm_output.layer_indices, rcm_output.logits_by_layer)
    }
    return [logits_by_index[int(layer_idx)] for layer_idx in targets.layer_indices]


def _save_preview(output_dir: Path, batch, targets, rcm_output, step: int, token_factor: int) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    from training.rcm_targets import flat_to_grid

    step_dir = output_dir / "previews" / f"step-{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    source = _rgb_tensor_to_pil(batch.pixel_values[0])
    source.save(step_dir / "input.png")
    image_size = source.size
    logits_by_layer = {
        int(layer_idx): logits.detach()
        for layer_idx, logits in zip(rcm_output.layer_indices, rcm_output.logits_by_layer)
    }

    pred_grids = []
    target_grids = []
    conf_grids = []
    layer_ids = []
    for item_idx, layer_idx in enumerate(targets.layer_indices):
        box = targets.boxes[item_idx]
        target_grid = targets.target_grid_by_layer[item_idx][0, 0].detach().cpu()
        conf_grid = targets.confidence_grid_by_layer[item_idx][0, 0].detach().cpu()
        pred_flat = logits_by_layer[int(layer_idx)].sigmoid().detach().cpu()
        pred_grid = flat_to_grid(pred_flat, box, target_grid.shape, token_factor=token_factor)
        error_grid = (pred_grid - target_grid).abs()

        prefix = f"layer_{int(layer_idx):02d}"
        _colorize(target_grid).resize(image_size, Image.Resampling.NEAREST).save(step_dir / f"{prefix}_target.png")
        _colorize(conf_grid).resize(image_size, Image.Resampling.NEAREST).save(step_dir / f"{prefix}_conf.png")
        _colorize(pred_grid).resize(image_size, Image.Resampling.NEAREST).save(step_dir / f"{prefix}_pred.png")
        _colorize(error_grid).resize(image_size, Image.Resampling.NEAREST).save(step_dir / f"{prefix}_error.png")

        heat = _colorize(pred_grid).resize(image_size, Image.Resampling.NEAREST)
        overlay_image = Image.blend(source.convert("RGB"), heat, 0.45)
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle(tuple(box), outline="red", width=3)
        overlay_image.save(step_dir / f"{prefix}_pred_overlay.png")

        pred_grids.append(pred_grid.float().numpy())
        target_grids.append(target_grid.float().numpy())
        conf_grids.append(conf_grid.float().numpy())
        layer_ids.append(f"layer_{int(layer_idx):02d}")

    np.savez_compressed(
        step_dir / "rcm_predictions.npz",
        layer_ids=np.asarray(layer_ids),
        bboxes=np.asarray(targets.boxes, dtype=np.int32),
        rcm_pred_grid=np.stack(pred_grids, axis=0).astype(np.float32) if pred_grids else np.zeros((0, 1, 1), dtype=np.float32),
        rcm_target_grid=np.stack(target_grids, axis=0).astype(np.float32) if target_grids else np.zeros((0, 1, 1), dtype=np.float32),
        rcm_conf_grid=np.stack(conf_grids, axis=0).astype(np.float32) if conf_grids else np.zeros((0, 1, 1), dtype=np.float32),
        image_size=np.asarray([batch.pixel_values.shape[-2], batch.pixel_values.shape[-1]], dtype=np.int32),
        token_factor=np.asarray([token_factor], dtype=np.int32),
    )


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

    from models.rcm import RCMHead, infer_clean_dim_from_refiner
    from train_reveallayer_lora import _build_pipeline
    from training.dataset import RevealLayerDataset, reveal_layer_collate
    from training.lora import freeze_module
    from training.rcm_targets import (
        build_rcm_targets_from_alpha,
        concat_logits_for_targets,
        rcm_metrics,
        total_variation_from_logits,
        weighted_bce_with_logits,
    )
    from training.train_step import forward_flow_matching

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    rcm_cfg = cfg.get("rcm", {})
    token_factor = int(rcm_cfg.get("token_factor", 16))
    include_background = bool(rcm_cfg.get("include_background", False))

    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
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
    freeze_module(pipe.transformer)
    transformer_module = pipe.transformer.module if hasattr(pipe.transformer, "module") else pipe.transformer
    hidden_dim = int(transformer_module.inner_dim)
    clean_dim = int(rcm_cfg.get("clean_dim") or infer_clean_dim_from_refiner(transformer_module.refiner, hidden_dim))
    rcm_head = RCMHead(
        hidden_dim=hidden_dim,
        clean_dim=clean_dim,
        rcm_dim=int(rcm_cfg.get("rcm_dim", 512)),
        time_dim=hidden_dim,
        max_layers=int(data_cfg.get("max_layers", 12)),
        token_factor=token_factor,
    ).to(device=accelerator.device, dtype=torch.float32)
    checkpoint = rcm_cfg.get("checkpoint")
    if checkpoint:
        rcm_head.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    transformer_module.set_rcm_probe(rcm_head, enabled=True, include_background=include_background)
    for param in rcm_head.parameters():
        param.requires_grad_(True)

    optimizer = _build_optimizer(rcm_head.parameters(), cfg)
    pipe.transformer, optimizer, dataloader = accelerator.prepare(pipe.transformer, optimizer, dataloader)
    module = accelerator.unwrap_model(pipe.transformer)
    module.eval()
    module.rcm_head.train()

    resume = train_cfg.get("resume_from_checkpoint")
    global_step = 0
    if resume:
        if resume == "latest":
            checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            resume = str(checkpoints[-1]) if checkpoints else None
        if resume:
            state_path = Path(resume) / "rcm_head.pt"
            module.rcm_head.load_state_dict(torch.load(state_path, map_location=accelerator.device), strict=True)
            global_step = int(Path(resume).name.split("-")[-1])
            logger.info("Resumed RCM head from %s", state_path)

    accelerator.init_trackers("reveallayer_rcm_stage1", config=_flatten_for_tracker(cfg))
    max_train_steps = int(train_cfg.get("max_train_steps", 10000))
    preview_steps = int(train_cfg.get("preview_steps", 500))
    checkpointing_steps = int(train_cfg.get("checkpointing_steps", 1000))
    progress_bar = tqdm(
        total=max_train_steps,
        initial=global_step,
        desc="rcm-stage1",
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
                    num_train_timesteps=int(train_cfg.get("num_train_timesteps", 1000)),
                    min_timestep_boundary=float(train_cfg.get("min_timestep_boundary", 0.0)),
                    max_timestep_boundary=float(train_cfg.get("max_timestep_boundary", 1.0)),
                    max_sequence_length=int(train_cfg.get("max_sequence_length", 512)),
                    guidance_scale=float(train_cfg.get("guidance_scale", 1.0)),
                )
                module = accelerator.unwrap_model(pipe.transformer)
                rcm_output = module.last_rcm_outputs
                if rcm_output is None:
                    raise RuntimeError("RCM probe did not produce outputs. Check include_background and valid foreground layers.")
                targets = build_rcm_targets_from_alpha(
                    batch.target_alpha[0],
                    result.validation_boxes,
                    include_background=include_background,
                    order_convention=str(rcm_cfg.get("order_convention", "front_to_back")),
                    token_factor=token_factor,
                )
                logits = concat_logits_for_targets(rcm_output, targets)
                ref_loss = weighted_bce_with_logits(logits, targets.target, targets.confidence)
                selected_logits = _selected_logits_by_layer(rcm_output, targets)
                tv_loss = total_variation_from_logits(selected_logits, targets.boxes, token_factor=token_factor)
                loss = ref_loss + float(rcm_cfg.get("tv_weight", 0.0)) * tv_loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                metrics = rcm_metrics(logits.detach(), targets.target.detach(), targets.confidence.detach())
                logs = {
                    "train/loss": loss.detach().float().item(),
                    "train/loss_ref": ref_loss.detach().float().item(),
                    "train/loss_tv": tv_loss.detach().float().item(),
                    "train/mae": metrics["mae"].detach().float().item(),
                    "train/high_conf_mae": metrics["high_conf_mae"].detach().float().item(),
                    "train/visible_recall": metrics["visible_recall"].detach().float().item(),
                    "train/unreliable_suppression": metrics["unreliable_suppression"].detach().float().item(),
                    "train/boundary_calibration": metrics["boundary_calibration"].detach().float().item(),
                    "train/lr": _current_lr(optimizer),
                }
                accelerator.log(logs, step=global_step)
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{logs['train/loss']:.4f}", "mae": f"{logs['train/mae']:.4f}"})
                if accelerator.is_main_process:
                    logger.info(
                        "step=%d/%d loss=%.6f ref=%.6f mae=%.6f high_conf_mae=%.6f",
                        global_step,
                        max_train_steps,
                        logs["train/loss"],
                        logs["train/loss_ref"],
                        logs["train/mae"],
                        logs["train/high_conf_mae"],
                    )
                if checkpointing_steps > 0 and global_step % checkpointing_steps == 0 and accelerator.is_main_process:
                    _save_checkpoint(accelerator, pipe.transformer, output_dir, global_step)
                if preview_steps > 0 and global_step % preview_steps == 0 and accelerator.is_main_process:
                    _save_preview(output_dir, batch, targets, rcm_output, global_step, token_factor)
                if global_step >= max_train_steps:
                    break

    progress_bar.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        _save_checkpoint(accelerator, pipe.transformer, output_dir, global_step)
    accelerator.end_training()


if __name__ == "__main__":
    main()
