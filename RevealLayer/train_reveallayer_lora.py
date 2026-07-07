import argparse
import math
import os
import shutil
from types import SimpleNamespace
from typing import Any, Dict, Optional

logger = None


def _load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Reading YAML configs requires pyyaml. Install pyyaml or pass all options by CLI.") from exc
    with open(path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Training config must be a YAML mapping: {path}")
    return data


def _cfg_get(args, config: Dict[str, Any], key: str, default=None):
    value = getattr(args, key, None)
    if value is not None:
        return value
    return config.get(key, default)


def _namespace_from_args(args, config: Dict[str, Any]) -> SimpleNamespace:
    keys = {
        "pretrained_model_name_or_path": None,
        "cfg_path": "./configs/ld_resolution1024_test.py",
        "ckpt_dir": "./ckpts/RevealLayer",
        "train_manifest": None,
        "data_root": None,
        "output_dir": "./output/reveallayer_lora",
        "logging_dir": "logs",
        "resolution": 1024,
        "max_layer": 12,
        "prompt": "Decompose the image into foreground and background.",
        "rank": 64,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "train_layer_pe": True,
        "train_refiner": True,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "max_train_steps": 50000,
        "learning_rate": 1.0,
        "optimizer": "prodigy",
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "max_grad_norm": 1.0,
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "allow_tf32": True,
        "lambda_alpha": 1.0,
        "lambda_orth": 1.0,
        "enable_alpha_loss": False,
        "enable_orth_loss": False,
        "checkpointing_steps": 2000,
        "checkpoints_total_limit": 5,
        "resume_from_checkpoint": None,
        "report_to": "tensorboard",
        "seed": 42,
        "dataloader_num_workers": 4,
        "max_sequence_length": 512,
        "guidance_scale": 1.0,
    }
    values = {key: _cfg_get(args, config, key, default) for key, default in keys.items()}
    return SimpleNamespace(**values)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RevealLayer LoRA adapters with Accelerate.")
    parser.add_argument("--config", type=str, default="./configs/train_reveallayer_lora.yaml")
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--train_manifest", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--logging_dir", type=str)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--max_layer", type=int)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_dropout", type=float)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--max_train_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--optimizer", choices=["adamw", "prodigy"])
    parser.add_argument("--lr_scheduler", type=str)
    parser.add_argument("--lr_warmup_steps", type=int)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int)
    parser.add_argument("--checkpoints_total_limit", type=int)
    parser.add_argument("--resume_from_checkpoint", type=str)
    parser.add_argument("--report_to", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataloader_num_workers", type=int)
    parser.add_argument("--guidance_scale", type=float)
    parser.add_argument("--disable_gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--no_train_layer_pe", action="store_true")
    parser.add_argument("--no_train_refiner", action="store_true")
    return parser.parse_args()


def build_transformer(config, train_args: SimpleNamespace, dtype):
    from diffusers import FluxTransformer2DModel
    from diffusers.configuration_utils import FrozenDict
    from infer import _load_layer_pe, _load_refiner
    from models.custom_model_mmdit import CustomFluxTransformer2DModel

    transformer_orig = FluxTransformer2DModel.from_pretrained(
        config.transformer_varient if hasattr(config, "transformer_varient") else train_args.pretrained_model_name_or_path,
        subfolder="" if hasattr(config, "transformer_varient") else "transformer",
        revision=config.revision,
        variant=config.variant,
        torch_dtype=dtype,
        cache_dir=config.get("cache_dir", None),
    )
    mmdit_config = dict(transformer_orig.config)
    mmdit_config["_class_name"] = "CustomSD3Transformer2DModel"
    mmdit_config["max_layer_num"] = train_args.max_layer
    transformer = CustomFluxTransformer2DModel.from_config(FrozenDict(mmdit_config)).to(dtype=dtype)
    transformer.load_state_dict(transformer_orig.state_dict(), strict=False)
    _load_layer_pe(transformer, train_args.ckpt_dir)
    _load_refiner(transformer, train_args.ckpt_dir)
    del transformer_orig
    return transformer


def build_pipeline(train_args: SimpleNamespace, accelerator):
    from diffusers import AutoencoderKL
    from infer import parse_config
    from models.custom_pipeline import CustomFluxPipelineCfg

    config = parse_config(train_args.cfg_path)
    dtype = torch.bfloat16 if train_args.mixed_precision == "bf16" else torch.float16 if train_args.mixed_precision == "fp16" else torch.float32
    transformer = build_transformer(config, train_args, dtype)
    vae = AutoencoderKL.from_pretrained(
        train_args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=None,
        variant=None,
        torch_dtype=dtype,
        cache_dir=config.get("cache_dir", None),
    )
    pipeline = CustomFluxPipelineCfg.from_pretrained(
        train_args.pretrained_model_name_or_path,
        transformer=transformer,
        vae=vae,
        revision=config.revision,
        variant=config.variant,
        torch_dtype=dtype,
        cache_dir=config.get("cache_dir", None),
    )
    pipeline.to(accelerator.device)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    return pipeline


def build_optimizer(train_args: SimpleNamespace, params):
    import torch

    if train_args.optimizer == "prodigy":
        try:
            import prodigyopt
        except ImportError as exc:
            raise ImportError("optimizer=prodigy requires prodigyopt. Install prodigyopt or use --optimizer adamw.") from exc
        return prodigyopt.Prodigy(params, lr=train_args.learning_rate)
    return torch.optim.AdamW(params, lr=train_args.learning_rate)


def rotate_checkpoints(output_dir: str, total_limit: Optional[int]):
    if not total_limit:
        return
    checkpoints = [
        path for path in os.listdir(output_dir)
        if path.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, path))
    ]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    for ckpt in checkpoints[: max(0, len(checkpoints) - total_limit)]:
        shutil.rmtree(os.path.join(output_dir, ckpt), ignore_errors=True)


def main():
    raw_args = parse_args()

    import torch
    from accelerate import Accelerator
    from accelerate.logging import get_logger
    from accelerate.utils import ProjectConfiguration, set_seed
    from diffusers.optimization import get_scheduler
    from torch.utils.data import DataLoader

    from training.dataset import RevealLayerDataset, collate_reveallayer
    from training.lora import (
        add_lora_to_transformer,
        mark_extra_trainable,
        save_trainable_state_dict,
        save_extra_trainable_state_dict,
        trainable_parameter_summary,
        trainable_parameters,
    )
    from training.losses import compose_losses, flow_matching_loss
    from training.train_step import forward_flow_matching

    global logger
    logger = get_logger(__name__)

    config = _load_yaml_config(raw_args.config)
    train_args = _namespace_from_args(raw_args, config)
    if raw_args.disable_gradient_checkpointing:
        train_args.gradient_checkpointing = False
    if raw_args.disable_tf32:
        train_args.allow_tf32 = False
    if raw_args.no_train_layer_pe:
        train_args.train_layer_pe = False
    if raw_args.no_train_refiner:
        train_args.train_refiner = False
    if not train_args.pretrained_model_name_or_path:
        raise ValueError("--pretrained_model_name_or_path is required")
    if not train_args.train_manifest:
        raise ValueError("--train_manifest is required")
    if train_args.train_batch_size != 1:
        raise ValueError("Current custom transformer path requires --train_batch_size 1; use gradient accumulation for global batch.")

    if train_args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    project_config = ProjectConfiguration(project_dir=train_args.output_dir, logging_dir=os.path.join(train_args.output_dir, train_args.logging_dir))
    accelerator = Accelerator(
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        mixed_precision=train_args.mixed_precision,
        log_with=train_args.report_to,
        project_config=project_config,
    )
    if train_args.seed is not None:
        set_seed(train_args.seed)

    os.makedirs(train_args.output_dir, exist_ok=True)
    dataset = RevealLayerDataset(
        manifest_path=train_args.train_manifest,
        data_root=train_args.data_root,
        resolution=train_args.resolution,
        max_layers=train_args.max_layer,
        prompt=train_args.prompt,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_args.train_batch_size,
        shuffle=True,
        num_workers=train_args.dataloader_num_workers,
        collate_fn=collate_reveallayer,
    )

    pipeline = build_pipeline(train_args, accelerator)
    if train_args.gradient_checkpointing:
        pipeline.transformer.gradient_checkpointing = True
    targets = add_lora_to_transformer(
        pipeline.transformer,
        rank=train_args.rank,
        alpha=train_args.lora_alpha,
        dropout=train_args.lora_dropout,
    )
    extras = mark_extra_trainable(
        pipeline.transformer,
        train_layer_pe=train_args.train_layer_pe,
        train_refiner=train_args.train_refiner,
    )
    logger.info(f"LoRA targets: {targets}")
    logger.info(f"Extra trainable modules: {extras}")
    logger.info(trainable_parameter_summary(pipeline.transformer))

    optimizer = build_optimizer(train_args, trainable_parameters(pipeline.transformer))
    updates_per_epoch = math.ceil(len(dataloader) / train_args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(train_args.max_train_steps / max(1, updates_per_epoch))
    lr_scheduler = get_scheduler(
        train_args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=train_args.max_train_steps * accelerator.num_processes,
    )

    pipeline.transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        pipeline.transformer, optimizer, dataloader, lr_scheduler
    )
    pipeline.transformer.train()
    accelerator.init_trackers("reveallayer_lora")

    global_step = 0
    if train_args.resume_from_checkpoint:
        resume = train_args.resume_from_checkpoint
        if resume == "latest":
            checkpoints = [
                path for path in os.listdir(train_args.output_dir)
                if path.startswith("checkpoint-") and os.path.isdir(os.path.join(train_args.output_dir, path))
            ]
            resume = os.path.join(train_args.output_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]) if checkpoints else None
        if resume:
            accelerator.load_state(resume)
            global_step = int(os.path.basename(resume).split("-")[-1])
            logger.info(f"Resumed from {resume}")

    for _epoch in range(num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(pipeline.transformer):
                outputs = forward_flow_matching(
                    pipeline,
                    batch,
                    max_sequence_length=train_args.max_sequence_length,
                    guidance_scale=train_args.guidance_scale,
                )
                fm_loss = flow_matching_loss(outputs["pred_velocity"], outputs["target_velocity"])
                losses = compose_losses(
                    fm_loss,
                    alpha_loss=None,
                    orth_loss=None,
                    alpha_weight=train_args.lambda_alpha,
                    orth_weight=train_args.lambda_orth,
                )
                loss = losses["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_parameters(pipeline.transformer), train_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                logs = {key: value.item() for key, value in losses.items() if key != "loss"}
                logs["loss"] = loss.detach().item()
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                accelerator.log(logs, step=global_step)

                if global_step % train_args.checkpointing_steps == 0:
                    checkpoint_dir = os.path.join(train_args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(checkpoint_dir)
                    unwrapped = accelerator.unwrap_model(pipeline.transformer)
                    save_trainable_state_dict(unwrapped, os.path.join(checkpoint_dir, "pytorch_lora_weights.safetensors"))
                    save_extra_trainable_state_dict(unwrapped, os.path.join(checkpoint_dir, "reveallayer_extra_trainables.safetensors"))
                    rotate_checkpoints(train_args.output_dir, train_args.checkpoints_total_limit)

                if global_step >= train_args.max_train_steps:
                    break
        if global_step >= train_args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(pipeline.transformer)
        save_trainable_state_dict(unwrapped, os.path.join(train_args.output_dir, "pytorch_lora_weights.safetensors"))
        save_extra_trainable_state_dict(unwrapped, os.path.join(train_args.output_dir, "reveallayer_extra_trainables.safetensors"))
    accelerator.end_training()


if __name__ == "__main__":
    main()
