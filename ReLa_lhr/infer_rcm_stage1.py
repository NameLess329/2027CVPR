from __future__ import annotations

import argparse
import csv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-1 RCM inference/evaluation.")
    parser.add_argument("--config", type=str, default="configs/infer_rcm_stage1.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--max_layers", type=int, default=None)
    parser.add_argument("--layer_order", choices=("as_is", "reverse"), default=None)
    parser.add_argument("--rcm_checkpoint", type=str, default=None)
    parser.add_argument("--tap_block", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--include_background", action="store_true", default=None)
    return parser.parse_args()


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        elif value is not None:
            base[key] = value
    return base


def _config_with_cli(args: argparse.Namespace) -> dict[str, Any]:
    from train_rcm_stage1 import _load_config

    cfg = _load_config(args.config)
    overrides: dict[str, Any] = {}
    for key in ("model_path", "ckpt_dir", "output_dir"):
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
    inference = {}
    for key in ("max_samples", "mixed_precision", "seed"):
        value = getattr(args, key)
        if value is not None:
            inference[key] = value
    if inference:
        overrides["inference"] = inference
    rcm = {}
    if args.rcm_checkpoint is not None:
        rcm["checkpoint"] = args.rcm_checkpoint
    if args.tap_block is not None:
        rcm["tap_block"] = args.tap_block
    if args.include_background is not None:
        rcm["include_background"] = args.include_background
    if rcm:
        overrides["rcm"] = rcm
    return _deep_update(cfg, overrides)


def _resolve_rcm_checkpoint(rcm_cfg: dict[str, Any]) -> Path:
    checkpoint = rcm_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("Set rcm.checkpoint to a rcm_head.pt path or to 'latest'.")
    if str(checkpoint).lower() != "latest":
        path = Path(checkpoint).resolve()
        if not path.exists():
            raise FileNotFoundError(f"RCM checkpoint not found: {path}")
        return path

    train_output_dir = Path(rcm_cfg.get("train_output_dir", "outputs/rcm_stage1")).resolve()
    checkpoints = sorted(
        train_output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else -1,
    )
    for checkpoint_dir in reversed(checkpoints):
        candidate = checkpoint_dir / "rcm_head.pt"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No rcm_head.pt found under {train_output_dir}")


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


def _rgb_tensor_to_pil(tensor):
    import numpy as np
    from PIL import Image

    tensor = ((tensor.detach().float().cpu() + 1.0) / 2.0).clamp(0.0, 1.0)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _save_sample_outputs(
    sample_dir: Path,
    batch: Any,
    targets: Any,
    rcm_output: Any,
    sample_metrics: dict[str, float],
    token_factor: int,
    save_visualization: bool,
    save_npz: bool,
    tap_block: str,
) -> list[dict[str, Any]]:
    import numpy as np
    from PIL import Image, ImageDraw

    from training.rcm_targets import flat_to_grid, rcm_metrics

    sample_dir.mkdir(parents=True, exist_ok=True)
    source = _rgb_tensor_to_pil(batch.pixel_values[0].detach().cpu())
    image_size = source.size
    if save_visualization:
        source.save(sample_dir / "input.png")

    logits_by_layer = {
        int(layer_idx): logits.detach()
        for layer_idx, logits in zip(rcm_output.layer_indices, rcm_output.logits_by_layer)
    }

    pred_grids = []
    target_grids = []
    conf_grids = []
    bboxes = []
    layer_ids = []
    layer_records: list[dict[str, Any]] = []

    for item_idx, layer_idx in enumerate(targets.layer_indices):
        box = targets.boxes[item_idx]
        target_grid = targets.target_grid_by_layer[item_idx][0, 0].detach().cpu()
        conf_grid = targets.confidence_grid_by_layer[item_idx][0, 0].detach().cpu()
        logits = logits_by_layer[int(layer_idx)].detach().cpu()
        pred_flat = logits.sigmoid()
        pred_grid = flat_to_grid(pred_flat, box, target_grid.shape, token_factor=token_factor)
        error_grid = (pred_grid.float() - target_grid.float()).abs()
        metrics = rcm_metrics(logits.float(), targets.target_by_layer[item_idx].detach().cpu().float(), targets.confidence_by_layer[item_idx].detach().cpu().float())
        record = {
            "sample_id": batch.sample_ids[0],
            "layer_id": int(layer_idx),
            "tap_block": str(tap_block),
            "bbox": box,
            "mae": float(metrics["mae"].item()),
            "high_conf_mae": float(metrics["high_conf_mae"].item()),
            "visible_recall": float(metrics["visible_recall"].item()),
            "unreliable_suppression": float(metrics["unreliable_suppression"].item()),
            "boundary_calibration": float(metrics["boundary_calibration"].item()),
        }
        layer_records.append(record)

        prefix = f"layer_{int(layer_idx):02d}"
        if save_visualization:
            _colorize(target_grid).resize(image_size, Image.Resampling.NEAREST).save(sample_dir / f"{prefix}_target.png")
            _colorize(conf_grid).resize(image_size, Image.Resampling.NEAREST).save(sample_dir / f"{prefix}_conf.png")
            _colorize(pred_grid).resize(image_size, Image.Resampling.NEAREST).save(sample_dir / f"{prefix}_pred.png")
            _colorize(error_grid).resize(image_size, Image.Resampling.NEAREST).save(sample_dir / f"{prefix}_error.png")
            heat = _colorize(pred_grid).resize(image_size, Image.Resampling.NEAREST)
            overlay = Image.blend(source.convert("RGB"), heat, 0.45)
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(tuple(box), outline="red", width=3)
            overlay.save(sample_dir / f"{prefix}_pred_overlay.png")

        pred_grids.append(pred_grid.float().numpy())
        target_grids.append(target_grid.float().numpy())
        conf_grids.append(conf_grid.float().numpy())
        bboxes.append(box)
        layer_ids.append(prefix)

    if save_npz:
        np.savez_compressed(
            sample_dir / "rcm_predictions.npz",
            sample_id=np.asarray(batch.sample_ids[0]),
            layer_ids=np.asarray(layer_ids),
            bboxes=np.asarray(bboxes, dtype=np.int32),
            rcm_pred_grid=np.stack(pred_grids, axis=0).astype(np.float32) if pred_grids else np.zeros((0, 1, 1), dtype=np.float32),
            rcm_target_grid=np.stack(target_grids, axis=0).astype(np.float32) if target_grids else np.zeros((0, 1, 1), dtype=np.float32),
            rcm_conf_grid=np.stack(conf_grids, axis=0).astype(np.float32) if conf_grids else np.zeros((0, 1, 1), dtype=np.float32),
            image_size=np.asarray([batch.pixel_values.shape[-2], batch.pixel_values.shape[-1]], dtype=np.int32),
            token_factor=np.asarray([token_factor], dtype=np.int32),
        )

    with (sample_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"sample": sample_metrics, "layers": layer_records}, f, ensure_ascii=False, indent=2)
    return layer_records


def _mean_records(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["mae", "high_conf_mae", "visible_recall", "unreliable_suppression", "boundary_calibration"]
    if not records:
        return {key: 0.0 for key in keys}
    return {
        key: float(sum(float(record[key]) for record in records) / len(records))
        for key in keys
    }


def _safe_name(value: str) -> str:
    keep = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("._") or "sample"


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    cfg = _config_with_cli(args)

    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from models.rcm import RCMHead, infer_clean_dim_from_refiner
    from train_rcm_stage1 import _dtype_from_mixed_precision
    from train_reveallayer_lora import _build_pipeline
    from training.dataset import RevealLayerDataset, reveal_layer_collate
    from training.lora import freeze_module
    from training.rcm_targets import build_rcm_targets_from_alpha, concat_logits_for_targets, rcm_metrics
    from training.train_step import forward_flow_matching

    data_cfg = cfg["data"]
    infer_cfg = cfg.get("inference", {})
    rcm_cfg = cfg.get("rcm", {})
    output_dir = Path(cfg.get("output_dir", "outputs/rcm_stage1_eval")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    seed = infer_cfg.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype_from_mixed_precision(str(infer_cfg.get("mixed_precision", "bf16")))
    token_factor = int(rcm_cfg.get("token_factor", 16))
    include_background = bool(rcm_cfg.get("include_background", False))

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
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=reveal_layer_collate,
        pin_memory=torch.cuda.is_available(),
    )

    pipe = _build_pipeline(cfg, device, dtype)
    freeze_module(pipe.transformer)
    transformer = pipe.transformer.module if hasattr(pipe.transformer, "module") else pipe.transformer
    hidden_dim = int(transformer.inner_dim)
    clean_dim = int(rcm_cfg.get("clean_dim") or infer_clean_dim_from_refiner(transformer.refiner, hidden_dim))
    rcm_head = RCMHead(
        hidden_dim=hidden_dim,
        clean_dim=clean_dim,
        rcm_dim=int(rcm_cfg.get("rcm_dim", 512)),
        time_dim=hidden_dim,
        max_layers=int(data_cfg.get("max_layers", 12)),
        token_factor=token_factor,
    ).to(device=device, dtype=torch.float32)
    checkpoint_path = _resolve_rcm_checkpoint(rcm_cfg)
    rcm_head.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    tap_block = rcm_cfg.get("tap_block", "last")
    transformer.set_rcm_probe(rcm_head, enabled=True, include_background=include_background, tap_block=tap_block)
    transformer.eval()
    transformer.rcm_head.eval()

    max_samples = infer_cfg.get("max_samples")
    max_samples = int(max_samples) if max_samples is not None else None
    sample_records: list[dict[str, Any]] = []
    layer_records: list[dict[str, Any]] = []

    total = min(len(dataset), max_samples) if max_samples is not None else len(dataset)
    progress = tqdm(total=total, desc="rcm-infer", dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_samples is not None and batch_idx >= max_samples:
                break
            batch.pixel_values = batch.pixel_values.to(device)
            batch.target_rgb = batch.target_rgb.to(device)
            batch.target_alpha = batch.target_alpha.to(device)
            result = forward_flow_matching(
                pipe,
                batch,
                num_train_timesteps=int(infer_cfg.get("num_train_timesteps", 1000)),
                min_timestep_boundary=float(infer_cfg.get("min_timestep_boundary", 0.0)),
                max_timestep_boundary=float(infer_cfg.get("max_timestep_boundary", 1.0)),
                max_sequence_length=int(infer_cfg.get("max_sequence_length", 512)),
                guidance_scale=float(infer_cfg.get("guidance_scale", 1.0)),
            )
            rcm_output = transformer.last_rcm_outputs
            if rcm_output is None:
                raise RuntimeError("RCM probe did not produce outputs. Check valid foreground layers and include_background.")
            targets = build_rcm_targets_from_alpha(
                batch.target_alpha[0],
                result.validation_boxes,
                include_background=include_background,
                order_convention=str(rcm_cfg.get("order_convention", "front_to_back")),
                token_factor=token_factor,
            )
            logits = concat_logits_for_targets(rcm_output, targets)
            metrics = rcm_metrics(logits.detach(), targets.target.detach(), targets.confidence.detach())
            sample_record = {
                "sample_id": batch.sample_ids[0],
                "tap_block": str(transformer.last_rcm_tap_block),
                "mae": float(metrics["mae"].item()),
                "high_conf_mae": float(metrics["high_conf_mae"].item()),
                "visible_recall": float(metrics["visible_recall"].item()),
                "unreliable_suppression": float(metrics["unreliable_suppression"].item()),
                "boundary_calibration": float(metrics["boundary_calibration"].item()),
            }
            sample_records.append(sample_record)

            sample_dir = output_dir / f"{batch_idx:06d}_{_safe_name(batch.sample_ids[0])}"
            new_layer_records = _save_sample_outputs(
                sample_dir=sample_dir,
                batch=batch,
                targets=targets,
                rcm_output=rcm_output,
                sample_metrics=sample_record,
                token_factor=token_factor,
                save_visualization=bool(infer_cfg.get("save_visualization", True)),
                save_npz=bool(infer_cfg.get("save_npz", True)),
                tap_block=str(transformer.last_rcm_tap_block),
            )
            layer_records.extend(new_layer_records)
            progress.update(1)
            progress.set_postfix({"mae": f"{sample_record['mae']:.4f}"})
    progress.close()

    summary = {
        "checkpoint": str(checkpoint_path),
        "tap_block": str(tap_block),
        "num_samples": len(sample_records),
        "num_layers": len(layer_records),
        "sample_mean": _mean_records(sample_records),
        "layer_mean": _mean_records(layer_records),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": sample_records, "layers": layer_records}, f, ensure_ascii=False, indent=2)

    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["level", "sample_id", "layer_id", "tap_block", "mae", "high_conf_mae", "visible_recall", "unreliable_suppression", "boundary_calibration"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for record in sample_records:
            writer.writerow({"level": "sample", "layer_id": "", **record})
        for record in layer_records:
            writer.writerow({"level": "layer", **record})

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
