from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.rcm_targets import area_target_and_conf, bbox_from_alpha, crop_grid_by_box, visible_alpha_from_order


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


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    try:
        import yaml

        with config_path.open("r", encoding="utf-8-sig") as f:
            return yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        return _load_simple_yaml(config_path)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        samples = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("samples", "data", "items"):
        if isinstance(data.get(key), list):
            return data[key]
    raise ValueError(f"Unsupported manifest structure: {path}")


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _sample_id(sample: dict[str, Any], index: int) -> str:
    return str(sample.get("sample_id") or sample.get("id") or sample.get("imgid") or f"sample_{index:06d}")


def _extract_layer_items(sample: dict[str, Any]) -> list[dict[str, str]]:
    raw_layers = (
        sample.get("layers")
        or sample.get("target_layers")
        or sample.get("rgba_layers")
        or sample.get("foreground_layers")
        or sample.get("LayerInfoRaw")
        or []
    )
    items: list[dict[str, str]] = []
    for idx, item in enumerate(raw_layers):
        if isinstance(item, dict):
            layer_id = str(item.get("layer_id") or item.get("id") or f"layer_{idx}")
            rgba = item.get("rgba") or item.get("path") or item.get("image")
            if not rgba:
                raise ValueError(f"Layer item {idx} is missing rgba/path/image: {item}")
            items.append({"layer_id": layer_id, "rgba": str(rgba)})
        else:
            items.append({"layer_id": f"layer_{idx}", "rgba": str(item)})
    return items


def _ordered_layer_items(sample: dict[str, Any], default_convention: str) -> tuple[list[dict[str, str]], str]:
    items = _extract_layer_items(sample)
    order_convention = str(sample.get("order_convention") or default_convention)
    if order_convention not in {"front_to_back", "back_to_front"}:
        raise ValueError("order_convention must be 'front_to_back' or 'back_to_front'")

    order = sample.get("order")
    if order:
        by_id = {item["layer_id"]: item for item in items}
        ordered = [by_id[str(layer_id)] for layer_id in order]
    else:
        ordered = items
    if order_convention == "back_to_front":
        ordered = list(reversed(ordered))
    return ordered, "front_to_back"


def _image_to_alpha(image: Image.Image) -> torch.Tensor:
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgba[..., 3])


def _maybe_resize(image: Image.Image, resolution: int | None) -> Image.Image:
    if not resolution or image.size == (resolution, resolution):
        return image
    return image.resize((resolution, resolution), Image.Resampling.BICUBIC)


def _source_image(sample: dict[str, Any], data_root: Path, resolution: int | None) -> Image.Image | None:
    for key in ("image", "full_image", "source_image"):
        if sample.get(key):
            return _maybe_resize(Image.open(_resolve_path(sample[key], data_root)).convert("RGB"), resolution)
    return None


def _colorize(values: np.ndarray) -> Image.Image:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    red = (255.0 * values).astype(np.uint8)
    green = (255.0 * (1.0 - np.abs(values - 0.5) * 2.0)).astype(np.uint8)
    blue = (255.0 * (1.0 - values)).astype(np.uint8)
    return Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")


def _save_heatmap(path: Path, values: torch.Tensor, size: tuple[int, int] | None = None) -> None:
    array = values.detach().float().cpu().numpy()
    image = _colorize(array)
    if size is not None:
        image = image.resize(size, Image.Resampling.NEAREST)
    image.save(path)


def _save_bbox_overlay(path: Path, source: Image.Image | None, box: list[int], color: str = "red") -> None:
    if source is None:
        width = max(int(box[2]), 16)
        height = max(int(box[3]), 16)
        image = Image.new("RGB", (width, height), (32, 32, 32))
    else:
        image = source.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle(tuple(box), outline=color, width=3)
    image.save(path)


def _compute_overlap_pairs(layer_ids: list[str], alpha_full: torch.Tensor) -> list[dict[str, Any]]:
    pairs = []
    for front_idx in range(alpha_full.shape[0]):
        for back_idx in range(front_idx + 1, alpha_full.shape[0]):
            front = alpha_full[front_idx]
            back = alpha_full[back_idx]
            soft_overlap = front * back
            soft_sum = float(soft_overlap.sum().item())
            if soft_sum <= 0:
                continue
            pairs.append(
                {
                    "front_layer": layer_ids[front_idx],
                    "back_layer": layer_ids[back_idx],
                    "soft_overlap_sum": soft_sum,
                    "overlap_area": int((soft_overlap > 0.01).sum().item()),
                    "back_occluded_sum": soft_sum,
                }
            )
    return pairs


def _write_sample(
    sample: dict[str, Any],
    index: int,
    data_root: Path,
    output_dir: Path,
    resolution: int | None,
    order_convention: str,
    alpha_threshold: float,
    bbox_padding: int,
    token_factor: int,
    save_visualization: bool,
    store_pixel_alpha: bool,
) -> dict[str, Any]:
    sample_id = _sample_id(sample, index)
    layer_items, normalized_convention = _ordered_layer_items(sample, order_convention)
    if not layer_items:
        raise ValueError(f"Sample {sample_id} has no RGBA foreground layers.")

    source = _source_image(sample, data_root, resolution)
    layer_ids = [item["layer_id"] for item in layer_items]
    rgba_images = [
        _maybe_resize(Image.open(_resolve_path(item["rgba"], data_root)).convert("RGBA"), resolution)
        for item in layer_items
    ]
    sizes = {image.size for image in rgba_images}
    if len(sizes) != 1:
        raise ValueError(f"Sample {sample_id} has mismatched layer sizes: {sorted(sizes)}")
    width, height = rgba_images[0].size
    alpha_full = torch.stack([_image_to_alpha(image) for image in rgba_images], dim=0).float()
    alpha_visible = visible_alpha_from_order(alpha_full.unsqueeze(0).unsqueeze(2), normalized_convention)[0, :, 0]
    alpha_occluded = (alpha_full - alpha_visible).clamp_min(0.0)

    bboxes: list[list[int]] = []
    target_grids = []
    conf_grids = []
    target_flats = []
    conf_flats = []
    kept_ids = []
    kept_full = []
    kept_visible = []
    kept_occluded = []
    for layer_idx, layer_id in enumerate(layer_ids):
        bbox = bbox_from_alpha(
            alpha_full[layer_idx],
            alpha_threshold=alpha_threshold,
            padding=bbox_padding,
            token_factor=token_factor,
        )
        if bbox is None:
            continue
        target_grid, conf_grid = area_target_and_conf(alpha_visible[layer_idx].unsqueeze(0).unsqueeze(0), token_factor)
        target_flat = crop_grid_by_box(target_grid, bbox, token_factor)
        conf_flat = crop_grid_by_box(conf_grid, bbox, token_factor)

        kept_ids.append(layer_id)
        bboxes.append(bbox)
        target_grids.append(target_grid[0, 0])
        conf_grids.append(conf_grid[0, 0])
        target_flats.append(target_flat[0])
        conf_flats.append(conf_flat[0])
        kept_full.append(alpha_full[layer_idx])
        kept_visible.append(alpha_visible[layer_idx])
        kept_occluded.append(alpha_occluded[layer_idx])

    if not bboxes:
        raise ValueError(f"Sample {sample_id} has no non-empty alpha layers.")

    target_grid_np = torch.stack(target_grids, dim=0).cpu().numpy().astype(np.float32)
    conf_grid_np = torch.stack(conf_grids, dim=0).cpu().numpy().astype(np.float32)
    max_flat_len = max(int(flat.numel()) for flat in target_flats)
    target_flat_np = np.zeros((len(target_flats), max_flat_len), dtype=np.float32)
    conf_flat_np = np.zeros((len(conf_flats), max_flat_len), dtype=np.float32)
    flat_lengths = np.zeros((len(target_flats),), dtype=np.int32)
    for row, (target_flat, conf_flat) in enumerate(zip(target_flats, conf_flats)):
        length = int(target_flat.numel())
        flat_lengths[row] = length
        target_flat_np[row, :length] = target_flat.cpu().numpy()
        conf_flat_np[row, :length] = conf_flat.cpu().numpy()

    overlap_pairs = _compute_overlap_pairs(kept_ids, torch.stack(kept_full, dim=0))
    npz_dir = output_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / f"{sample_id}.npz"
    save_data: dict[str, Any] = {
        "sample_id": np.array(sample_id),
        "layer_ids": np.array(kept_ids),
        "bboxes": np.asarray(bboxes, dtype=np.int32),
        "rcm_target_grid": target_grid_np,
        "rcm_conf_grid": conf_grid_np,
        "rcm_target_flat": target_flat_np,
        "rcm_conf_flat": conf_flat_np,
        "flat_lengths": flat_lengths,
        "order_front_to_back": np.array(kept_ids),
        "overlap_pairs_json": np.array(json.dumps(overlap_pairs, ensure_ascii=False)),
        "image_size": np.asarray([height, width], dtype=np.int32),
        "token_factor": np.asarray([token_factor], dtype=np.int32),
    }
    if store_pixel_alpha:
        save_data.update(
            {
                "alpha_full": torch.stack(kept_full, dim=0).cpu().numpy().astype(np.float32),
                "alpha_visible": torch.stack(kept_visible, dim=0).cpu().numpy().astype(np.float32),
                "alpha_occluded": torch.stack(kept_occluded, dim=0).cpu().numpy().astype(np.float32),
            }
        )
    np.savez_compressed(npz_path, **save_data)

    vis_dir = output_dir / "vis" / sample_id
    if save_visualization:
        vis_dir.mkdir(parents=True, exist_ok=True)
        if source is not None:
            source.save(vis_dir / "input.png")
        for idx, layer_id in enumerate(kept_ids):
            prefix = f"{idx:02d}_{layer_id}"
            _save_heatmap(vis_dir / f"{prefix}_full_alpha.png", kept_full[idx], size=(width, height))
            _save_heatmap(vis_dir / f"{prefix}_visible_alpha.png", kept_visible[idx], size=(width, height))
            _save_heatmap(vis_dir / f"{prefix}_occluded_alpha.png", kept_occluded[idx], size=(width, height))
            _save_heatmap(vis_dir / f"{prefix}_rcm_target.png", target_grids[idx], size=(width, height))
            _save_heatmap(vis_dir / f"{prefix}_rcm_conf.png", conf_grids[idx], size=(width, height))
            _save_bbox_overlay(vis_dir / f"{prefix}_bbox_overlay.png", source, bboxes[idx])

    return {
        "sample_id": sample_id,
        "npz": str(npz_path.relative_to(output_dir.parent)),
        "vis_dir": str(vis_dir.relative_to(output_dir.parent)) if save_visualization else None,
        "num_layers": len(kept_ids),
        "has_background_rcm": False,
        "order_convention": "front_to_back",
        "overlap_pairs": overlap_pairs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage-1 RCM supervision from RGBA layers.")
    parser.add_argument("--config", type=str, default="configs/build_rcm_stage1_supervision.yaml")
    parser.add_argument("--input_manifest", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--order_convention", choices=("front_to_back", "back_to_front"), default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--alpha_threshold", type=float, default=None)
    parser.add_argument("--bbox_padding", type=int, default=None)
    parser.add_argument("--token_factor", type=int, default=None)
    parser.add_argument("--save_visualization", action="store_true", default=None)
    parser.add_argument("--no_save_visualization", action="store_false", dest="save_visualization")
    parser.add_argument("--store_pixel_alpha", action="store_true", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        cfg[key] = value

    input_manifest = Path(cfg["input_manifest"]).resolve()
    data_root = Path(cfg.get("data_root") or input_manifest.parent).resolve()
    output_dir = Path(cfg.get("output_dir", "supervision/rcm_stage1")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _read_manifest(input_manifest)
    records = []
    for index, sample in enumerate(samples):
        records.append(
            _write_sample(
                sample=sample,
                index=index,
                data_root=data_root,
                output_dir=output_dir,
                resolution=cfg.get("resolution"),
                order_convention=str(cfg.get("order_convention", "front_to_back")),
                alpha_threshold=float(cfg.get("alpha_threshold", 0.01)),
                bbox_padding=int(cfg.get("bbox_padding", 8)),
                token_factor=int(cfg.get("token_factor", 16)),
                save_visualization=bool(cfg.get("save_visualization", True)),
                store_pixel_alpha=bool(cfg.get("store_pixel_alpha", False)),
            )
        )

    manifest_path = output_dir / "manifest_rcm_stage1.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} RCM supervision samples to {output_dir}")


if __name__ == "__main__":
    main()
