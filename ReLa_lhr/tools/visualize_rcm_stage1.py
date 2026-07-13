from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _colorize(values: np.ndarray) -> Image.Image:
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    red = (255.0 * values).astype(np.uint8)
    green = (255.0 * (1.0 - np.abs(values - 0.5) * 2.0)).astype(np.uint8)
    blue = (255.0 * (1.0 - values)).astype(np.uint8)
    return Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")


def _save_map(path: Path, values: np.ndarray, image_size: tuple[int, int] | None) -> None:
    image = _colorize(values)
    if image_size is not None:
        image = image.resize(image_size, Image.Resampling.NEAREST)
    image.save(path)


def _flat_to_grid(flat: np.ndarray, box: np.ndarray, grid_shape: tuple[int, int], token_factor: int) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in box]
    lx1, ly1, lx2, ly2 = x1 // token_factor, y1 // token_factor, x2 // token_factor, y2 // token_factor
    crop_h = max(ly2 - ly1, 0)
    crop_w = max(lx2 - lx1, 0)
    expected = crop_h * crop_w
    if flat.shape[0] < expected:
        raise ValueError(f"flat prediction is shorter than bbox area: {flat.shape[0]} < {expected}")
    grid = np.zeros(grid_shape, dtype=np.float32)
    grid[ly1:ly2, lx1:lx2] = flat[:expected].reshape(crop_h, crop_w)
    return grid


def _prediction_grids(data: np.lib.npyio.NpzFile, bboxes: np.ndarray, grid_shape: tuple[int, int], token_factor: int) -> np.ndarray | None:
    if "rcm_pred_grid" in data:
        return data["rcm_pred_grid"].astype(np.float32)
    if "rcm_logits_grid" in data:
        return 1.0 / (1.0 + np.exp(-data["rcm_logits_grid"].astype(np.float32)))
    if "rcm_pred_flat" in data:
        return np.stack(
            [_flat_to_grid(data["rcm_pred_flat"][idx], bboxes[idx], grid_shape, token_factor) for idx in range(len(bboxes))],
            axis=0,
        )
    if "rcm_logits_flat" in data:
        probs = 1.0 / (1.0 + np.exp(-data["rcm_logits_flat"].astype(np.float32)))
        return np.stack([_flat_to_grid(probs[idx], bboxes[idx], grid_shape, token_factor) for idx in range(len(bboxes))], axis=0)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize stage-1 RCM npz files.")
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)

    target = data["rcm_target_grid"].astype(np.float32)
    conf = data["rcm_conf_grid"].astype(np.float32)
    bboxes = data["bboxes"]
    layer_ids = [str(item) for item in data["layer_ids"]]
    token_factor = int(data["token_factor"][0]) if "token_factor" in data else 16
    image_size = None
    source = None
    if args.image:
        source = Image.open(args.image).convert("RGB")
        image_size = source.size
        source.save(output_dir / "input.png")
    elif "image_size" in data:
        height, width = [int(v) for v in data["image_size"]]
        image_size = (width, height)

    pred = _prediction_grids(data, bboxes, target.shape[-2:], token_factor)
    for idx, layer_id in enumerate(layer_ids):
        prefix = f"{idx:02d}_{layer_id}"
        _save_map(output_dir / f"{prefix}_target.png", target[idx], image_size)
        _save_map(output_dir / f"{prefix}_conf.png", conf[idx], image_size)
        if pred is not None:
            _save_map(output_dir / f"{prefix}_pred.png", pred[idx], image_size)
            _save_map(output_dir / f"{prefix}_error.png", np.abs(pred[idx] - target[idx]), image_size)
        if source is not None:
            overlay = source.copy()
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(tuple(int(v) for v in bboxes[idx]), outline="red", width=3)
            overlay.save(output_dir / f"{prefix}_bbox_overlay.png")
    print(f"Wrote RCM visualization to {output_dir}")


if __name__ == "__main__":
    main()
