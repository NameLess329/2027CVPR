from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class RevealLayerBatch:
    pixel_values: torch.Tensor
    target_rgb: torch.Tensor
    target_alpha: torch.Tensor
    validation_boxes: list[list[list[int] | None]]
    prompts: list[str]
    sample_ids: list[str]


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

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


def _extract_boxes(sample: dict[str, Any]) -> list[list[int]]:
    boxes = sample.get("layout") or sample.get("bboxes") or sample.get("boxes") or sample.get("foreground_boxes")
    if boxes is None and sample.get("detections") is not None:
        boxes = [item.get("bbox") for item in sample["detections"] if item.get("bbox") is not None]
    if boxes is None:
        boxes = []
    result = []
    for box in boxes:
        if isinstance(box, dict):
            box = [box["x1"], box["y1"], box["x2"], box["y2"]]
        result.append([int(round(v)) for v in box])
    return result


def _extract_layer_paths(sample: dict[str, Any]) -> list[str]:
    paths = (
        sample.get("layers")
        or sample.get("target_layers")
        or sample.get("rgba_layers")
        or sample.get("foreground_layers")
        or sample.get("LayerInfoRaw")
        or []
    )
    return [str(path) for path in paths]


def _apply_layer_order(layer_paths: list[str], boxes: list[list[int]], layer_order: str) -> tuple[list[str], list[list[int]]]:
    if len(layer_paths) != len(boxes):
        raise ValueError(
            f"Layer path count ({len(layer_paths)}) does not match bbox count ({len(boxes)}). "
            "Check LayerInfoRaw/layers and detections/layout alignment."
        )
    pairs = list(zip(layer_paths, boxes))
    if layer_order == "reverse":
        pairs = list(reversed(pairs))
    elif layer_order != "as_is":
        raise ValueError("layer_order must be 'as_is' or 'reverse'")
    if not pairs:
        return [], []
    ordered_paths, ordered_boxes = zip(*pairs)
    return list(ordered_paths), list(ordered_boxes)


def _resize_crop_geometry(width: int, height: int, target_size: int) -> tuple[int, int, int, int, int, int]:
    scale = target_size / max(width, height)
    resized_w = max(16, int(round(width * scale / 16)) * 16)
    resized_h = max(16, int(round(height * scale / 16)) * 16)
    crop_x = max(0, (resized_w - target_size) // 2)
    crop_y = max(0, (resized_h - target_size) // 2)
    return resized_w, resized_h, crop_x, crop_y, target_size, target_size


def _resize_crop_image(image: Image.Image, geometry: tuple[int, int, int, int, int, int]) -> Image.Image:
    resized_w, resized_h, crop_x, crop_y, crop_w, crop_h = geometry
    image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
    return image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))


def _resize_crop_boxes(
    boxes: Iterable[list[int]],
    source_w: int,
    source_h: int,
    geometry: tuple[int, int, int, int, int, int],
) -> list[list[int]]:
    resized_w, resized_h, crop_x, crop_y, crop_w, crop_h = geometry
    scale_x = resized_w / source_w
    scale_y = resized_h / source_h
    out = []
    for x1, y1, x2, y2 in boxes:
        nx1 = int(round(x1 * scale_x)) - crop_x
        ny1 = int(round(y1 * scale_y)) - crop_y
        nx2 = int(round(x2 * scale_x)) - crop_x
        ny2 = int(round(y2 * scale_y)) - crop_y
        nx1 = max(0, min(crop_w, nx1))
        ny1 = max(0, min(crop_h, ny1))
        nx2 = max(0, min(crop_w, nx2))
        ny2 = max(0, min(crop_h, ny2))
        nx1 = (nx1 // 16) * 16
        ny1 = (ny1 // 16) * 16
        nx2 = ((nx2 + 15) // 16) * 16
        ny2 = ((ny2 + 15) // 16) * 16
        nx2 = max(nx1 + 16, min(crop_w, nx2))
        ny2 = max(ny1 + 16, min(crop_h, ny2))
        out.append([nx1, ny1, nx2, ny2])
    return out


def _image_to_rgb_tensor(image: Image.Image) -> torch.Tensor:
    return _pil_to_tensor(image.convert("RGB")) * 2.0 - 1.0


def _image_to_rgba_tensors(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    rgba = _pil_to_tensor(image.convert("RGBA"))
    return rgba[:3] * 2.0 - 1.0, rgba[3:4]


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    width, height = image.size
    channels = len(image.getbands())
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return data.view(height, width, channels).permute(2, 0, 1).float().div(255.0)


class RevealLayerDataset(Dataset):
    """Manifest dataset aligned to RevealLayer inference order.

    Required sample fields are flexible aliases:
    `image`/`full_image`, `prompt`/`caption`, `layout`/`bboxes`, and
    `layers`/`target_layers` for RGBA targets. `background` is optional; if it
    is missing the full image is used as the background target as an engineering
    fallback.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path | None = None,
        resolution: int = 1024,
        max_layers: int = 12,
        layer_order: str = "as_is",
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.data_root = Path(data_root).resolve() if data_root else self.manifest_path.parent
        self.resolution = int(resolution)
        self.max_layers = int(max_layers)
        self.layer_order = layer_order
        if self.max_layers < 2:
            raise ValueError("max_layers must include at least full image and background layers")
        if self.layer_order not in {"as_is", "reverse"}:
            raise ValueError("layer_order must be 'as_is' or 'reverse'")
        self.samples = _read_manifest(self.manifest_path)
        if not self.samples:
            raise ValueError(f"No samples found in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _path_from_sample(self, sample: dict[str, Any], keys: tuple[str, ...]) -> Path | None:
        for key in keys:
            value = sample.get(key)
            if value:
                return _resolve_path(value, self.data_root)
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_path = self._path_from_sample(sample, ("image", "full_image", "source_image"))
        if image_path is None:
            raise KeyError(f"Sample {index} does not contain image/full_image/source_image")

        full_image = Image.open(image_path).convert("RGB")
        source_w, source_h = full_image.size
        geometry = _resize_crop_geometry(source_w, source_h, self.resolution)
        full_image = _resize_crop_image(full_image, geometry)

        layer_paths, raw_boxes = _apply_layer_order(_extract_layer_paths(sample), _extract_boxes(sample), self.layer_order)
        fg_boxes = _resize_crop_boxes(raw_boxes, source_w, source_h, geometry)
        fg_boxes = fg_boxes[: max(0, self.max_layers - 2)]
        validation_boxes: list[list[int] | None] = [
            [0, 0, self.resolution, self.resolution],
            [0, 0, self.resolution, self.resolution],
            *fg_boxes,
        ]
        while len(validation_boxes) < self.max_layers:
            validation_boxes.append(None)

        background_path = self._path_from_sample(sample, ("background", "background_image", "bg"))
        if background_path is None:
            background = full_image.convert("RGBA")
        else:
            background = _resize_crop_image(Image.open(background_path).convert("RGBA"), geometry)

        layer_paths = list(layer_paths)[: max(0, self.max_layers - 2)]
        target_images = [background]
        for path in layer_paths:
            target_images.append(_resize_crop_image(Image.open(_resolve_path(path, self.data_root)).convert("RGBA"), geometry))
        while len(target_images) < self.max_layers - 1:
            target_images.append(Image.new("RGBA", (self.resolution, self.resolution), (0, 0, 0, 0)))

        rgb_layers = []
        alpha_layers = []
        for image in target_images:
            rgb, alpha = _image_to_rgba_tensors(image)
            rgb_layers.append(rgb)
            alpha_layers.append(alpha)

        return {
            "pixel_values": _image_to_rgb_tensor(full_image),
            "target_rgb": torch.stack(rgb_layers, dim=0),
            "target_alpha": torch.stack(alpha_layers, dim=0),
            "validation_boxes": validation_boxes,
            "prompt": sample.get("prompt") or sample.get("caption") or "",
            "sample_id": str(sample.get("id") or sample.get("imgid") or image_path.stem),
        }


def reveal_layer_collate(batch: list[dict[str, Any]]) -> RevealLayerBatch:
    return RevealLayerBatch(
        pixel_values=torch.stack([item["pixel_values"] for item in batch], dim=0),
        target_rgb=torch.stack([item["target_rgb"] for item in batch], dim=0),
        target_alpha=torch.stack([item["target_alpha"] for item in batch], dim=0),
        validation_boxes=[item["validation_boxes"] for item in batch],
        prompts=[item["prompt"] for item in batch],
        sample_ids=[item["sample_id"] for item in batch],
    )
