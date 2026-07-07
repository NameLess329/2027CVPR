import json
import math
import ntpath
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _looks_like_windows_path(path: str) -> bool:
    return "\\" in path or ntpath.splitdrive(path)[0] != ""


def _is_abs_path(path: str) -> bool:
    return os.path.isabs(path) or ntpath.isabs(path)


def _path_dirname(path: str) -> str:
    return ntpath.dirname(path) if _looks_like_windows_path(path) else os.path.dirname(path)


def _join_path(root: str, path: str) -> str:
    if _looks_like_windows_path(root):
        return ntpath.normpath(ntpath.join(root, path))
    return os.path.normpath(os.path.join(root, path))


def resolve_data_path(path: str, data_root: str) -> str:
    if not path or _is_abs_path(path):
        return path
    return _join_path(data_root, path)


def load_manifest(path: str) -> List[Dict[str, Any]]:
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8-sig") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("samples", "data", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError(f"Manifest dict at {path} must contain one of: samples, data, items")
    if not isinstance(data, list):
        raise ValueError(f"Manifest at {path} must be a list, JSONL, or dict with samples/data/items")
    return data


def adjust_coordinate(value: float, floor_or_ceil: str, k: int = 16, min_val: int = 0, max_val: int = 1024) -> int:
    rounded_value = math.floor(value / k) * k if floor_or_ceil == "floor" else math.ceil(value / k) * k
    return max(min_val, min(int(rounded_value), max_val))


def resize_geometry(image: Image.Image, resolution: int = 1024, k: int = 16):
    orig_w, orig_h = image.size
    long_side = max(orig_w, orig_h)
    scale = resolution / long_side
    content_w = int(round(orig_w * scale))
    content_h = int(round(orig_h * scale))
    target_w = max(k, (content_w // k) * k)
    target_h = max(k, (content_h // k) * k)
    offset_x = (content_w - target_w) // 2
    offset_y = (content_h - target_h) // 2
    return orig_w, orig_h, content_w, content_h, target_w, target_h, offset_x, offset_y


def transform_consistent_crop(image: Image.Image, content_w: int, content_h: int, target_w: int, target_h: int, offset_x: int, offset_y: int) -> Image.Image:
    resized = image.resize((content_w, content_h), Image.Resampling.LANCZOS)
    return resized.crop((offset_x, offset_y, offset_x + target_w, offset_y + target_h))


def filter_and_align_bboxes(
    raw_bbox_list: Sequence[Sequence[float]],
    orig_w: int,
    orig_h: int,
    content_w: int,
    content_h: int,
    offset_x: int,
    offset_y: int,
    target_w: int,
    target_h: int,
    k: int = 16,
) -> List[List[int]]:
    if orig_w <= 0 or orig_h <= 0:
        return []
    scale_w = content_w / orig_w
    scale_h = content_h / orig_h
    valid = []
    for box in raw_bbox_list:
        x1, y1, x2, y2 = box
        sx1, sy1, sx2, sy2 = x1 * scale_w, y1 * scale_h, x2 * scale_w, y2 * scale_h
        scaled_w, scaled_h = sx2 - sx1, sy2 - sy1
        cx1, cy1 = max(0, min(sx1 - offset_x, target_w)), max(0, min(sy1 - offset_y, target_h))
        cx2, cy2 = max(0, min(sx2 - offset_x, target_w)), max(0, min(sy2 - offset_y, target_h))
        visible_w, visible_h = cx2 - cx1, cy2 - cy1
        if visible_w > 1 and visible_h > 1 and scaled_w > 0 and scaled_h > 0:
            if visible_w / scaled_w >= 0.5 and visible_h / scaled_h >= 0.5:
                valid.append([
                    adjust_coordinate(cx1, "floor", k=k, max_val=target_w),
                    adjust_coordinate(cy1, "floor", k=k, max_val=target_h),
                    adjust_coordinate(cx2, "ceil", k=k, max_val=target_w),
                    adjust_coordinate(cy2, "ceil", k=k, max_val=target_h),
                ])
    return valid


@dataclass
class RevealLayerBatch:
    pixel_values: torch.Tensor
    target_rgb: torch.Tensor
    target_alpha: torch.Tensor
    validation_boxes: List[List[List[int]]]
    prompts: List[str]
    sample_ids: List[str]


class RevealLayerDataset(Dataset):
    """Dataset aligned with infer.py: [full image, background, foreground layers]."""

    def __init__(
        self,
        manifest_path: str,
        data_root: Optional[str] = None,
        resolution: int = 1024,
        max_layers: int = 12,
        prompt: str = "Decompose the image into foreground and background.",
        shuffle_layers: bool = False,
    ):
        self.manifest_path = manifest_path
        self.data_root = data_root or (_path_dirname(manifest_path) or ".")
        self.entries = load_manifest(manifest_path)
        self.resolution = resolution
        self.max_layers = max_layers
        self.prompt = prompt
        self.shuffle_layers = shuffle_layers
        self.rgb_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.alpha_transform = transforms.Compose([
            transforms.Lambda(lambda img: img.getchannel("A") if img.mode == "RGBA" else Image.new("L", img.size, 255)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.entries)

    def _entry_path(self, entry: Dict[str, Any], key: str) -> Optional[str]:
        value = entry.get(key)
        if not value:
            return None
        return resolve_data_path(value, self.data_root)

    def _layer_paths(self, entry: Dict[str, Any]) -> List[str]:
        layer_paths = entry.get("LayerInfoRaw") or entry.get("layers") or entry.get("foreground_layers") or []
        return [resolve_data_path(path, self.data_root) for path in layer_paths]

    def _load_layer_rgba(self, path: str, crop_args) -> Image.Image:
        image = Image.open(path).convert("RGBA")
        return transform_consistent_crop(image, *crop_args).convert("RGBA")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        full_image_path = self._entry_path(entry, "full_image")
        if not full_image_path:
            raise ValueError(f"Sample {index} has no full_image field")
        full_image = Image.open(full_image_path).convert("RGB")

        orig_w, orig_h, content_w, content_h, target_w, target_h, offset_x, offset_y = resize_geometry(
            full_image, resolution=self.resolution
        )
        crop_args = (content_w, content_h, target_w, target_h, offset_x, offset_y)
        full_image_resized = transform_consistent_crop(full_image, *crop_args)

        raw_boxes = [d["bbox"] for d in entry.get("detections", []) if "bbox" in d]
        layer_paths = self._layer_paths(entry)
        if self.shuffle_layers and len(raw_boxes) == len(layer_paths):
            order = torch.randperm(len(raw_boxes)).tolist()
            raw_boxes = [raw_boxes[i] for i in order]
            layer_paths = [layer_paths[i] for i in order]

        resized_boxes = filter_and_align_bboxes(
            raw_boxes, orig_w, orig_h, content_w, content_h, offset_x, offset_y, target_w, target_h
        )
        max_fg = max(0, self.max_layers - 2)
        resized_boxes = resized_boxes[:max_fg]
        layer_paths = layer_paths[: len(resized_boxes)]

        validation_boxes = [[0, 0, target_w, target_h], [0, 0, target_w, target_h]] + resized_boxes
        layer_images = [full_image_resized.convert("RGBA")]

        background_path = self._entry_path(entry, "background")
        if background_path:
            bg_image = Image.open(background_path).convert("RGBA")
            layer_images.append(transform_consistent_crop(bg_image, *crop_args).convert("RGBA"))
        else:
            layer_images.append(full_image_resized.convert("RGBA"))

        for layer_path in layer_paths:
            layer_images.append(self._load_layer_rgba(layer_path, crop_args))

        # Pad to a fixed layer count so batches can stack tensors. Missing boxes stay None.
        while len(layer_images) < self.max_layers:
            layer_images.append(Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0)))
            validation_boxes.append(None)
        layer_images = layer_images[: self.max_layers]
        validation_boxes = validation_boxes[: self.max_layers]

        target_rgb = torch.stack([self.rgb_transform(img) for img in layer_images], dim=0)
        target_alpha = torch.stack([self.alpha_transform(img) for img in layer_images], dim=0)

        return {
            "pixel_values": self.rgb_transform(full_image_resized),
            "target_rgb": target_rgb,
            "target_alpha": target_alpha,
            "validation_boxes": validation_boxes,
            "prompt": entry.get("prompt") or entry.get("description") or self.prompt,
            "sample_id": entry.get("imgid") or entry.get("id") or str(index),
        }


def collate_reveallayer(batch: List[Dict[str, Any]]) -> RevealLayerBatch:
    return RevealLayerBatch(
        pixel_values=torch.stack([item["pixel_values"] for item in batch], dim=0),
        target_rgb=torch.stack([item["target_rgb"] for item in batch], dim=0),
        target_alpha=torch.stack([item["target_alpha"] for item in batch], dim=0),
        validation_boxes=[item["validation_boxes"] for item in batch],
        prompts=[item["prompt"] for item in batch],
        sample_ids=[item["sample_id"] for item in batch],
    )

