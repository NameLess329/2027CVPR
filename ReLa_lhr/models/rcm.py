from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RCMHeadOutput:
    logits: torch.Tensor
    logits_by_layer: list[torch.Tensor]
    layer_indices: list[int]
    boxes: list[list[int]]
    split_sizes: list[int]


def infer_clean_dim_from_refiner(refiner: nn.Module, hidden_dim: int) -> int:
    """Infer adapter/refiner condition token dim from the existing refiner."""

    input_proj = getattr(refiner, "input_proj", None)
    if input_proj is None or not hasattr(input_proj, "in_features"):
        raise ValueError("Cannot infer clean token dim: refiner.input_proj.in_features is missing.")
    clean_dim = int(input_proj.in_features) - int(hidden_dim)
    if clean_dim <= 0:
        raise ValueError(f"Invalid inferred clean token dim: {clean_dim}")
    return clean_dim


def _valid_boxes_with_indices(
    list_layer_box: Sequence[Sequence[int] | None],
    layer_offset: int,
) -> list[tuple[int, list[int]]]:
    valid: list[tuple[int, list[int]]] = []
    for local_idx, box in enumerate(list_layer_box):
        if box is None:
            continue
        valid.append((local_idx + layer_offset, [int(v) for v in box]))
    return valid


def _box_embedding(
    box: Sequence[int],
    image_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    image_h, image_w = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    values = [
        x1 / image_w,
        y1 / image_h,
        x2 / image_w,
        y2 / image_h,
        ((x1 + x2) * 0.5) / image_w,
        ((y1 + y2) * 0.5) / image_h,
        w / image_w,
        h / image_h,
    ]
    return torch.tensor(values, device=device, dtype=dtype).view(1, -1)


def _token_positions(
    box: Sequence[int],
    image_size: tuple[int, int],
    token_factor: int,
    token_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    image_h, image_w = image_size
    x1, y1, x2, y2 = [int(v) for v in box]
    grid_w = max((x2 - x1) // token_factor, 1)
    grid_h = max((y2 - y1) // token_factor, 1)
    expected = grid_h * grid_w
    if expected != int(token_count):
        raise ValueError(
            "RCM token count does not match bbox crop: "
            f"box={list(box)} expected={expected} actual={token_count}"
        )

    ys = torch.arange(grid_h, device=device, dtype=dtype)
    xs = torch.arange(grid_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    abs_x = (float(x1) + (xx + 0.5) * token_factor) / float(image_w)
    abs_y = (float(y1) + (yy + 0.5) * token_factor) / float(image_h)
    local_x = (xx + 0.5) / float(grid_w)
    local_y = (yy + 0.5) / float(grid_h)
    return torch.stack([abs_x, abs_y, local_x, local_y], dim=-1).reshape(1, expected, 4)


class RCMHead(nn.Module):
    """Lightweight Reference Contribution Map scorer.

    The head is intentionally side-effect free: it predicts per-token RCM
    logits from detached RevealLayer features and does not alter generation.
    """

    def __init__(
        self,
        hidden_dim: int,
        clean_dim: int,
        rcm_dim: int = 512,
        time_dim: int | None = None,
        max_layers: int = 32,
        token_factor: int = 16,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.clean_dim = int(clean_dim)
        self.rcm_dim = int(rcm_dim)
        self.time_dim = int(time_dim or hidden_dim)
        self.token_factor = int(token_factor)

        self.x_proj = nn.Linear(self.hidden_dim, self.rcm_dim)
        self.c_proj = nn.Linear(self.clean_dim, self.rcm_dim)
        self.pos_mlp = nn.Sequential(
            nn.Linear(4, self.rcm_dim),
            nn.SiLU(),
            nn.Linear(self.rcm_dim, self.rcm_dim),
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim + self.clean_dim + 8 + self.time_dim, self.rcm_dim),
            nn.SiLU(),
            nn.Linear(self.rcm_dim, self.rcm_dim),
        )
        self.layer_emb = nn.Embedding(max_layers, self.rcm_dim)
        self.out = nn.Linear(self.rcm_dim, 1)

    def forward_layer(
        self,
        hidden_tokens: torch.Tensor,
        clean_tokens: torch.Tensor,
        box: Sequence[int],
        layer_index: int,
        temb: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        compute_dtype = next(self.parameters()).dtype
        hidden_tokens = hidden_tokens.to(dtype=compute_dtype)
        clean_tokens = clean_tokens.to(dtype=compute_dtype)
        temb = temb.to(dtype=compute_dtype)

        if hidden_tokens.shape[:2] != clean_tokens.shape[:2]:
            raise ValueError(
                "RCM hidden/clean token shapes must align: "
                f"hidden={tuple(hidden_tokens.shape)} clean={tuple(clean_tokens.shape)}"
            )
        if clean_tokens.shape[-1] != self.clean_dim:
            raise ValueError(f"Expected clean_dim={self.clean_dim}, got {clean_tokens.shape[-1]}")

        batch, token_count, _ = hidden_tokens.shape
        pos = _token_positions(
            box,
            image_size=image_size,
            token_factor=self.token_factor,
            token_count=token_count,
            device=hidden_tokens.device,
            dtype=compute_dtype,
        ).expand(batch, -1, -1)
        box_emb = _box_embedding(box, image_size, hidden_tokens.device, compute_dtype).expand(batch, -1)

        x_pool = hidden_tokens.mean(dim=1)
        c_pool = clean_tokens.mean(dim=1)
        query = self.query_mlp(torch.cat([x_pool, c_pool, box_emb, temb], dim=-1))
        layer_ids = torch.full((batch,), int(layer_index), device=hidden_tokens.device, dtype=torch.long)
        layer_cond = self.layer_emb(layer_ids)

        features = (
            self.x_proj(hidden_tokens)
            + self.c_proj(clean_tokens)
            + self.pos_mlp(pos)
            + query[:, None, :]
            + layer_cond[:, None, :]
        )
        return self.out(F.silu(features)).squeeze(-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        clean_tokens: torch.Tensor,
        split_sizes: Iterable[int],
        list_layer_box: Sequence[Sequence[int] | None],
        temb: torch.Tensor,
        image_size: tuple[int, int],
        layer_offset: int = 1,
    ) -> RCMHeadOutput:
        split_sizes = [int(size) for size in split_sizes]
        valid_boxes = _valid_boxes_with_indices(list_layer_box, layer_offset=layer_offset)
        if len(valid_boxes) != len(split_sizes):
            raise ValueError(
                "RCM split sizes must match non-padded layer boxes: "
                f"splits={len(split_sizes)} boxes={len(valid_boxes)}"
            )

        hidden_splits = torch.split(hidden_states, split_sizes, dim=1)
        clean_splits = torch.split(clean_tokens, split_sizes, dim=1)
        logits_by_layer: list[torch.Tensor] = []
        layer_indices: list[int] = []
        boxes: list[list[int]] = []

        for (layer_index, box), hidden_i, clean_i in zip(valid_boxes, hidden_splits, clean_splits):
            logits_i = self.forward_layer(
                hidden_tokens=hidden_i,
                clean_tokens=clean_i,
                box=box,
                layer_index=layer_index,
                temb=temb,
                image_size=image_size,
            )
            logits_by_layer.append(logits_i)
            layer_indices.append(layer_index)
            boxes.append(box)

        logits = torch.cat(logits_by_layer, dim=1) if logits_by_layer else hidden_states.new_zeros((hidden_states.shape[0], 0))
        return RCMHeadOutput(
            logits=logits,
            logits_by_layer=logits_by_layer,
            layer_indices=layer_indices,
            boxes=boxes,
            split_sizes=split_sizes,
        )
