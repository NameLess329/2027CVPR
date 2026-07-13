from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F


@dataclass
class RCMTargets:
    target: torch.Tensor
    confidence: torch.Tensor
    target_by_layer: list[torch.Tensor]
    confidence_by_layer: list[torch.Tensor]
    target_grid_by_layer: list[torch.Tensor]
    confidence_grid_by_layer: list[torch.Tensor]
    layer_indices: list[int]
    boxes: list[list[int]]


def visible_alpha_from_order(alpha_full: torch.Tensor, order_convention: str = "front_to_back") -> torch.Tensor:
    """Compute visible alpha with alpha-over transmission.

    `alpha_full` is `[B, L, 1, H, W]`, ordered front-to-back by default.
    The returned tensor keeps the same layer order as the input.
    """

    if alpha_full.ndim != 5:
        raise ValueError(f"alpha_full must be [B, L, 1, H, W], got {tuple(alpha_full.shape)}")
    if order_convention not in {"front_to_back", "back_to_front"}:
        raise ValueError("order_convention must be 'front_to_back' or 'back_to_front'")

    input_alpha = alpha_full.float().clamp(0.0, 1.0)
    if order_convention == "back_to_front":
        work_alpha = torch.flip(input_alpha, dims=(1,))
    else:
        work_alpha = input_alpha

    transmission = torch.ones_like(work_alpha[:, :1])
    visible = []
    for layer_idx in range(work_alpha.shape[1]):
        alpha_i = work_alpha[:, layer_idx:layer_idx + 1]
        visible_i = alpha_i * transmission
        visible.append(visible_i)
        transmission = transmission * (1.0 - alpha_i)
    visible_alpha = torch.cat(visible, dim=1) if visible else work_alpha

    if order_convention == "back_to_front":
        visible_alpha = torch.flip(visible_alpha, dims=(1,))
    return visible_alpha


def area_target_and_conf(alpha: torch.Tensor, token_factor: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample alpha to token target and confidence using area statistics."""

    if alpha.ndim == 3:
        alpha = alpha.unsqueeze(1)
    if alpha.ndim != 4:
        raise ValueError(f"alpha must be [B, 1, H, W] or [B, H, W], got {tuple(alpha.shape)}")
    alpha = alpha.float().clamp(0.0, 1.0)
    height, width = alpha.shape[-2:]
    token_h = height // token_factor
    token_w = width // token_factor
    if token_h <= 0 or token_w <= 0:
        raise ValueError(f"Image is too small for token_factor={token_factor}: {(height, width)}")

    target = F.interpolate(alpha, size=(token_h, token_w), mode="area")
    target2 = F.interpolate(alpha * alpha, size=(token_h, token_w), mode="area")
    variance = (target2 - target * target).clamp_min(0.0)
    confidence = 1.0 - (variance / 0.25).clamp(0.0, 1.0)
    confidence = 0.2 + 0.8 * confidence
    return target, confidence


def crop_grid_by_box(grid: torch.Tensor, box: Sequence[int], token_factor: int = 16) -> torch.Tensor:
    """Crop `[B, 1, Ht, Wt]` token grid by pixel-space bbox and flatten row-major."""

    if grid.ndim != 4:
        raise ValueError(f"grid must be [B, 1, Ht, Wt], got {tuple(grid.shape)}")
    x1, y1, x2, y2 = [int(v) for v in box]
    lx1, ly1, lx2, ly2 = x1 // token_factor, y1 // token_factor, x2 // token_factor, y2 // token_factor
    lx1 = max(0, min(grid.shape[-1], lx1))
    lx2 = max(lx1, min(grid.shape[-1], lx2))
    ly1 = max(0, min(grid.shape[-2], ly1))
    ly2 = max(ly1, min(grid.shape[-2], ly2))
    return grid[:, :, ly1:ly2, lx1:lx2].reshape(grid.shape[0], -1)


def flat_to_grid(
    flat: torch.Tensor,
    box: Sequence[int],
    grid_shape: tuple[int, int],
    token_factor: int = 16,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Place a flattened bbox crop back into a full token grid for visualization."""

    if flat.ndim == 2:
        flat = flat[0]
    grid_h, grid_w = grid_shape
    out = flat.new_full((grid_h, grid_w), fill_value)
    x1, y1, x2, y2 = [int(v) for v in box]
    lx1, ly1, lx2, ly2 = x1 // token_factor, y1 // token_factor, x2 // token_factor, y2 // token_factor
    crop_h = max(ly2 - ly1, 0)
    crop_w = max(lx2 - lx1, 0)
    expected = crop_h * crop_w
    if expected != int(flat.numel()):
        raise ValueError(f"flat length {flat.numel()} does not match bbox grid area {expected}")
    out[ly1:ly2, lx1:lx2] = flat.reshape(crop_h, crop_w)
    return out


def bbox_from_alpha(
    alpha: torch.Tensor,
    alpha_threshold: float = 0.01,
    padding: int = 8,
    token_factor: int = 16,
) -> list[int] | None:
    """Build a token-aligned pixel-space bbox from full layer alpha."""

    if alpha.ndim == 3:
        alpha = alpha.squeeze(0)
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be [H, W], got {tuple(alpha.shape)}")
    height, width = alpha.shape
    coords = torch.nonzero(alpha.float() > float(alpha_threshold), as_tuple=False)
    if coords.numel() == 0:
        return None
    y1 = int(coords[:, 0].min().item()) - int(padding)
    y2 = int(coords[:, 0].max().item()) + 1 + int(padding)
    x1 = int(coords[:, 1].min().item()) - int(padding)
    x2 = int(coords[:, 1].max().item()) + 1 + int(padding)
    x1 = max(0, (x1 // token_factor) * token_factor)
    y1 = max(0, (y1 // token_factor) * token_factor)
    x2 = min(width, ((x2 + token_factor - 1) // token_factor) * token_factor)
    y2 = min(height, ((y2 + token_factor - 1) // token_factor) * token_factor)
    if x2 <= x1:
        x2 = min(width, x1 + token_factor)
    if y2 <= y1:
        y2 = min(height, y1 + token_factor)
    return [x1, y1, x2, y2]


def build_rcm_targets_from_alpha(
    target_alpha: torch.Tensor,
    validation_boxes: Sequence[Sequence[int] | None],
    include_background: bool = False,
    order_convention: str = "front_to_back",
    token_factor: int = 16,
) -> RCMTargets:
    """Build RCM targets from RevealLayer training batch target alpha.

    `target_alpha` follows the existing dataset contract:
    `[background, foreground_0, foreground_1, ...]`, while
    `validation_boxes` follows `[full_image, background, foreground_0, ...]`.
    """

    if target_alpha.ndim == 4:
        target_alpha = target_alpha.unsqueeze(0)
    if target_alpha.ndim != 5:
        raise ValueError(f"target_alpha must be [B, L, 1, H, W] or [L, 1, H, W], got {tuple(target_alpha.shape)}")

    target_alpha = target_alpha.float().clamp(0.0, 1.0)
    fg_entries: list[tuple[int, int, list[int]]] = []
    for layer_idx, box in enumerate(validation_boxes):
        if layer_idx < 2 or box is None:
            continue
        target_idx = layer_idx - 1
        if target_idx >= target_alpha.shape[1]:
            continue
        fg_entries.append((layer_idx, target_idx, [int(v) for v in box]))

    fg_indices = [target_idx for _, target_idx, _ in fg_entries]
    if fg_indices:
        fg_alpha = target_alpha[:, fg_indices]
        fg_visible = visible_alpha_from_order(fg_alpha, order_convention=order_convention)
    else:
        fg_visible = target_alpha.new_zeros((target_alpha.shape[0], 0, 1, target_alpha.shape[-2], target_alpha.shape[-1]))

    pieces: list[torch.Tensor] = []
    conf_pieces: list[torch.Tensor] = []
    grids: list[torch.Tensor] = []
    conf_grids: list[torch.Tensor] = []
    layer_indices: list[int] = []
    boxes: list[list[int]] = []

    if include_background and len(validation_boxes) > 1 and validation_boxes[1] is not None:
        if fg_indices:
            bg_visible = torch.prod(1.0 - target_alpha[:, fg_indices], dim=1, keepdim=True)
        else:
            bg_visible = torch.ones_like(target_alpha[:, :1])
        target_grid, conf_grid = area_target_and_conf(bg_visible[:, 0], token_factor=token_factor)
        box = [int(v) for v in validation_boxes[1]]
        pieces.append(crop_grid_by_box(target_grid, box, token_factor=token_factor))
        conf_pieces.append(crop_grid_by_box(conf_grid, box, token_factor=token_factor))
        grids.append(target_grid)
        conf_grids.append(conf_grid)
        layer_indices.append(1)
        boxes.append(box)

    for visible_offset, (layer_idx, _target_idx, box) in enumerate(fg_entries):
        target_grid, conf_grid = area_target_and_conf(fg_visible[:, visible_offset], token_factor=token_factor)
        pieces.append(crop_grid_by_box(target_grid, box, token_factor=token_factor))
        conf_pieces.append(crop_grid_by_box(conf_grid, box, token_factor=token_factor))
        grids.append(target_grid)
        conf_grids.append(conf_grid)
        layer_indices.append(layer_idx)
        boxes.append(box)

    if pieces:
        target = torch.cat(pieces, dim=1)
        confidence = torch.cat(conf_pieces, dim=1)
    else:
        target = target_alpha.new_zeros((target_alpha.shape[0], 0))
        confidence = target_alpha.new_zeros((target_alpha.shape[0], 0))

    return RCMTargets(
        target=target,
        confidence=confidence,
        target_by_layer=pieces,
        confidence_by_layer=conf_pieces,
        target_grid_by_layer=grids,
        confidence_grid_by_layer=conf_grids,
        layer_indices=layer_indices,
        boxes=boxes,
    )


def concat_logits_for_targets(rcm_output: Any, targets: RCMTargets) -> torch.Tensor:
    logits_by_index = {
        int(layer_idx): logits
        for layer_idx, logits in zip(rcm_output.layer_indices, rcm_output.logits_by_layer)
    }
    selected = []
    for layer_idx, target_i in zip(targets.layer_indices, targets.target_by_layer):
        if layer_idx not in logits_by_index:
            raise KeyError(f"RCM output does not contain layer {layer_idx}; available={sorted(logits_by_index)}")
        logits_i = logits_by_index[layer_idx]
        if logits_i.shape != target_i.shape:
            raise ValueError(
                f"RCM logits/target shape mismatch for layer {layer_idx}: "
                f"logits={tuple(logits_i.shape)} target={tuple(target_i.shape)}"
            )
        selected.append(logits_i)
    if not selected:
        return targets.target.new_zeros(targets.target.shape)
    return torch.cat(selected, dim=1)


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    return (loss * confidence.float()).sum() / (confidence.float().sum() + eps)


def total_variation_from_logits(
    logits_by_layer: Sequence[torch.Tensor],
    boxes: Sequence[Sequence[int]],
    token_factor: int = 16,
) -> torch.Tensor:
    losses = []
    for logits, box in zip(logits_by_layer, boxes):
        if logits.numel() == 0:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        grid_h = max((y2 - y1) // token_factor, 1)
        grid_w = max((x2 - x1) // token_factor, 1)
        if grid_h < 2 and grid_w < 2:
            continue
        prob = torch.sigmoid(logits).reshape(logits.shape[0], grid_h, grid_w)
        terms = []
        if grid_h >= 2:
            terms.append((prob[:, 1:] - prob[:, :-1]).abs().mean())
        if grid_w >= 2:
            terms.append((prob[:, :, 1:] - prob[:, :, :-1]).abs().mean())
        if terms:
            losses.append(torch.stack(terms).mean())
    if not losses:
        if logits_by_layer:
            return logits_by_layer[0].sum() * 0.0
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def rcm_metrics(logits: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor) -> dict[str, torch.Tensor]:
    prob = torch.sigmoid(logits.float())
    target = target.float()
    confidence = confidence.float()
    abs_err = (prob - target).abs()
    weighted_mae = (abs_err * confidence).sum() / confidence.sum().clamp_min(1e-6)

    high_conf = confidence > 0.7
    visible = target > 0.7
    unreliable = target < 0.1
    boundary = (target > 0.1) & (target < 0.9)

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return values[mask].float().mean()
        return values.sum() * 0.0

    return {
        "mae": weighted_mae,
        "high_conf_mae": masked_mean(abs_err, high_conf),
        "visible_recall": masked_mean((prob > 0.5).float(), visible),
        "unreliable_suppression": masked_mean((prob < 0.5).float(), unreliable),
        "boundary_calibration": masked_mean(abs_err, boundary),
    }
