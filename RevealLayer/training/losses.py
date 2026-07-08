from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_matching_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
    training_weight: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Masked FM loss for RevealLayer tokens.

    `token_mask` is expected to be broadcastable to `[batch, tokens, channels]`.
    Layer 0, the full-image condition layer, must already be zeroed by the
    caller.
    """

    while token_mask.ndim < prediction.ndim:
        token_mask = token_mask.unsqueeze(-1)
    sq = (prediction.float() - target.float()).pow(2) * token_mask.float()
    denom = token_mask.float().sum().clamp_min(1.0) * prediction.shape[-1]
    loss = sq.sum() / denom
    if training_weight is not None:
        loss = loss * torch.as_tensor(training_weight, dtype=loss.dtype, device=loss.device)
    return loss


def hard_constraint_alpha_loss(
    pred_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    validation_boxes: list[list[int] | None],
    tau: float = 0.95,
    gamma: float = 1.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RevealLayer focal-style alpha loss over foreground layers.

    `pred_alpha` is `[num_layers, 1, H, W]` from the transparent decoder.
    `target_alpha` is `[num_layers - 1, 1, H, W]` for background + foregrounds.
    """

    losses = []
    for layer_idx, box in enumerate(validation_boxes):
        if layer_idx < 2 or box is None:
            continue
        target_idx = layer_idx - 1
        if target_idx >= target_alpha.shape[0]:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        pred = ((pred_alpha[layer_idx:layer_idx + 1] + 1.0) / 2.0).clamp(0.0, 1.0)
        target = target_alpha[target_idx:target_idx + 1].float().clamp(0.0, 1.0)
        delta = tau * (pred[..., y1:y2, x1:x2] - target[..., y1:y2, x1:x2]).abs()
        delta = delta.clamp(0.0, 1.0 - eps)
        losses.append(-((delta ** gamma) * torch.log(1.0 - delta + eps)).mean())
    if not losses:
        return pred_alpha.sum() * 0.0
    return torch.stack(losses).mean()


def soft_orthogonality_loss(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    target_alpha: torch.Tensor,
    validation_boxes: list[list[int] | None],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pixel-space soft orthogonality loss.

    The paper defines a cosine-style constraint to suppress foreground-like
    residuals in background reconstruction. This implementation compares the
    predicted background/foreground cosine in each foreground region against
    the same cosine measured from the target layers.
    """

    pred_rgb = ((pred_rgb + 1.0) / 2.0).float().clamp(0.0, 1.0)
    target_rgb = ((target_rgb + 1.0) / 2.0).float().clamp(0.0, 1.0)
    target_alpha = target_alpha.float().clamp(0.0, 1.0)
    losses = []
    pred_bg = pred_rgb[1]
    target_bg = target_rgb[0]
    for layer_idx, box in enumerate(validation_boxes):
        if layer_idx < 2 or box is None:
            continue
        target_idx = layer_idx - 1
        if target_idx >= target_rgb.shape[0]:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        mask = target_alpha[target_idx, :, y1:y2, x1:x2]
        if mask.sum() <= eps:
            continue
        pred_bg_vec = (pred_bg[:, y1:y2, x1:x2] * mask).reshape(-1)
        pred_fg_vec = (pred_rgb[layer_idx, :, y1:y2, x1:x2] * mask).reshape(-1)
        target_bg_vec = (target_bg[:, y1:y2, x1:x2] * mask).reshape(-1)
        target_fg_vec = (target_rgb[target_idx, :, y1:y2, x1:x2] * mask).reshape(-1)
        pred_cos = F.cosine_similarity(pred_bg_vec, pred_fg_vec, dim=0, eps=eps)
        target_cos = F.cosine_similarity(target_bg_vec, target_fg_vec, dim=0, eps=eps).detach()
        losses.append((pred_cos - target_cos).abs())
    if not losses:
        return pred_rgb.sum() * 0.0
    return torch.stack(losses).mean()


alpha_reconstruction_loss = hard_constraint_alpha_loss
orthogonality_loss = soft_orthogonality_loss
