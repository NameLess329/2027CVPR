from typing import Dict, Optional

import torch
import torch.nn.functional as F


def flow_matching_loss(pred_velocity: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_velocity.float(), target_velocity.float())


def hard_constraint_alpha_loss(
    pred_alpha: torch.Tensor,
    target_alpha: torch.Tensor,
    gamma: float = 1.5,
    tau: float = 0.95,
    eps: float = 1e-6,
) -> torch.Tensor:
    if pred_alpha.shape != target_alpha.shape:
        target_alpha = F.interpolate(target_alpha.float(), size=pred_alpha.shape[-2:], mode="bilinear", align_corners=False)
    delta = tau * (pred_alpha.float() - target_alpha.float()).abs()
    delta = delta.clamp(min=0.0, max=1.0 - eps)
    return -((delta ** gamma) * torch.log1p(-delta + eps)).mean()


def soft_orthogonality_loss(background_rgb: torch.Tensor, foreground_rgb: torch.Tensor, foreground_alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
    if foreground_rgb.numel() == 0:
        return background_rgb.new_tensor(0.0)
    bg = background_rgb.float()
    fg = foreground_rgb.float()
    if foreground_alpha is not None:
        fg = fg * foreground_alpha.float()
    bg_vec = bg.flatten(start_dim=2)
    fg_vec = fg.flatten(start_dim=2)
    bg_vec = F.normalize(bg_vec, dim=-1, eps=1e-6)
    fg_vec = F.normalize(fg_vec, dim=-1, eps=1e-6)
    return (bg_vec.unsqueeze(2) * fg_vec.unsqueeze(1)).sum(dim=-1).abs().mean()


def compose_losses(
    fm_loss: torch.Tensor,
    alpha_loss: Optional[torch.Tensor] = None,
    orth_loss: Optional[torch.Tensor] = None,
    alpha_weight: float = 1.0,
    orth_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    total = fm_loss
    out = {"loss": total, "loss_fm": fm_loss.detach()}
    if alpha_loss is not None:
        total = total + alpha_weight * alpha_loss
        out["loss_alpha"] = alpha_loss.detach()
    if orth_loss is not None:
        total = total + orth_weight * orth_loss
        out["loss_orth"] = orth_loss.detach()
    out["loss"] = total
    return out

