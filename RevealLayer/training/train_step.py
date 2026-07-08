from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FlowMatchBatchOutput:
    prediction: torch.Tensor
    target: torch.Tensor
    token_mask: torch.Tensor
    timestep: torch.Tensor
    sigma: torch.Tensor
    noisy_latents: torch.Tensor
    clean_latents: torch.Tensor
    clean_prediction: torch.Tensor
    validation_boxes: list[list[int] | None]


def _pack_latents(latents: torch.Tensor, batch_size: int, num_channels: int, height: int, width: int) -> torch.Tensor:
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)


def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
    batch_size, num_patches, channels = latents.shape
    latents = latents.reshape(batch_size, height // vae_scale_factor, width // vae_scale_factor, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, height // (vae_scale_factor // 2), width // (vae_scale_factor // 2))


def _prepare_latent_image_ids(
    height: int,
    width: int,
    validation_boxes: list[list[int] | None],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ids = []
    for layer_idx, box in enumerate(validation_boxes):
        if box is None:
            continue
        layer_ids = torch.zeros(height // 2, width // 2, 3, device=device, dtype=dtype)
        layer_ids[..., 0] = layer_idx
        layer_ids[..., 1] = layer_ids[..., 1] + torch.arange(height // 2, device=device, dtype=dtype)[:, None]
        layer_ids[..., 2] = layer_ids[..., 2] + torch.arange(width // 2, device=device, dtype=dtype)[None, :]
        x1, y1, x2, y2 = [int(v) // 16 for v in box]
        ids.append(layer_ids[y1:y2, x1:x2].reshape(-1, 3))
    return torch.cat(ids, dim=0)


def _crop_layer_tokens(tokens: torch.Tensor, validation_boxes: list[list[int] | None], layer: int, h_p: int, w_p: int) -> torch.Tensor:
    if validation_boxes[layer] is None:
        raise ValueError(f"Layer {layer} is padded and cannot be cropped")
    x1, y1, x2, y2 = [int(v) // 16 for v in validation_boxes[layer]]
    return tokens[:, layer, y1:y2, x1:x2].reshape(tokens.shape[0], -1, tokens.shape[-1])


def _concat_visible_layer_tokens(tokens: torch.Tensor, validation_boxes: list[list[int] | None], h_p: int, w_p: int) -> torch.Tensor:
    pieces = []
    for layer in range(len(validation_boxes)):
        if validation_boxes[layer] is None:
            continue
        pieces.append(_crop_layer_tokens(tokens, validation_boxes, layer, h_p, w_p))
    return torch.cat(pieces, dim=1)


def _build_grid_loss_mask(latents: torch.Tensor, validation_boxes: list[list[int] | None]) -> torch.Tensor:
    mask = torch.zeros(
        (latents.shape[0], latents.shape[1], 1, latents.shape[-2], latents.shape[-1]),
        device=latents.device,
        dtype=torch.float32,
    )
    for layer, box in enumerate(validation_boxes):
        if layer == 0 or box is None:
            continue
        if layer == 1:
            mask[:, layer] = 1.0
            continue
        x1, y1, x2, y2 = [int(v) // 8 for v in box]
        mask[:, layer, :, y1:y2, x1:x2] = 1.0
    return mask


def _flow_timesteps(num_steps: int, image_seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps, device=device, dtype=dtype)
    base_shift = 0.5
    max_shift = 1.15
    base_seq_len = 256
    max_seq_len = 4096
    mu = (max_shift - base_shift) / (max_seq_len - base_seq_len) * image_seq_len + base_shift
    exp_mu = math.exp(float(mu))
    sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))
    timesteps = sigmas * 1000.0
    weights = torch.exp(-2.0 * ((timesteps - 500.0) / 1000.0).pow(2))
    weights = (weights - weights.min()).clamp_min(0)
    weights = weights * (num_steps / weights.sum().clamp_min(1e-6))
    return sigmas, timesteps, weights


def _select_timestep(
    num_steps: int,
    image_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    min_timestep_boundary: float = 0.0,
    max_timestep_boundary: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sigmas, timesteps, weights = _flow_timesteps(num_steps, image_seq_len, device, dtype)
    low = int(min_timestep_boundary * num_steps)
    high = max(low + 1, int(max_timestep_boundary * num_steps))
    idx = torch.randint(low, min(high, num_steps), (1,), device=device)
    return sigmas[idx].reshape(1), timesteps[idx].reshape(1), weights[idx].reshape(1)


def _encode_vae_image(vae: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    first_param = next(vae.parameters())
    image = image.to(device=first_param.device, dtype=first_param.dtype)
    encoded = vae.encode(image)
    if hasattr(encoded, "latent_dist"):
        encoded = encoded.latent_dist.sample()
    elif hasattr(encoded, "sample"):
        encoded = encoded.sample()
    if hasattr(vae, "config") and hasattr(vae.config, "shift_factor"):
        encoded = (encoded - vae.config.shift_factor) * vae.config.scaling_factor
    return encoded


def _build_adapter_data(
    vae: torch.nn.Module,
    full_image: torch.Tensor,
    validation_boxes: list[list[int] | None],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[int]]:
    b, _, height, width = full_image.shape
    num_bg_fgs = sum(box is not None for box in validation_boxes[1:])
    visible_images = full_image[:, None].expand(-1, num_bg_fgs, -1, -1, -1)

    masks = torch.ones((b, num_bg_fgs, 1, height, width), device=full_image.device, dtype=full_image.dtype)
    overlap_counter = torch.zeros((b, height, width), device=full_image.device, dtype=full_image.dtype)
    for box in validation_boxes[2:]:
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        overlap_counter[:, y1:y2, x1:x2] += 1.0

    masks[:, 0] = (overlap_counter > 0).to(masks.dtype)
    overlap = overlap_counter > 1
    for idx, box_idx in enumerate(range(2, len(validation_boxes)), start=1):
        if idx >= num_bg_fgs:
            break
        box = validation_boxes[box_idx]
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        masks[:, idx, :, y1:y2, x1:x2] = overlap[:, y1:y2, x1:x2].to(masks.dtype).unsqueeze(1)

    masked_images = visible_images * (1.0 - masks)
    flat = masked_images.reshape(b * num_bg_fgs, 3, height, width)
    masked_latents = _encode_vae_image(vae, flat)
    masked_latents = _pack_latents(masked_latents, b * num_bg_fgs, masked_latents.shape[1], masked_latents.shape[2], masked_latents.shape[3])
    h_p = height // 16
    w_p = width // 16
    masked_latents = masked_latents.view(b, num_bg_fgs, h_p, w_p, -1)

    vae_scale_factor = 8
    mask_flat = masks.reshape(b * num_bg_fgs, 1, height, width)
    h_lat = height // vae_scale_factor
    w_lat = width // vae_scale_factor
    mask_latents = mask_flat.view(b * num_bg_fgs, 1, h_lat, vae_scale_factor, w_lat, vae_scale_factor)
    mask_latents = mask_latents.permute(0, 2, 4, 1, 3, 5)
    mask_latents = mask_latents.reshape(b * num_bg_fgs, h_lat, w_lat, vae_scale_factor * vae_scale_factor)
    mask_latents = mask_latents.permute(0, 3, 1, 2)
    mask_latents = torch.nn.functional.max_pool2d(mask_latents, kernel_size=2, stride=2)
    mask_latents = mask_latents.permute(0, 2, 3, 1).reshape(b, num_bg_fgs, h_p, w_p, -1)
    adapter = torch.cat([masked_latents, mask_latents.to(masked_latents.dtype)], dim=-1).to(dtype=dtype)

    split_sizes = [h_p * w_p]
    for box in validation_boxes[2:]:
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        split_sizes.append(((y2 - y1) // 16) * ((x2 - x1) // 16))
    return adapter, split_sizes


def _prepare_prompt_inputs(pipe: Any, prompts: list[str], device: torch.device, dtype: torch.dtype, max_sequence_length: int) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
    return {
        "prompt_embeds": prompt_embeds.to(device=device, dtype=dtype),
        "pooled_prompt_embeds": pooled_prompt_embeds.to(device=device, dtype=dtype),
        "text_ids": text_ids.to(device=device, dtype=dtype),
    }


def prepare_transparent_decoder_latents(
    pipe: Any,
    latents: torch.Tensor,
    validation_boxes: list[list[int] | None],
) -> torch.Tensor:
    """Match the inference-time latent preparation before `transparent_decoder`.

    `latents` are FLUX VAE-scaled `[B, L, C, H, W]` tensors. The transparent
    decoder expects unscaled `[L, C, H, W]` latents for the single sample.
    """

    if latents.shape[0] != 1:
        raise ValueError("transparent decoder path currently supports batch size 1")
    b, n_layers, channel_latent, height, width = latents.shape
    with torch.no_grad():
        pixel_grey = torch.zeros(
            size=(b * n_layers, 3, height * 8, width * 8),
            device=latents.device,
            dtype=latents.dtype,
        )
        latent_grey = _encode_vae_image(pipe.vae, pixel_grey)
        latent_grey = latent_grey.view(b, n_layers, channel_latent, height, width)

    filled = latent_grey.clone()
    for layer_idx, box in enumerate(validation_boxes):
        if box is None:
            continue
        x1, y1, x2, y2 = [int(v) // 8 for v in box]
        filled[:, layer_idx, :, y1:y2, x1:x2] = latents[:, layer_idx, :, y1:y2, x1:x2]

    filled = (filled / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    return filled.reshape(b * n_layers, channel_latent, height, width)


def decode_predicted_rgba(
    pipe: Any,
    transparent_decoder: torch.nn.Module,
    flow_output: FlowMatchBatchOutput,
) -> tuple[torch.Tensor, torch.Tensor]:
    decoder_latents = prepare_transparent_decoder_latents(
        pipe,
        flow_output.clean_prediction,
        flow_output.validation_boxes,
    )
    return transparent_decoder(decoder_latents, [flow_output.validation_boxes])


def forward_flow_matching(
    pipe: Any,
    batch: Any,
    num_train_timesteps: int = 1000,
    min_timestep_boundary: float = 0.0,
    max_timestep_boundary: float = 1.0,
    max_sequence_length: int = 512,
    guidance_scale: float = 1.0,
) -> FlowMatchBatchOutput:
    if batch.pixel_values.shape[0] != 1:
        raise ValueError("RevealLayer custom list_layer_box path is unbatched; use train_batch_size=1 and gradient accumulation.")

    device = batch.pixel_values.device
    transformer = pipe.transformer
    transformer_module = transformer.module if hasattr(transformer, "module") else transformer
    dtype = next(transformer_module.parameters()).dtype
    validation_boxes = batch.validation_boxes[0]
    full_image = batch.pixel_values.to(device=device, dtype=dtype)
    target_images = batch.target_rgb.to(device=device, dtype=dtype)
    target_images = torch.cat([full_image[:, None], target_images], dim=1)

    with torch.no_grad():
        layer_latents = _encode_vae_image(pipe.vae, target_images.reshape(-1, 3, target_images.shape[-2], target_images.shape[-1]))

    b = full_image.shape[0]
    num_layers = target_images.shape[1]
    h_lat, w_lat = layer_latents.shape[-2:]
    clean_grid = layer_latents.view(b, num_layers, layer_latents.shape[1], h_lat, w_lat)

    latent_image_ids = _prepare_latent_image_ids(h_lat, w_lat, validation_boxes, device, dtype)
    sigma, timestep, weight = _select_timestep(
        num_train_timesteps,
        latent_image_ids.shape[0],
        device,
        torch.float32,
        min_timestep_boundary=min_timestep_boundary,
        max_timestep_boundary=max_timestep_boundary,
    )
    sigma = sigma.to(device=device, dtype=clean_grid.dtype).view(1, 1, 1, 1, 1)
    noise = torch.randn_like(clean_grid)
    noisy_grid = (1.0 - sigma) * clean_grid + sigma * noise
    target = noise - clean_grid
    noisy_grid[:, 0] = clean_grid[:, 0]
    token_mask = _build_grid_loss_mask(clean_grid, validation_boxes)

    with torch.no_grad():
        adapter_data, split_sizes = _build_adapter_data(
            pipe.vae,
            full_image,
            validation_boxes,
            dtype=dtype,
        )
        prompt_inputs = _prepare_prompt_inputs(pipe, batch.prompts, device, dtype, max_sequence_length)

    guidance = None
    if getattr(transformer_module.config, "guidance_embeds", False):
        guidance = torch.full((1,), guidance_scale, device=device, dtype=dtype)

    prediction = transformer(
        hidden_states=noisy_grid.to(dtype=dtype),
        adapter_data=adapter_data,
        split_sizes=split_sizes,
        list_layer_box=validation_boxes,
        timestep=(timestep.to(device=device, dtype=dtype) / 1000.0),
        guidance=guidance,
        pooled_projections=prompt_inputs["pooled_prompt_embeds"],
        encoder_hidden_states=prompt_inputs["prompt_embeds"],
        txt_ids=prompt_inputs["text_ids"],
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]
    clean_prediction = noisy_grid - sigma * prediction
    clean_prediction[:, 0] = clean_grid[:, 0]

    return FlowMatchBatchOutput(
        prediction=prediction,
        target=target,
        token_mask=token_mask,
        timestep=timestep,
        sigma=sigma.reshape(1),
        noisy_latents=noisy_grid,
        clean_latents=clean_grid,
        clean_prediction=clean_prediction,
        validation_boxes=validation_boxes,
    )
