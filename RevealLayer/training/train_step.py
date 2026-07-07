from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _encode_rgb_batch(vae, images: torch.Tensor) -> torch.Tensor:
    bsz, layers, channels, height, width = images.shape
    flat = images.reshape(bsz * layers, channels, height, width)
    latents = vae.encode(flat).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents.reshape(bsz, layers, latents.shape[1], latents.shape[2], latents.shape[3])


def _patchify_adapter_latents(masked_latents: torch.Tensor) -> torch.Tensor:
    b_m, c_m, h_m, w_m = masked_latents.shape
    masked_latents = masked_latents.view(b_m, c_m, h_m // 2, 2, w_m // 2, 2)
    masked_latents = masked_latents.permute(0, 2, 4, 1, 3, 5)
    return masked_latents.reshape(b_m, h_m // 2, w_m // 2, c_m * 4)


def build_adapter_data(vae, full_image: torch.Tensor, validation_box: List, num_layers: int, dtype: torch.dtype) -> Tuple[torch.Tensor, List[int]]:
    device = full_image.device
    init_full = full_image.unsqueeze(1)
    bsz, _, channels, height, width = init_full.shape
    num_bg_fgs = num_layers - 1
    init_images = init_full.expand(-1, num_bg_fgs, -1, -1, -1)
    all_images_for_vae = init_images.flatten(0, 1)

    split_sizes = [(height // 16) * (width // 16)]
    mask_pixel = torch.ones(bsz, num_bg_fgs, 1, height, width, device=device, dtype=dtype)
    overlap_counter = torch.zeros((bsz, height, width), device=device)

    for fg_idx in range(2, len(validation_box)):
        if validation_box[fg_idx] is None:
            continue
        bx1, by1, bx2, by2 = validation_box[fg_idx]
        overlap_counter[:, by1:by2, bx1:bx2] += 1
        split_sizes.append(((by2 - by1) // 16) * ((bx2 - bx1) // 16))

    is_fg_union = overlap_counter > 0
    is_overlap_region_global = overlap_counter > 1
    mask_pixel[:, 0] = is_fg_union
    for i, fg_idx in enumerate(range(2, len(validation_box)), start=1):
        if validation_box[fg_idx] is None:
            continue
        x1, y1, x2, y2 = validation_box[fg_idx]
        mask_pixel[:, i, :, y1:y2, x1:x2] = is_overlap_region_global[:, y1:y2, x1:x2]

    mask_image_flat = mask_pixel.view(bsz * num_bg_fgs, 1, height, width)
    masked_image = all_images_for_vae * (1 - mask_image_flat)
    masked_image = masked_image.to(device=device, dtype=dtype)

    with torch.no_grad():
        masked_latents = vae.encode(masked_image).latent_dist.sample()
        masked_latents = (masked_latents - vae.config.shift_factor) * vae.config.scaling_factor
    masked_latents = _patchify_adapter_latents(masked_latents)

    vae_scale_factor = 8
    h_lat = height // vae_scale_factor
    w_lat = width // vae_scale_factor
    mask_latents = mask_image_flat.view(bsz * num_bg_fgs, 1, h_lat, vae_scale_factor, w_lat, vae_scale_factor)
    mask_latents = mask_latents.permute(0, 2, 4, 1, 3, 5).reshape(
        bsz * num_bg_fgs, h_lat, w_lat, vae_scale_factor * vae_scale_factor
    )
    mask_latents = F.max_pool2d(mask_latents.permute(0, 3, 1, 2), kernel_size=2, stride=2).permute(0, 2, 3, 1)
    mask_latents = mask_latents.to(dtype=dtype)
    adapter_flat = torch.cat([masked_latents, mask_latents], dim=-1)
    return adapter_flat.view(bsz, num_bg_fgs, adapter_flat.shape[1], adapter_flat.shape[2], adapter_flat.shape[3]), split_sizes


def sample_flowmatch_noisy_latents(target_latents: torch.Tensor, full_image_latent: torch.Tensor, generator: Optional[torch.Generator] = None):
    noise = torch.randn(target_latents.shape, generator=generator, device=target_latents.device, dtype=target_latents.dtype)
    bsz = target_latents.shape[0]
    t = torch.rand((bsz,), generator=generator, device=target_latents.device, dtype=target_latents.dtype)
    while t.ndim < target_latents.ndim:
        t = t.view(*t.shape, *([1] * (target_latents.ndim - t.ndim)))
    noisy = t * target_latents + (1 - t) * noise
    noisy[:, 0] = full_image_latent[:, 0]
    target_velocity = target_latents - noise
    return noisy, target_velocity, t.flatten()[:bsz]


def prepare_prompt_embeds(pipeline, prompts: List[str], device: torch.device, max_sequence_length: int = 512):
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )
    return prompt_embeds, pooled_prompt_embeds, text_ids


def forward_flow_matching(
    pipeline,
    batch,
    generator: Optional[torch.Generator] = None,
    max_sequence_length: int = 512,
    guidance_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    device = pipeline._execution_device
    dtype = pipeline.transformer.dtype
    full_image = batch.pixel_values.to(device=device, dtype=dtype)
    target_rgb = batch.target_rgb.to(device=device, dtype=dtype)

    if len(batch.validation_boxes) != 1:
        raise NotImplementedError("Current RevealLayer transformer path expects batch size 1 because list_layer_box is not batched.")
    validation_box = batch.validation_boxes[0]
    num_layers = len(validation_box)
    height, width = full_image.shape[-2:]

    with torch.no_grad():
        target_latents = _encode_rgb_batch(pipeline.vae, target_rgb)
        full_image_latent = _encode_rgb_batch(pipeline.vae, full_image.unsqueeze(1))

    prompt_embeds, pooled_prompt_embeds, text_ids = prepare_prompt_embeds(
        pipeline, batch.prompts, device=device, max_sequence_length=max_sequence_length
    )
    latents, target_velocity, t = sample_flowmatch_noisy_latents(target_latents, full_image_latent, generator)
    latent_image_ids = pipeline._prepare_latent_image_ids(
        2 * (height // (pipeline.vae_scale_factor * 2)),
        2 * (width // (pipeline.vae_scale_factor * 2)),
        validation_box,
        device,
        prompt_embeds.dtype,
    )
    adapter_data, split_sizes = build_adapter_data(pipeline.vae, full_image, validation_box, num_layers, dtype)

    if pipeline.transformer.config.guidance_embeds:
        guidance = torch.full([latents.shape[0]], guidance_scale, device=device, dtype=torch.float32)
    else:
        guidance = None

    pred_velocity = pipeline.transformer(
        hidden_states=latents,
        adapter_data=adapter_data,
        split_sizes=split_sizes,
        list_layer_box=validation_box,
        timestep=t,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    return {
        "pred_velocity": pred_velocity,
        "target_velocity": target_velocity,
        "target_latents": target_latents,
        "timestep": t,
    }
