from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


DIFFUSERS_LORA_NAME = "pytorch_lora_weights.safetensors"
LEGACY_LORA_NAME = "reveallayer_peft_lora_weights.safetensors"


DEFAULT_LORA_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
)


def freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def discover_lora_targets(module: torch.nn.Module, requested: Iterable[str] | None = None) -> list[str]:
    requested = tuple(requested or DEFAULT_LORA_TARGETS)
    linear_names = {
        name
        for name, submodule in module.named_modules()
        if isinstance(submodule, torch.nn.Linear)
    }
    targets = []
    for suffix in requested:
        if any(name == suffix or name.endswith(f".{suffix}") for name in linear_names):
            targets.append(suffix)
    if not targets:
        raise ValueError(f"No LoRA target modules matched requested suffixes: {requested}")
    return targets


def add_lora_to_transformer(
    transformer: torch.nn.Module,
    rank: int = 64,
    alpha: int | None = None,
    dropout: float = 0.0,
    target_modules: Iterable[str] | None = None,
    upcast_trainable_dtype: torch.dtype | None = torch.float32,
) -> tuple[torch.nn.Module, list[str]]:
    from peft import LoraConfig, inject_adapter_in_model

    freeze_module(transformer)
    targets = discover_lora_targets(transformer, target_modules)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha or rank,
        lora_dropout=dropout,
        target_modules=targets,
        bias="none",
    )
    transformer = inject_adapter_in_model(config, transformer)
    if upcast_trainable_dtype is not None:
        for param in transformer.parameters():
            if param.requires_grad:
                param.data = param.data.to(upcast_trainable_dtype)
    return transformer, targets


def trainable_state_dict(module: torch.nn.Module, prefix: str = "") -> dict[str, torch.Tensor]:
    state = {}
    for name, param in module.named_parameters():
        if param.requires_grad:
            state[prefix + name] = param.detach().cpu()
    return state


def save_lora_state(module: torch.nn.Module, path: str | Path, prefix: str = "transformer.") -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value
        for key, value in trainable_state_dict(module, prefix=prefix).items()
        if "lora_A" in key or "lora_B" in key
    }
    if not state:
        raise ValueError("No LoRA parameters found to save")
    save_file(state, str(path))


def save_diffusers_lora_state(
    module: torch.nn.Module,
    path: str | Path,
    prefix: str = "transformer.",
    adapter_name: str = "default",
) -> None:
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = get_peft_model_state_dict(module, adapter_name=adapter_name)
    if not state:
        raise ValueError("No LoRA parameters found to save")
    packed = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        packed[prefix + key] = value.detach().cpu()
    save_file(packed, str(path), metadata={"format": "pt"})


def load_lora_state(module: torch.nn.Module, path: str | Path, prefix: str = "transformer.") -> None:
    from safetensors.torch import load_file

    state = load_file(str(path))
    stripped = {}
    for key, value in state.items():
        if key.startswith("transformer.module."):
            key = "transformer." + key[len("transformer.module."):]
        if key.startswith(prefix):
            key = key[len(prefix):]
        if ".lora_A.weight" in key:
            key = key.replace(".lora_A.weight", ".lora_A.default.weight")
        if ".lora_B.weight" in key:
            key = key.replace(".lora_B.weight", ".lora_B.default.weight")
        stripped[key] = value
    missing, unexpected = module.load_state_dict(stripped, strict=False)
    unexpected_lora = [key for key in unexpected if "lora" in key]
    if unexpected_lora:
        raise ValueError(f"Unexpected LoRA keys while loading {path}: {unexpected_lora[:8]}")
