from typing import Iterable, List, Optional

import torch
from safetensors.torch import save_file


DEFAULT_TARGET_MODULES = [
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
    "proj_mlp",
    "proj_out",
    "x_embedder",
    "context_embedder",
]


def _ensure_peft():
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError("LoRA training requires peft. Install peft==0.15.2 or newer.") from exc
    return LoraConfig


def freeze_module(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def discover_target_modules(model: torch.nn.Module, requested: Optional[Iterable[str]] = None) -> List[str]:
    requested = list(requested or DEFAULT_TARGET_MODULES)
    module_names = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    found = []
    for target in requested:
        if any(name == target or name.endswith("." + target) for name in module_names):
            found.append(target)
    if not found:
        sample = ", ".join(module_names[:20])
        raise ValueError(f"No LoRA target modules matched. First Linear modules: {sample}")
    return found


def add_lora_to_transformer(
    transformer: torch.nn.Module,
    rank: int = 64,
    alpha: Optional[int] = None,
    dropout: float = 0.0,
    target_modules: Optional[Iterable[str]] = None,
):
    LoraConfig = _ensure_peft()
    freeze_module(transformer)
    targets = discover_target_modules(transformer, target_modules)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha or rank,
        init_lora_weights="gaussian",
        target_modules=targets,
        lora_dropout=dropout,
    )
    transformer.add_adapter(config)
    return targets


def mark_extra_trainable(transformer: torch.nn.Module, train_layer_pe: bool = True, train_refiner: bool = True) -> List[str]:
    trainable = []
    if train_layer_pe and hasattr(transformer, "layer_pe"):
        transformer.layer_pe.requires_grad_(True)
        trainable.append("layer_pe")
    if train_refiner and hasattr(transformer, "refiner"):
        for param in transformer.refiner.parameters():
            param.requires_grad_(True)
        trainable.append("refiner")
    return trainable


def trainable_parameters(module: torch.nn.Module):
    return (param for param in module.parameters() if param.requires_grad)


def trainable_parameter_summary(module: torch.nn.Module) -> str:
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    total = sum(param.numel() for param in module.parameters())
    pct = 100.0 * trainable / max(total, 1)
    return f"trainable={trainable:,} total={total:,} ({pct:.4f}%)"


def save_trainable_state_dict(module: torch.nn.Module, output_path: str) -> None:
    state_dict = {
        f"transformer.{name}": param.detach().cpu()
        for name, param in module.state_dict().items()
        if "lora_" in name
    }
    save_file(state_dict, output_path)


def save_extra_trainable_state_dict(module: torch.nn.Module, output_path: str) -> None:
    state_dict = {
        name: param.detach().cpu()
        for name, param in module.state_dict().items()
        if name == "layer_pe" or name.startswith("refiner.")
    }
    if state_dict:
        save_file(state_dict, output_path)
