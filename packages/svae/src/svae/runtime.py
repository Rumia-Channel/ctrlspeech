"""Device selection shared by training and inference; ROCm uses the CUDA API."""

import torch


def accelerator_module(device):
    """Return the selected GPU's control API, or None for CPU."""
    if device.type == "cpu":
        return None
    if device.type not in {"cuda", "xpu"}:
        raise ValueError("supported devices are cpu, cuda/rocm and xpu")
    return getattr(torch, device.type, None)


def backend_name(device):
    if device.type == "cuda" and torch.version.hip:
        return "rocm"
    return device.type


def resolve_device(device=None):
    """Resolve auto and explicit GPU indices without falling back on a request."""
    if device is None or str(device) == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"
    requested = str(device)
    if requested == "rocm" or requested.startswith("rocm:"):
        if not torch.version.hip:
            raise ValueError("rocm requires a ROCm PyTorch build; run uv sync --extra rocm")
        requested = "cuda" + requested[len("rocm"):]
    try:
        resolved = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError("device must be auto, cpu, cuda[:N], rocm[:N] or xpu[:N]") from error
    module = accelerator_module(resolved)
    if resolved.type == "cpu":
        return torch.device("cpu")
    if module is None or not module.is_available():
        extra = backend_name(resolved)
        raise ValueError(
            f"{extra} is unavailable; install its driver and run uv sync --extra {extra}, "
            "or explicitly select --device cpu"
        )
    index = module.current_device() if resolved.index is None else resolved.index
    if not 0 <= index < module.device_count():
        raise ValueError(f"{backend_name(resolved)} device index {index} is out of range")
    return torch.device(resolved.type, index)


def validate_training_precision(device, precision):
    """Reject unsupported AMP, including Arc devices without FP64 loss scaling."""
    if precision not in {"none", "bf16", "fp16"}:
        raise ValueError("precision must be none, bf16 or fp16")
    if precision == "none":
        return
    if precision == "fp16" and device.type == "cpu":
        raise ValueError("fp16 training requires a CUDA/ROCm or XPU GPU; use none or bf16 on CPU")
    if not torch.amp.autocast_mode.is_autocast_available(device.type):
        raise ValueError(f"AMP is unavailable on {device.type}; use --precision none")
    module = accelerator_module(device)
    if module is None:
        return
    with module.device(device):
        if precision == "bf16" and not module.is_bf16_supported():
            raise ValueError(f"this {backend_name(device)} device does not support bf16; use fp16 or none")
        if precision == "fp16" and device.type == "xpu":
            properties = module.get_device_properties(device)
            if not properties.has_fp16 or not properties.has_fp64:
                raise ValueError(
                    "XPU fp16 training with GradScaler requires FP16 and FP64 hardware support; "
                    "Intel Arc devices without FP64 must use --precision bf16 or none"
                )
