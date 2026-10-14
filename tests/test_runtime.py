from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from svae.runtime import backend_name, resolve_device, validate_training_precision


def gpu_controls(monkeypatch, kind, *, available=True, current=1, count=2, bf16=True, fp64=True):
    module = getattr(torch, kind)
    monkeypatch.setattr(module, "is_available", lambda: available)
    monkeypatch.setattr(module, "current_device", lambda: current)
    monkeypatch.setattr(module, "device_count", lambda: count)
    monkeypatch.setattr(module, "device", lambda *args: nullcontext())
    monkeypatch.setattr(module, "is_bf16_supported", lambda: bf16)
    if kind == "xpu":
        monkeypatch.setattr(module, "get_device_properties", lambda *args: SimpleNamespace(has_fp16=True, has_fp64=fp64))


def test_auto_selection_keeps_current_gpu_and_rejects_out_of_range(monkeypatch):
    gpu_controls(monkeypatch, "cuda")
    gpu_controls(monkeypatch, "xpu", current=2, count=3)
    assert resolve_device("auto") == torch.device("cuda:1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device() == torch.device("xpu:2")
    assert resolve_device("xpu:0") == torch.device("xpu:0")
    with pytest.raises(ValueError):
        resolve_device("xpu:3")
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    assert resolve_device() == torch.device("cpu")


def test_explicit_unavailable_accelerators_do_not_fall_back(monkeypatch):
    gpu_controls(monkeypatch, "cuda", available=False)
    gpu_controls(monkeypatch, "xpu", available=False)
    monkeypatch.setattr(torch.version, "hip", None)
    for requested in ("cuda", "xpu", "rocm"):
        with pytest.raises(ValueError):
            resolve_device(requested)
    assert resolve_device("cpu") == torch.device("cpu")


def test_rocm_selection_requires_hip_and_resolves_its_cuda_api(monkeypatch):
    gpu_controls(monkeypatch, "cuda")
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(ValueError):
        resolve_device("rocm:0")
    monkeypatch.setattr(torch.version, "hip", "10.1.0")
    assert resolve_device("rocm:0") == torch.device("cuda:0")
    assert resolve_device("rocm") == torch.device("cuda:1")
    assert backend_name(resolve_device("rocm")) == "rocm"


@pytest.mark.parametrize("requested", ["mps", "meta", "xpu:-1", "invalid", ""])
def test_unsupported_device_requests_are_errors(requested):
    with pytest.raises(ValueError):
        resolve_device(requested)


def test_unsafe_training_precision_is_rejected_before_model_transfer(monkeypatch):
    with pytest.raises(ValueError):
        validate_training_precision(torch.device("cpu"), "fp16")
    gpu_controls(monkeypatch, "cuda", bf16=False)
    with pytest.raises(ValueError):
        validate_training_precision(torch.device("cuda:1"), "bf16")
    gpu_controls(monkeypatch, "xpu", fp64=False)
    with pytest.raises(ValueError):
        validate_training_precision(torch.device("xpu:1"), "fp16")
