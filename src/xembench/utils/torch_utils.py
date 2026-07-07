"""Device selection and small tensor helpers."""

import torch


def auto_device(requested: str = "auto") -> torch.device:
    """cuda -> mps -> cpu, unless a specific device is requested."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: dict, device: torch.device) -> dict:
    # non_blocking copies are only safe with CUDA pinned-memory semantics;
    # on MPS they race with numpy-backed source buffers and deliver garbage.
    nb = device.type == "cuda"
    return {
        k: (v.to(device, non_blocking=nb) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
