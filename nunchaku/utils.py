import os

import safetensors
import torch
from huggingface_hub import hf_hub_download


def fetch_or_download(path: str) -> str:
    if not os.path.exists(path):
        hf_repo_id = os.path.dirname(path)
        filename = os.path.basename(path)
        path = hf_hub_download(repo_id=hf_repo_id, filename=filename)
    return path


def ceil_divide(x: int, divisor: int) -> int:
    """Ceiling division.

    Args:
        x (`int`):
            dividend.
        divisor (`int`):
            divisor.

    Returns:
        `int`:
            ceiling division result.
    """
    return (x + divisor - 1) // divisor


def load_state_dict_in_safetensors(
    path: str, device: str | torch.device = "cpu", filter_prefix: str = ""
) -> dict[str, torch.Tensor]:
    """Load state dict in SafeTensors.

    Args:
        path (`str`):
            file path.
        device (`str` | `torch.device`, optional, defaults to `"cpu"`):
            device.
        filter_prefix (`str`, optional, defaults to `""`):
            filter prefix.

    Returns:
        `dict`:
            loaded SafeTensors.
    """
    state_dict = {}
    with safetensors.safe_open(fetch_or_download(path), framework="pt", device=device) as f:
        for k in f.keys():
            if filter_prefix and not k.startswith(filter_prefix):
                continue
            state_dict[k.removeprefix(filter_prefix)] = f.get_tensor(k)
    return state_dict


def filter_state_dict(state_dict: dict[str, torch.Tensor], filter_prefix: str = "") -> dict[str, torch.Tensor]:
    """Filter state dict.

    Args:
        state_dict (`dict`):
            state dict.
        filter_prefix (`str`):
            filter prefix.

    Returns:
        `dict`:
            filtered state dict.
    """
    return {k.removeprefix(filter_prefix): v for k, v in state_dict.items() if k.startswith(filter_prefix)}
