# convert the xlab lora to diffusers format
import os

import torch
from safetensors.torch import save_file

from ...utils import load_state_dict_in_safetensors


def xlab2diffusers(
    input_lora: str | dict[str, torch.Tensor], output_path: str | None = None
) -> dict[str, torch.Tensor]:
    if isinstance(input_lora, str):
        tensors = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        tensors = input_lora

    new_tensors = {}

    # lora1 is for img, lora2 is for text
    for k, v in tensors.items():
        assert "double_blocks" in k
        new_k = k.replace("double_blocks", "transformer.transformer_blocks").replace("processor", "attn")
        new_k = new_k.replace(".down.", ".lora_A.")
        new_k = new_k.replace(".up.", ".lora_B.")
        if ".proj_lora" in new_k:
            new_k = new_k.replace(".proj_lora1", ".to_out.0")
            new_k = new_k.replace(".proj_lora2", ".to_add_out")
            new_tensors[new_k] = v
        else:
            assert "qkv_lora" in new_k
            if "lora_A" in new_k:
                for p in ["q", "k", "v"]:
                    if ".qkv_lora1." in new_k:
                        new_tensors[new_k.replace(".qkv_lora1.", f".to_{p}.")] = v.clone()
                    else:
                        assert ".qkv_lora2." in new_k
                        new_tensors[new_k.replace(".qkv_lora2.", f".add_{p}_proj.")] = v.clone()
            else:
                assert "lora_B" in new_k
                for i, p in enumerate(["q", "k", "v"]):
                    assert v.shape[0] % 3 == 0
                    chunk_size = v.shape[0] // 3
                    if ".qkv_lora1." in new_k:
                        new_tensors[new_k.replace(".qkv_lora1.", f".to_{p}.")] = v[
                            i * chunk_size : (i + 1) * chunk_size
                        ]
                    else:
                        assert ".qkv_lora2." in new_k
                        new_tensors[new_k.replace(".qkv_lora2.", f".add_{p}_proj.")] = v[
                            i * chunk_size : (i + 1) * chunk_size
                        ]
    if output_path is not None:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        save_file(new_tensors, output_path)
    return new_tensors
