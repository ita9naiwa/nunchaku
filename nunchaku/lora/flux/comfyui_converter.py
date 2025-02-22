# convert the comfyui lora to diffusers format
import os

import torch
from safetensors.torch import save_file

from ...utils import load_state_dict_in_safetensors


def comfyui2diffusers(
    input_lora: str | dict[str, torch.Tensor], output_path: str | None = None
) -> dict[str, torch.Tensor]:
    if isinstance(input_lora, str):
        tensors = load_state_dict_in_safetensors(input_lora, device="cpu")
    else:
        tensors = input_lora

    new_tensors = {}

    for k, v in tensors.items():
        if "alpha" in k:
            continue
        new_k = k.replace("lora_down", "lora_A").replace("lora_up", "lora_B")
        if "lora_unet_double_blocks_" in k:
            new_k = new_k.replace("lora_unet_double_blocks_", "transformer.transformer_blocks.")
            if "qkv" in new_k:
                for i, p in enumerate(["q", "k", "v"]):
                    if "lora_A" in new_k:
                        # Copy the tensor
                        new_k = new_k.replace("_img_attn_qkv", f".attn.to_{p}")
                        new_k = new_k.replace("_txt_attn_qkv", f".attn.add_{p}_proj")
                        new_tensors[new_k] = v.clone()
                    else:
                        assert "lora_B" in new_k
                        assert v.shape[0] % 3 == 0
                        chunk_size = v.shape[0] // 3
                        new_k = new_k.replace("_img_attn_qkv", f".attn.to_{p}")
                        new_k = new_k.replace("_txt_attn_qkv", f".attn.add_{p}_proj")
                        new_tensors[new_k] = v[i * chunk_size : (i + 1) * chunk_size]
            else:
                new_k = new_k.replace("_img_attn_proj", ".attn.to_out.0")
                new_k = new_k.replace("_img_mlp_0", ".ff.net.0.proj")
                new_k = new_k.replace("_img_mlp_2", ".ff.net.2")
                new_k = new_k.replace("_img_mod_lin", ".norm1.linear")
                new_k = new_k.replace("_txt_attn_proj", ".attn.to_add_out")
                new_k = new_k.replace("_txt_mlp_0", ".ff_context.net.0.proj")
                new_k = new_k.replace("_txt_mlp_2", ".ff_context.net.2")
                new_k = new_k.replace("_txt_mod_lin", ".norm1_context.linear")
                new_tensors[new_k] = v
        else:
            assert "lora_unet_single_blocks" in k
            new_k = new_k.replace("lora_unet_single_blocks_", "transformer.single_transformer_blocks.")
            if "linear1" in k:
                start = 0
                for i, p in enumerate(["q", "k", "v", "i"]):
                    if "lora_A" in new_k:
                        if p == "i":
                            new_k1 = new_k.replace("_linear1", ".proj_mlp")
                        else:
                            new_k1 = new_k.replace("_linear1", f".attn.to_{p}")
                        new_tensors[new_k1] = v.clone()
                    else:
                        if p == "i":
                            new_k1 = new_k.replace("_linear1", ".proj_mlp")
                        else:
                            new_k1 = new_k.replace("_linear1", f".attn.to_{p}")
                        chunk_size = 12288 if p == "i" else 3072
                        new_tensors[new_k1] = v[start : start + chunk_size]
                        start += chunk_size
            else:
                new_k = new_k.replace("_linear2", ".proj_out")
                new_k = new_k.replace("_modulation_lin", ".norm.linear")
                new_tensors[new_k] = v

    if output_path is not None:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)
        save_file(new_tensors, output_path)
    return new_tensors
