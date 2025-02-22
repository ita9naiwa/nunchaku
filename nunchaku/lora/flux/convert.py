import argparse
import os

import torch
from safetensors.torch import save_file

from .comfyui_converter import comfyui2diffusers
from .diffusers_converter import convert_to_nunchaku_flux_lowrank_dict
from .xlab_converter import xlab2diffusers
from ...utils import filter_state_dict, load_state_dict_in_safetensors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quant-path",
        type=str,
        help="path to the quantized model safetensor file",
        default="mit-han-lab/svdq-int4-flux.1-dev/transformer_blocks.safetensors",
    )
    parser.add_argument("--lora-path", type=str, required=True, help="path to LoRA weights safetensor file")
    parser.add_argument(
        "--lora-format",
        type=str,
        default="diffusers",
        choices=["comfyui", "diffusers", "xlab"],
        help="format of the LoRA weights",
    )
    parser.add_argument("--output-root", type=str, default="", help="root to the output safetensor file")
    parser.add_argument("--lora-name", type=str, default=None, help="name of the LoRA weights")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="data type of the converted weights",
    )
    args = parser.parse_args()

    if not args.output_root:
        # output to the parent directory of the quantized model safetensor file
        args.output_root = os.path.dirname(args.quant_path)
    if args.lora_name is None:
        base_name = os.path.basename(args.lora_path)
        lora_name = base_name.rsplit(".", 1)[0]
        lora_name = "svdq-int4-" + lora_name
        print(f"LoRA name not provided, using {lora_name} as the LoRA name")
    else:
        lora_name = args.lora_name
    assert lora_name, "LoRA name must be provided."

    assert args.quant_path.endswith(".safetensors"), "Quantized model must be a safetensor file"
    assert args.lora_path.endswith(".safetensors"), "LoRA weights must be a safetensor file"
    orig_state_dict = load_state_dict_in_safetensors(args.quant_path)
    lora_format = args.lora_format

    if lora_format == "diffusers":
        extra_lora_dict = load_state_dict_in_safetensors(args.lora_path)
    else:
        if lora_format == "comfyui":
            extra_lora_dict = comfyui2diffusers(args.lora_path)
        elif lora_format == "xlab":
            extra_lora_dict = xlab2diffusers(args.lora_path)
        else:
            raise NotImplementedError(f"LoRA format {lora_format} is not supported.")
        extra_lora_dict = filter_state_dict(extra_lora_dict)

    converted = convert_to_nunchaku_flux_lowrank_dict(
        base_model=orig_state_dict,
        lora=extra_lora_dict,
        default_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
    )
    os.makedirs(args.output_root, exist_ok=True)
    save_file(converted, os.path.join(args.output_root, f"{lora_name}.safetensors"))
    print(f"Saved LoRA weights to {args.output_root}.")
