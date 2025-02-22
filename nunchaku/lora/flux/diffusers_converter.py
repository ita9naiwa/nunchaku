# convert the diffusers lora to nunchaku format
"""Convert LoRA weights to Nunchaku format."""

import typing as tp

import torch
import tqdm

from ...utils import ceil_divide, filter_state_dict, load_state_dict_in_safetensors


# region utilities


def pad(
    tensor: tp.Optional[torch.Tensor],
    divisor: int | tp.Sequence[int],
    dim: int | tp.Sequence[int],
    fill_value: float | int = 0,
) -> torch.Tensor | None:
    if isinstance(divisor, int):
        if divisor <= 1:
            return tensor
    elif all(d <= 1 for d in divisor):
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if isinstance(dim, int):
        assert isinstance(divisor, int)
        shape[dim] = ceil_divide(shape[dim], divisor) * divisor
    else:
        if isinstance(divisor, int):
            divisor = [divisor] * len(dim)
        for d, div in zip(dim, divisor, strict=True):
            shape[d] = ceil_divide(shape[d], div) * div
    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    result[[slice(0, extent) for extent in tensor.shape]] = tensor
    return result


def update_state_dict(
    lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor], prefix: str = ""
) -> dict[str, torch.Tensor]:
    for rkey, value in rhs.items():
        lkey = f"{prefix}.{rkey}" if prefix else rkey
        assert lkey not in lhs, f"Key {lkey} already exists in the state dict."
        lhs[lkey] = value
    return lhs


# endregion


def pack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """Pack Low-Rank Weight.

    Args:
        weight (`torch.Tensor`):
            low-rank weight tensor.
        down (`bool`):
            whether the weight is for down projection in low-rank branch.
    """
    assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
    lane_n, lane_k = 1, 2  # lane_n is always 1, lane_k is 32 bits // 16 bits = 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k
    weight = pad(weight, divisor=(frag_n, frag_k), dim=(0, 1))
    if down:
        r, c = weight.shape
        r_frags, c_frags = r // frag_n, c // frag_k
        weight = weight.view(r_frags, frag_n, c_frags, frag_k).permute(2, 0, 1, 3)
    else:
        c, r = weight.shape
        c_frags, r_frags = c // frag_n, r // frag_k
        weight = weight.view(c_frags, frag_n, r_frags, frag_k).permute(0, 2, 1, 3)
    weight = weight.reshape(c_frags, r_frags, n_pack_size, num_n_lanes, k_pack_size, num_k_lanes, lane_k)
    weight = weight.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return weight.view(c, r)


def unpack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """Unpack Low-Rank Weight.

    Args:
        weight (`torch.Tensor`):
            low-rank weight tensor.
        down (`bool`):
            whether the weight is for down projection in low-rank branch.
    """
    c, r = weight.shape
    assert weight.dtype in (torch.float16, torch.bfloat16), f"Unsupported weight dtype {weight.dtype}."
    lane_n, lane_k = 1, 2  # lane_n is always 1, lane_k is 32 bits // 16 bits = 2
    n_pack_size, k_pack_size = 2, 2
    num_n_lanes, num_k_lanes = 8, 4
    frag_n = n_pack_size * num_n_lanes * lane_n
    frag_k = k_pack_size * num_k_lanes * lane_k
    if down:
        r_frags, c_frags = r // frag_n, c // frag_k
    else:
        c_frags, r_frags = c // frag_n, r // frag_k
    weight = weight.view(c_frags, r_frags, num_n_lanes, num_k_lanes, n_pack_size, k_pack_size, lane_k)
    weight = weight.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    weight = weight.view(c_frags, r_frags, frag_n, frag_k)
    if down:
        weight = weight.permute(1, 2, 0, 3).contiguous().view(r, c)
    else:
        weight = weight.permute(0, 2, 1, 3).contiguous().view(c, r)
    return weight


def reorder_adanorm_lora_up(lora_up: torch.Tensor, splits: int) -> torch.Tensor:
    c, r = lora_up.shape
    assert c % splits == 0
    return lora_up.view(splits, c // splits, r).transpose(0, 1).reshape(c, r).contiguous()


def convert_to_nunchaku_transformer_block_lowrank_dict(  # noqa: C901
    orig_state_dict: dict[str, torch.Tensor],
    extra_lora_dict: dict[str, torch.Tensor],
    converted_block_name: str,
    candidate_block_name: str,
    local_name_map: dict[str, str | list[str]],
    convert_map: dict[str, str],
    default_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    print(f"Converting LoRA branch for block {candidate_block_name}...")
    converted: dict[str, torch.Tensor] = {}
    for converted_local_name, candidate_local_names in tqdm.tqdm(
        local_name_map.items(), desc=f"Converting {candidate_block_name}", dynamic_ncols=True
    ):
        if isinstance(candidate_local_names, str):
            candidate_local_names = [candidate_local_names]
        # region original LoRA
        orig_lora = (
            orig_state_dict.get(f"{converted_block_name}.{converted_local_name}.lora_down", None),
            orig_state_dict.get(f"{converted_block_name}.{converted_local_name}.lora_up", None),
        )
        if orig_lora[0] is None or orig_lora[1] is None:
            assert orig_lora[0] is None and orig_lora[1] is None
            orig_lora = None
        else:
            assert orig_lora[0] is not None and orig_lora[1] is not None
            orig_lora = (
                unpack_lowrank_weight(orig_lora[0], down=True),
                unpack_lowrank_weight(orig_lora[1], down=False),
            )
            print(f" - Found {converted_block_name} LoRA of {converted_local_name} (rank: {orig_lora[0].shape[0]})")
        # endregion
        # region extra LoRA
        extra_lora = [
            (
                extra_lora_dict.get(f"{candidate_block_name}.{candidate_local_name}.lora_A.weight", None),
                extra_lora_dict.get(f"{candidate_block_name}.{candidate_local_name}.lora_B.weight", None),
            )
            for candidate_local_name in candidate_local_names
        ]
        if any(lora[0] is not None or lora[1] is not None for lora in extra_lora):
            # merge extra LoRAs into one LoRA
            if len(extra_lora) > 1:
                first_lora = None
                for lora in extra_lora:
                    if lora[0] is not None:
                        assert lora[1] is not None
                        first_lora = lora
                        break
                assert first_lora is not None
                for lora_index in range(len(extra_lora)):
                    if extra_lora[lora_index][0] is None:
                        assert extra_lora[lora_index][1] is None
                        extra_lora[lora_index] = (first_lora[0].clone(), torch.zeros_like(first_lora[1]))
                if all(lora[0].equal(extra_lora[0][0]) for lora in extra_lora):
                    # if all extra LoRAs have the same lora_down, use it
                    extra_lora_down = extra_lora[0][0]
                    extra_lora_up = torch.cat([lora[1] for lora in extra_lora], dim=0)
                else:
                    extra_lora_down = torch.cat([lora[0] for lora in extra_lora], dim=0)
                    extra_lora_up_c = sum(lora[1].shape[0] for lora in extra_lora)
                    extra_lora_up_r = sum(lora[1].shape[1] for lora in extra_lora)
                    assert extra_lora_up_r == extra_lora_down.shape[0]
                    extra_lora_up = torch.zeros((extra_lora_up_c, extra_lora_up_r), dtype=extra_lora_down.dtype)
                    c, r = 0, 0
                    for lora in extra_lora:
                        c_next, r_next = c + lora[1].shape[0], r + lora[1].shape[1]
                        extra_lora_up[c:c_next, r:r_next] = lora[1]
                        c, r = c_next, r_next
            else:
                extra_lora_down, extra_lora_up = extra_lora[0]
            extra_lora: tuple[torch.Tensor, torch.Tensor] = (extra_lora_down, extra_lora_up)
            print(f" - Found {candidate_block_name} LoRA of {candidate_local_names} (rank: {extra_lora[0].shape[0]})")
        else:
            extra_lora = None
        # endregion
        # region merge LoRA
        if orig_lora is None:
            if extra_lora is None:
                lora = None
            else:
                print("    - Using extra LoRA")
                lora = (extra_lora[0].to(default_dtype), extra_lora[1].to(default_dtype))
        elif extra_lora is None:
            print("    - Using original LoRA")
            lora = orig_lora
        else:
            lora = (
                torch.cat([orig_lora[0], extra_lora[0].to(orig_lora[0].dtype)], dim=0),
                torch.cat([orig_lora[1], extra_lora[1].to(orig_lora[1].dtype)], dim=1),
            )
            print(f"    - Merging original and extra LoRA (rank: {lora[0].shape[0]})")
        # endregion
        if lora is not None:
            if convert_map[converted_local_name] == "adanorm_single":
                update_state_dict(
                    converted,
                    {
                        "lora_down": lora[0],
                        "lora_up": reorder_adanorm_lora_up(lora[1], splits=3),
                    },
                    prefix=converted_local_name,
                )
            elif convert_map[converted_local_name] == "adanorm_zero":
                update_state_dict(
                    converted,
                    {
                        "lora_down": lora[0],
                        "lora_up": reorder_adanorm_lora_up(lora[1], splits=6),
                    },
                    prefix=converted_local_name,
                )
            elif convert_map[converted_local_name] == "linear":
                update_state_dict(
                    converted,
                    {
                        "lora_down": pack_lowrank_weight(lora[0], down=True),
                        "lora_up": pack_lowrank_weight(lora[1], down=False),
                    },
                    prefix=converted_local_name,
                )
    return converted


def convert_to_nunchaku_flux_single_transformer_block_lowrank_dict(
    orig_state_dict: dict[str, torch.Tensor],
    extra_lora_dict: dict[str, torch.Tensor],
    converted_block_name: str,
    candidate_block_name: str,
    default_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    if f"{candidate_block_name}.proj_out.lora_A.weight" in extra_lora_dict:
        assert f"{converted_block_name}.out_proj.qweight" in orig_state_dict
        assert f"{converted_block_name}.mlp_fc2.qweight" in orig_state_dict
        n1 = orig_state_dict[f"{converted_block_name}.out_proj.qweight"].shape[1] * 2
        n2 = orig_state_dict[f"{converted_block_name}.mlp_fc2.qweight"].shape[1] * 2
        lora_down = extra_lora_dict[f"{candidate_block_name}.proj_out.lora_A.weight"]
        lora_up = extra_lora_dict[f"{candidate_block_name}.proj_out.lora_B.weight"]
        assert lora_down.shape[1] == n1 + n2
        extra_lora_dict[f"{candidate_block_name}.proj_out.linears.0.lora_A.weight"] = lora_down[:, :n1].clone()
        extra_lora_dict[f"{candidate_block_name}.proj_out.linears.0.lora_B.weight"] = lora_up.clone()
        extra_lora_dict[f"{candidate_block_name}.proj_out.linears.1.lora_A.weight"] = lora_down[:, n1:].clone()
        extra_lora_dict[f"{candidate_block_name}.proj_out.linears.1.lora_B.weight"] = lora_up.clone()
        extra_lora_dict.pop(f"{candidate_block_name}.proj_out.lora_A.weight")
        extra_lora_dict.pop(f"{candidate_block_name}.proj_out.lora_B.weight")

    return convert_to_nunchaku_transformer_block_lowrank_dict(
        orig_state_dict=orig_state_dict,
        extra_lora_dict=extra_lora_dict,
        converted_block_name=converted_block_name,
        candidate_block_name=candidate_block_name,
        local_name_map={
            "norm.linear": "norm.linear",
            "qkv_proj": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "proj_mlp",
            "mlp_fc2": "proj_out.linears.1",
        },
        convert_map={
            "norm.linear": "adanorm_single",
            "qkv_proj": "linear",
            "out_proj": "linear",
            "mlp_fc1": "linear",
            "mlp_fc2": "linear",
        },
        default_dtype=default_dtype,
    )


def convert_to_nunchaku_flux_transformer_block_lowrank_dict(
    orig_state_dict: dict[str, torch.Tensor],
    extra_lora_dict: dict[str, torch.Tensor],
    converted_block_name: str,
    candidate_block_name: str,
    default_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    return convert_to_nunchaku_transformer_block_lowrank_dict(
        orig_state_dict=orig_state_dict,
        extra_lora_dict=extra_lora_dict,
        converted_block_name=converted_block_name,
        candidate_block_name=candidate_block_name,
        local_name_map={
            "norm1.linear": "norm1.linear",
            "norm1_context.linear": "norm1_context.linear",
            "qkv_proj": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "qkv_proj_context": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "norm_added_q": "attn.norm_added_q",
            "norm_added_k": "attn.norm_added_k",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_add_out",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": "ff.net.2",
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": "ff_context.net.2",
        },
        convert_map={
            "norm1.linear": "adanorm_zero",
            "norm1_context.linear": "adanorm_zero",
            "qkv_proj": "linear",
            "qkv_proj_context": "linear",
            "out_proj": "linear",
            "out_proj_context": "linear",
            "mlp_fc1": "linear",
            "mlp_fc2": "linear",
            "mlp_context_fc1": "linear",
            "mlp_context_fc2": "linear",
        },
        default_dtype=default_dtype,
    )


def convert_to_nunchaku_flux_lowrank_dict(
    base_model: dict[str, torch.Tensor] | str,
    lora: dict[str, torch.Tensor] | str,
    default_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    if isinstance(base_model, str):
        orig_state_dict = load_state_dict_in_safetensors(base_model)
    else:
        orig_state_dict = base_model

    if isinstance(lora, str):
        extra_lora_dict = load_state_dict_in_safetensors(lora, filter_prefix="transformer.")
    else:
        extra_lora_dict = filter_state_dict(lora, filter_prefix="transformer.")

    block_names: set[str] = set()
    for param_name in orig_state_dict.keys():
        if param_name.startswith(("transformer_blocks.", "single_transformer_blocks.")):
            block_names.add(".".join(param_name.split(".")[:2]))
    block_names = sorted(block_names, key=lambda x: (x.split(".")[0], int(x.split(".")[-1])))
    print(f"Converting {len(block_names)} transformer blocks...")
    converted: dict[str, torch.Tensor] = {}
    for block_name in block_names:
        if block_name.startswith("transformer_blocks"):
            convert_fn = convert_to_nunchaku_flux_transformer_block_lowrank_dict
        else:
            convert_fn = convert_to_nunchaku_flux_single_transformer_block_lowrank_dict
        update_state_dict(
            converted,
            convert_fn(
                orig_state_dict=orig_state_dict,
                extra_lora_dict=extra_lora_dict,
                converted_block_name=block_name,
                candidate_block_name=block_name,
                default_dtype=default_dtype,
            ),
            prefix=block_name,
        )
    return converted
