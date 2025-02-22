import os

import torch
from diffusers import __version__
from huggingface_hub import constants, hf_hub_download
from safetensors.torch import load_file
from typing import Optional, Any


class NunchakuModelLoaderMixin:
    @classmethod
    def _build_model(cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs):
        subfolder = kwargs.get("subfolder", None)
        if os.path.exists(pretrained_model_name_or_path):
            dirname = (
                pretrained_model_name_or_path
                if subfolder is None
                else os.path.join(pretrained_model_name_or_path, subfolder)
            )
            unquantized_part_path = os.path.join(dirname, "unquantized_layers.safetensors")
            transformer_block_path = os.path.join(dirname, "transformer_blocks.safetensors")
        else:
            download_kwargs = {
                "subfolder": subfolder,
                "repo_type": "model",
                "revision": kwargs.get("revision", None),
                "cache_dir": kwargs.get("cache_dir", None),
                "local_dir": kwargs.get("local_dir", None),
                "user_agent": kwargs.get("user_agent", None),
                "force_download": kwargs.get("force_download", False),
                "proxies": kwargs.get("proxies", None),
                "etag_timeout": kwargs.get("etag_timeout", constants.DEFAULT_ETAG_TIMEOUT),
                "token": kwargs.get("token", None),
                "local_files_only": kwargs.get("local_files_only", None),
                "headers": kwargs.get("headers", None),
                "endpoint": kwargs.get("endpoint", None),
                "resume_download": kwargs.get("resume_download", None),
                "force_filename": kwargs.get("force_filename", None),
                "local_dir_use_symlinks": kwargs.get("local_dir_use_symlinks", "auto"),
            }
            unquantized_part_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename="unquantized_layers.safetensors", **download_kwargs
            )
            transformer_block_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename="transformer_blocks.safetensors", **download_kwargs
            )

        config, _, _ = cls.load_config(
            pretrained_model_name_or_path,
            subfolder=subfolder,
            cache_dir=kwargs.get("cache_dir", None),
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=kwargs.get("force_download", False),
            proxies=kwargs.get("proxies", None),
            local_files_only=kwargs.get("local_files_only", None),
            token=kwargs.get("token", None),
            revision=kwargs.get("revision", None),
            user_agent={"diffusers": __version__, "file_type": "model", "framework": "pytorch"},
            **kwargs,
        )

        transformer = cls.from_config(config).to(kwargs.get("torch_dtype", torch.bfloat16))
        state_dict = load_file(unquantized_part_path)
        transformer.load_state_dict(state_dict, strict=False)

        return transformer, transformer_block_path

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y

def pad_tensor(tensor: Optional[torch.Tensor], multiples: int, dim: int, fill: Any = 0) -> torch.Tensor:
    if multiples <= 1:
        return tensor
    if tensor is None:
        return None
    shape = list(tensor.shape)
    if shape[dim] % multiples == 0:
        return tensor
    shape[dim] = ceil_div(shape[dim], multiples) * multiples
    result = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    result.fill_(fill)
    result[[slice(0, extent) for extent in tensor.shape]] = tensor
    return result

class CachedTransformerBlocks(torch.nn.Module):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        residual_diff_threshold,
        return_hidden_states_first=True,
        return_hidden_states_only=False,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.residual_diff_threshold = residual_diff_threshold
        self.return_hidden_states_first = return_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        if self.residual_diff_threshold <= 0.0:
            for block in self.transformer_blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
                if not isinstance(hidden_states, torch.Tensor):
                    hidden_states, encoder_hidden_states = hidden_states
                    if not self.return_hidden_states_first:
                        hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
            if self.single_transformer_blocks is not None:
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                for block in self.single_transformer_blocks:
                    hidden_states = block(hidden_states, *args, **kwargs)
                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :]
            return (
                hidden_states
                if self.return_hidden_states_only
                else (
                    (hidden_states, encoder_hidden_states)
                    if self.return_hidden_states_first
                    else (encoder_hidden_states, hidden_states)
                )
            )

        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]
        hidden_states = first_transformer_block(hidden_states, encoder_hidden_states, *args, **kwargs)
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states, encoder_hidden_states = hidden_states
            if not self.return_hidden_states_first:
                hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        first_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        can_use_cache = get_can_use_cache(
            first_hidden_states_residual,
            threshold=self.residual_diff_threshold,
            parallelized=self.transformer is not None and getattr(self.transformer, "_is_parallelized", False),
        )

        torch._dynamo.graph_break()
        if can_use_cache:
            del first_hidden_states_residual
            hidden_states, encoder_hidden_states = apply_prev_hidden_states_residual(
                hidden_states, encoder_hidden_states
            )
        else:
            set_buffer("first_hidden_states_residual", first_hidden_states_residual)
            del first_hidden_states_residual
            (
                hidden_states,
                encoder_hidden_states,
                hidden_states_residual,
                encoder_hidden_states_residual,
            ) = self.call_remaining_transformer_blocks(hidden_states, encoder_hidden_states, *args, **kwargs)
            set_buffer("hidden_states_residual", hidden_states_residual)
            set_buffer("encoder_hidden_states_residual", encoder_hidden_states_residual)
        torch._dynamo.graph_break()

        return (
            hidden_states
            if self.return_hidden_states_only
            else (
                (hidden_states, encoder_hidden_states)
                if self.return_hidden_states_first
                else (encoder_hidden_states, hidden_states)
            )
        )

    def call_remaining_transformer_blocks(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        for block in self.transformer_blocks[1:]:
            hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
            if not isinstance(hidden_states, torch.Tensor):
                hidden_states, encoder_hidden_states = hidden_states
                if not self.return_hidden_states_first:
                    hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        if self.single_transformer_blocks is not None:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            for block in self.single_transformer_blocks:
                hidden_states = block(hidden_states, *args, **kwargs)
            encoder_hidden_states, hidden_states = hidden_states.split(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )

        # hidden_states_shape = hidden_states.shape
        # encoder_hidden_states_shape = encoder_hidden_states.shape
        hidden_states = hidden_states.reshape(-1).contiguous().reshape(original_hidden_states.shape)
        encoder_hidden_states = (
            encoder_hidden_states.reshape(-1).contiguous().reshape(original_encoder_hidden_states.shape)
        )

        # hidden_states = hidden_states.contiguous()
        # encoder_hidden_states = encoder_hidden_states.contiguous()

        hidden_states_residual = hidden_states - original_hidden_states
        encoder_hidden_states_residual = encoder_hidden_states - original_encoder_hidden_states

        hidden_states_residual = hidden_states_residual.reshape(-1).contiguous().reshape(original_hidden_states.shape)
        encoder_hidden_states_residual = (
            encoder_hidden_states_residual.reshape(-1).contiguous().reshape(original_encoder_hidden_states.shape)
        )

        return hidden_states, encoder_hidden_states, hidden_states_residual, encoder_hidden_states_residual
