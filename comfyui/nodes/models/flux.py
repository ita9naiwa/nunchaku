import os

import comfy.model_patcher
import folder_paths
import GPUtil
import torch
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.supported_models import Flux, FluxSchnell
from diffusers import FluxTransformer2DModel
from einops import rearrange, repeat
from torch import nn

from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel


class ComfyUIFluxForwardWrapper(nn.Module):
    def __init__(self, model: NunchakuFluxTransformer2dModel, config):
        super(ComfyUIFluxForwardWrapper, self).__init__()
        self.model = model
        self.dtype = next(model.parameters()).dtype
        self.config = config

    def forward(
        self,
        x,
        timestep,
        context,
        y,
        guidance,
        control=None,
        transformer_options={},
        **kwargs,
    ):
        assert control is None  # for now
        bs, c, h, w = x.shape
        patch_size = self.config["patch_size"]
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = (h + (patch_size // 2)) // patch_size
        w_len = (w + (patch_size // 2)) // patch_size
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(
            0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype
        ).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(
            0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype
        ).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        out = self.model(
            hidden_states=img,
            encoder_hidden_states=context,
            pooled_projections=y,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance if self.config["guidance_embed"] else None,
        ).sample

        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:, :, :h, :w]
        return out


class SVDQuantFluxDiTLoader:
    @classmethod
    def INPUT_TYPES(s):
        # folder_paths.get_filename_list("loras"),
        model_paths = [
            "mit-han-lab/svdq-int4-flux.1-schnell",
            "mit-han-lab/svdq-int4-flux.1-dev",
            "mit-han-lab/svdq-int4-flux.1-canny-dev",
            "mit-han-lab/svdq-int4-flux.1-depth-dev",
            "mit-han-lab/svdq-int4-flux.1-fill-dev",
        ]
        prefix = os.path.join(folder_paths.models_dir, "diffusion_models")
        local_folders = os.listdir(prefix)
        local_folders = sorted(
            [
                folder
                for folder in local_folders
                if not folder.startswith(".") and os.path.isdir(os.path.join(prefix, folder))
            ]
        )
        model_paths = local_folders + model_paths
        ngpus = len(GPUtil.getGPUs())
        return {
            "required": {
                "model_path": (model_paths,),
                "device_id": (
                    "INT",
                    {"default": 0, "min": 0, "max": ngpus, "step": 1, "display": "number", "lazy": True},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "SVDQuant"
    TITLE = "SVDQuant Flux DiT Loader"

    def load_model(self, model_path: str, device_id: int, **kwargs) -> tuple[FluxTransformer2DModel]:
        device = f"cuda:{device_id}"
        prefix = os.path.join(folder_paths.models_dir, "diffusion_models")
        if os.path.exists(os.path.join(prefix, model_path)):
            model_path = os.path.join(prefix, model_path)
        else:
            model_path = model_path
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(model_path).to(device)
        dit_config = {
            "image_model": "flux",
            "patch_size": 2,
            "out_channels": 16,
            "vec_in_dim": 768,
            "context_in_dim": 4096,
            "hidden_size": 3072,
            "mlp_ratio": 4.0,
            "num_heads": 24,
            "depth": 19,
            "depth_single_blocks": 38,
            "axes_dim": [16, 56, 56],
            "theta": 10000,
            "qkv_bias": True,
            "guidance_embed": True,
            "disable_unet_model_creation": True,
        }

        if "schnell" in model_path:
            dit_config["guidance_embed"] = False
            dit_config["in_channels"] = 16
            model_config = FluxSchnell(dit_config)
        elif "canny" in model_path or "depth" in model_path:
            dit_config["in_channels"] = 32
            model_config = Flux(dit_config)
        elif "fill" in model_path:
            dit_config["in_channels"] = 64
            model_config = Flux(dit_config)
        else:
            dit_config["in_channels"] = 16
            model_config = Flux(dit_config)

        model_config.set_inference_dtype(torch.bfloat16, None)
        model_config.custom_operations = None

        model = model_config.get_model({})
        model.diffusion_model = ComfyUIFluxForwardWrapper(transformer, config=dit_config)
        model = comfy.model_patcher.ModelPatcher(model, device, device_id)
        return (model,)
