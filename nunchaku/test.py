import torch
from diffusers import FluxPipeline

from .models.transformer_flux import NunchakuFluxTransformer2dModel

if __name__ == "__main__":
    capability = torch.cuda.get_device_capability(0)
    sm = f"{capability[0]}{capability[1]}"
    precision = "fp4" if sm == "120" else "int4"

    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"mit-han-lab/svdq-{precision}-flux.1-schnell", precision=precision
    )
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", transformer=transformer, torch_dtype=torch.bfloat16
    ).to("cuda")
    image = pipeline(
        "A cat holding a sign that says hello world", width=1024, height=1024, num_inference_steps=4, guidance_scale=0
    ).images[0]
