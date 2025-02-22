import torch
from diffusers import FluxPipeline

from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-schnell")
pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
image = pipeline(
    "A cat holding a sign that says hello world", width=1024, height=1024, num_inference_steps=4, guidance_scale=0
).images[0]
image.save("flux.1-schnell.png")
