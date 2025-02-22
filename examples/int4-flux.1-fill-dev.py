import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image

from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

image = load_image("https://huggingface.co/mit-han-lab/svdq-int4-flux.1-fill-dev/resolve/main/example.png")
mask = load_image("https://huggingface.co/mit-han-lab/svdq-int4-flux.1-fill-dev/resolve/main/mask.png")

transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-fill-dev")
pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev", transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")
image = pipe(
    prompt="A wooden basket of a cat.",
    image=image,
    mask_image=mask,
    height=1024,
    width=1024,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
).images[0]
image.save("flux.1-fill-dev.png")
