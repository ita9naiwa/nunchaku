import torch
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor

from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-depth-dev")

pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Depth-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")

prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(control_image)[0].convert("RGB")

image = pipe(
    prompt=prompt, control_image=control_image, height=1024, width=1024, num_inference_steps=30, guidance_scale=10.0
).images[0]
image.save("flux.1-depth-dev.png")
