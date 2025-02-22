import torch
from diffusers import FluxPipeline, FluxPriorReduxPipeline
from diffusers.utils import load_image

from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Redux-dev", torch_dtype=torch.bfloat16
).to("cuda")
transformer = NunchakuFluxTransformer2dModel.from_pretrained("mit-han-lab/svdq-int4-flux.1-dev")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=None,
    text_encoder_2=None,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")
pipe_prior_output = pipe_prior_redux(image)
images = pipe(guidance_scale=2.5, num_inference_steps=50, **pipe_prior_output).images
images[0].save("flux.1-redux-dev.png")
