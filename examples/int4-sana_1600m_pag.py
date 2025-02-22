import torch
from diffusers import SanaPAGPipeline

from nunchaku.models.transformer_sana import NunchakuSanaTransformer2DModel

transformer = NunchakuSanaTransformer2DModel.from_pretrained("mit-han-lab/svdq-int4-sana-1600m", pag_layers=8)
pipe = SanaPAGPipeline.from_pretrained(
    "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    transformer=transformer,
    variant="bf16",
    torch_dtype=torch.bfloat16,
    pag_applied_layers="transformer_blocks.8",
).to("cuda")
pipe._set_pag_attn_processor = lambda *args, **kwargs: None

pipe.text_encoder.to(torch.bfloat16)
pipe.vae.to(torch.bfloat16)

image = pipe(
    prompt="A cute 🐼 eating 🎋, ink drawing style",
    height=1024,
    width=1024,
    guidance_scale=5.0,
    pag_scale=2.0,
    num_inference_steps=20,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save("sana_1600m_pag.png")
