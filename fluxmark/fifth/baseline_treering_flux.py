import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image
import numpy as np
import gc

print("🚀 加载 FLUX 模型进行 Tree-Ring Baseline 测试...")
pipe = FluxPipeline.from_pretrained(
    '/home/daiyn/project_flux/model/flux-schnell',
    torch_dtype=torch.bfloat16
).to("cuda")

prompt = "a cat holding a sign that says hello world"

# 先把 prompt 编码完！！！（修复核心1）
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=256
    )

# ===================== 修复核心2：先生成完图像，再删模型 =====================
# ==========================================
# 1. 构造 Tree-Ring 初始噪声
# ==========================================
generator = torch.Generator(device='cuda').manual_seed(42)
noise_spatial = torch.randn((1, 32, 32, 64), generator=generator, device='cuda', dtype=torch.float32)

F_noise = torch.fft.fftshift(torch.fft.fft2(noise_spatial, dim=(1, 2)), dim=(1, 2))

Y, X = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
radius = torch.sqrt((Y - 16)**2 + (X - 16)**2).to('cuda')

ring_mask = torch.zeros((1, 32, 32), device='cuda', dtype=torch.float32)
ring_mask[0][(radius >= 4) & (radius <= 10)] = 1.0
ring_mask = ring_mask.unsqueeze(-1).expand(1, 32, 32, 64).contiguous()

key_ring = torch.randn((1, 32, 32, 64), device='cuda', dtype=torch.float32) \
         + 1j * torch.randn((1, 32, 32, 64), device='cuda', dtype=torch.float32)
key_ring = key_ring * ring_mask

F_noise_watermarked = F_noise * (1 - ring_mask) + key_ring * ring_mask

z_1_spatial = torch.fft.ifft2(torch.fft.ifftshift(F_noise_watermarked, dim=(1, 2)), dim=(1, 2)).real
z_1 = z_1_spatial.view(1, 1024, 64).to(torch.bfloat16)

# ==========================================
# 2. 生成图像（现在还没删模型，正常运行）
# ==========================================
print("🎨 用 Tree-Ring 噪声生成图像...")
with torch.no_grad():
    image_out = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        num_inference_steps=4,
        guidance_scale=0.0,
        height=512, width=512,
        latents=z_1
    ).images[0]

image_out.save("treering_flux_cat.png")

# ===================== 现在才卸载文本编码器 =====================
del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 3. ODE 逆推提取
# ==========================================
print("🔍 对 Tree-Ring 进行 4步 ODE 逆向提取...")
img_tensor = T.ToTensor()(image_out).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
img_tensor = (img_tensor - 0.5) * 2.0 

with torch.no_grad():
    z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
    z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)

    h = w = 32
    img_ids = torch.zeros(h, w, 3, device='cuda', dtype=torch.bfloat16)
    img_ids[..., 1] = torch.arange(h, device='cuda', dtype=torch.bfloat16).unsqueeze(1) / max(h - 1, 1)
    img_ids[..., 2] = torch.arange(w, device='cuda', dtype=torch.bfloat16).unsqueeze(0) / max(w - 1, 1)
    img_ids = img_ids.view(1, h * w, 3)

    latents = z_0
    sigmas = pipe.scheduler.sigmas
    
    for i in range(len(sigmas) - 1, 0, -1):
        t_curr = sigmas[i]
        t_next = sigmas[i-1]
        
        v_pred = pipe.transformer(
            hidden_states=latents,
            timestep=torch.tensor([t_curr], device="cuda", dtype=torch.bfloat16) / 1000,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
        
        dt = t_next - t_curr
        latents = latents + v_pred * dt

    z_1_hat = latents

# ==========================================
# 4. 计算相关性
# ==========================================
z_1_hat_spatial = z_1_hat.view(1, 32, 32, 64).to(torch.float32)
F_z_1_hat = torch.fft.fftshift(torch.fft.fft2(z_1_hat_spatial, dim=(1, 2)), dim=(1, 2))

orig_key_flat = key_ring[ring_mask == 1.0].flatten()
extracted_flat = F_z_1_hat[ring_mask == 1.0].flatten()

real_sim = torch.nn.functional.cosine_similarity(orig_key_flat.real.unsqueeze(0), extracted_flat.real.unsqueeze(0)).item()
imag_sim = torch.nn.functional.cosine_similarity(orig_key_flat.imag.unsqueeze(0), extracted_flat.imag.unsqueeze(0)).item()

final_score = (real_sim + imag_sim) / 2
print(f"=======================================")
print(f"🎯 Tree-Ring (Baseline) 在 FLUX 上的提取分数: {final_score:.4f}")
print(f"=======================================")