import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import gc
import json

with open('config.json', 'r') as f:
    config = json.load(f)

os.environ['HF_HOME'] = config.get('hf_cache', '/home/daiyn/hf_cache')
model_path = config.get('model_path', '/home/daiyn/project_flux/model/flux-schnell')
gpu_id = config.get('gpu_id', 0)
device = f'cuda:{gpu_id}'

print("🚀 [Baseline 1] 加载 FLUX 模型进行 Tree-Ring (ICCV'23) 测试...")
pipe = FluxPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
).to(device)

prompt = "a cat holding a sign that says hello world"
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=256
    )

pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()

# 工具函数：FLUX 特有的 Packing / Unpacking
def pack_latents(latents):
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)

def unpack_latents(latents, h=64, w=64):
    b, _, c4 = latents.shape
    c = c4 // 4
    latents = latents.view(b, h // 2, w // 2, c, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, c, h, w)

# ==========================================
# 1. Tree-Ring 注入：在初始噪声的频域画圈
# ==========================================
print("⭕ 构造 Tree-Ring 频域环形水印...")
generator = torch.Generator(device=device).manual_seed(42)
# FLUX 的基础原生潜空间是 16 通道 64x64
z_1_raw = torch.randn((1, 16, 64, 64), generator=generator, device=device, dtype=torch.float32)

# 转换到频域
F_z1 = torch.fft.fftshift(torch.fft.fft2(z_1_raw, dim=(2, 3)), dim=(2, 3))

# 定义 Tree-Ring 掩码 (半径 10 到 20 之间)
Y, X = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
radius = torch.sqrt((Y - 32)**2 + (X - 32)**2).to(device)
ring_mask = ((radius >= 10) & (radius <= 20)).float().unsqueeze(0).unsqueeze(0)

# 生成一个特定的复数环形密钥
key_ring = torch.randn((1, 16, 64, 64), device=device, dtype=torch.float32) + \
           1j * torch.randn((1, 16, 64, 64), device=device, dtype=torch.float32)
key_ring = key_ring * ring_mask

# 强行植入频域
F_z1_watermarked = F_z1 * (1 - ring_mask) + key_ring
z_1_wm = torch.fft.ifft2(torch.fft.ifftshift(F_z1_watermarked, dim=(2, 3)), dim=(2, 3)).real
z_1_wm = z_1_wm.to(torch.bfloat16)

# 生成图像 (强行接管初始噪声)
print("🎨 用 Tree-Ring 噪声生成图像...")
z_1_wm_packed = pack_latents(z_1_wm)
image_out = pipe(
    prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=4, guidance_scale=0.0, height=512, width=512,
    latents=z_1_wm_packed
).images[0]

# 保存生成的带水印图像
os.makedirs("pic/baseline_treering_img", exist_ok=True)
output_path = "pic/baseline_treering_img/baseline_treering_flux.png"
image_out.save(output_path)
print(f"💾 生成图像已保存至: {output_path}")

# ==========================================
# 2. Tree-Ring 提取：多步 ODE 逆向推导
# ==========================================
print("⏪ 执行 Tree-Ring 的 4步 ODE 逆推 (Inversion)...")
img_tensor = T.ToTensor()(image_out).unsqueeze(0).to(device, dtype=torch.bfloat16)
img_tensor = (img_tensor - 0.5) * 2.0 

with torch.no_grad():
    z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
    z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    
    z_t_packed = pack_latents(z_0)
    
    img_ids = torch.zeros(32, 32, 3, device=device, dtype=torch.bfloat16)
    img_ids[..., 1] = torch.arange(32, device=device, dtype=torch.bfloat16).unsqueeze(1) / 31.0
    img_ids[..., 2] = torch.arange(32, device=device, dtype=torch.bfloat16).unsqueeze(0) / 31.0
    img_ids = img_ids.view(1, 1024, 3)

    # 模拟 4 步反转欧拉积分 (从 t=0 倒推回 t=1)
    sigmas = pipe.scheduler.sigmas  # FLUX sigmas are typically[1.0, 0.75, 0.5, 0.25, 0.0]
    for i in range(len(sigmas) - 1, 0, -1):
        t_curr = sigmas[i]   # 0.0, 0.25...
        t_next = sigmas[i-1] # 0.25, 0.5...
        
        v_pred = pipe.transformer(
            hidden_states=z_t_packed, timestep=torch.tensor([t_curr], device=device, dtype=torch.bfloat16) / 1000.0,
            pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids, img_ids=img_ids, return_dict=False,
        )[0]
        
        # 逆向欧拉: x_{t_next} = x_{t_curr} + v * dt
        dt = t_next - t_curr 
        z_t_packed = z_t_packed + v_pred * dt

    z_1_hat = unpack_latents(z_t_packed).to(torch.float32)

# ==========================================
# 3. 统计学宣判
# ==========================================
F_z_1_hat = torch.fft.fftshift(torch.fft.fft2(z_1_hat, dim=(2, 3)), dim=(2, 3))

ring_mask_bool = (ring_mask == 1.0).expand_as(key_ring)
orig_key_flat = key_ring[ring_mask_bool].flatten()
extracted_flat = F_z_1_hat[ring_mask_bool].flatten()

# 复数拆分为实部虚部计算余弦相似度
real_sim = torch.nn.functional.cosine_similarity(orig_key_flat.real.unsqueeze(0), extracted_flat.real.unsqueeze(0)).item()
imag_sim = torch.nn.functional.cosine_similarity(orig_key_flat.imag.unsqueeze(0), extracted_flat.imag.unsqueeze(0)).item()
final_score = (real_sim + imag_sim) / 2

print("\n" + "="*60)
print(f"Tree-Ring 在 FLUX 上的提取分数: {final_score:.4f}")
print("="*60 + "\n")