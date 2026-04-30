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

print("🚀 [Baseline 2] 加载 FLUX 模型进行 Gaussian Shading (CVPR'24) 测试...")
pipe = FluxPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
).to(device)

prompt = "Elderly  gray haired man in a suit scowling into the camera."
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=256
    )
    # 预编码空 Prompt（用于盲提取实验）
    empty_prompt_embeds, empty_pooled_prompt_embeds, empty_text_ids = pipe.encode_prompt(
        prompt="", prompt_2=None, max_sequence_length=256
    )

pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()

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

def extract_gs_score(image, pipe, device, prompt_embeds, pooled_prompt_embeds, text_ids, key_W):
    """对给定图像执行 Gaussian Shading 提取并返回余弦相似度分数"""
    img_tensor = T.ToTensor()(image).unsqueeze(0).to(device, dtype=torch.bfloat16)
    img_tensor = (img_tensor - 0.5) * 2.0
    with torch.no_grad():
        z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
        z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z_t_packed = pack_latents(z_0)
        img_ids = torch.zeros(32, 32, 3, device=device, dtype=torch.bfloat16)
        img_ids[..., 1] = torch.arange(32, device=device, dtype=torch.bfloat16).unsqueeze(1) / 31.0
        img_ids[..., 2] = torch.arange(32, device=device, dtype=torch.bfloat16).unsqueeze(0) / 31.0
        img_ids = img_ids.view(1024, 3)  # 新版 diffusers 要求 2D
        # txt_ids 同理，确保 2D
        txt_ids_2d = text_ids if text_ids.dim() == 2 else text_ids.squeeze(0)
        sigmas = pipe.scheduler.sigmas
        for i in range(len(sigmas) - 1, 0, -1):
            t_curr = sigmas[i]
            t_next = sigmas[i-1]
            v_pred = pipe.transformer(
                hidden_states=z_t_packed, timestep=torch.tensor([t_curr], device=device, dtype=torch.bfloat16) / 1000.0,
                pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids_2d, img_ids=img_ids, return_dict=False,
            )[0]
            dt = t_next - t_curr
            z_t_packed = z_t_packed + v_pred * dt
        z_1_hat = unpack_latents(z_t_packed).to(torch.float32)
    orig_key_flat = key_W.flatten()
    extracted_flat = z_1_hat.flatten()
    score = torch.nn.functional.cosine_similarity(orig_key_flat.unsqueeze(0), extracted_flat.unsqueeze(0)).item()
    return score

# ==========================================
# 1. Gaussian Shading 注入：空间混合高斯噪声密钥
# ==========================================
print("🌫️ 构造 Gaussian Shading 隐空间混合水印...")
generator = torch.Generator(device=device).manual_seed(42)
z_1_raw = torch.randn((1, 16, 64, 64), generator=generator, device=device, dtype=torch.float32)

# 生成高斯水印密钥 W
key_W = torch.randn((1, 16, 64, 64), device=device, dtype=torch.float32)

# 执行高斯着色混合 (典型的基于初值的隐空间混入)
alpha_gs = 0.5 
z_1_wm = (z_1_raw + alpha_gs * key_W) / np.sqrt(1 + alpha_gs**2)
z_1_wm = z_1_wm.to(torch.bfloat16)

print("🎨 用 Gaussian Shading 噪声生成图像...")
z_1_wm_packed = pack_latents(z_1_wm)
image_out = pipe(
    prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=4, guidance_scale=0.0, height=512, width=512,
    latents=z_1_wm_packed
).images[0]

# 保存生成的原始带水印图像
os.makedirs("pic/baseline_gaussionshading_img", exist_ok=True)
orig_path = "pic/baseline_gaussionshading_img/baseline_gaussionshading_flux.png"
image_out.save(orig_path)
print(f"💾 生成图像已保存至: {orig_path}")

# 保存 key_W（供后续攻击与提取使用）
key_path = "pic/baseline_gaussionshading_img/baseline_gaussionshading_key_W.npy"
np.save(key_path, key_W.cpu().numpy())
print(f"💾 key_W 已保存至: {key_path}")

# 用 PIL 压缩 JPEG Quality=50，再送进去逆推
print("🗜️ 对生成图像施加 JPEG 压缩 (Quality=50)...")
image_out = image_out.convert("RGB")
import io
jpeg_buffer = io.BytesIO()
image_out.save(jpeg_buffer, format="JPEG", quality=50)
image_out = Image.open(jpeg_buffer)

# 保存 JPEG 压缩后的图像（供对比）
jpeg_path = "pic/baseline_gaussionshading_img/baseline_gaussionshading_flux_jpeg50.jpg"
image_out.save(jpeg_path)
print(f"💾 JPEG 压缩图像已保存至: {jpeg_path}")

# ==========================================
# 2. JPEG-50 攻击下的 Gaussian Shading 提取
# ==========================================
print("⏪ 对 JPEG-50 图像执行 Gaussian Shading 提取...")
final_score = extract_gs_score(image_out, pipe, device, prompt_embeds, pooled_prompt_embeds, text_ids, key_W)

print("\n" + "="*60)
print(f"Gaussian Shading 在 JPEG-50 攻击后的提取分数: {final_score:.4f}")
print("="*60 + "\n")

# ==========================================
# 3. SDEdit 0.4 攻击 + 提取
# ==========================================
print("🎨 执行 SDEdit 0.4 攻击...")
from diffusers import FluxImg2ImgPipeline

sdedit_pipe = FluxImg2ImgPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
).to(device)

orig_img = Image.open(orig_path)
attacked_img = sdedit_pipe(
    prompt_embeds=prompt_embeds,                  # 必须用 embeds，因为 encoder 已被卸载
    pooled_prompt_embeds=pooled_prompt_embeds,
    image=orig_img,
    strength=0.4,
    num_inference_steps=4,
    guidance_scale=0.0,
    height=512,
    width=512,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

sdedit_path = "pic/baseline_gaussionshading_img/baseline_gaussionshading_flux_sdedit04.png"
attacked_img.save(sdedit_path)
print(f"💾 SDEdit 0.4 攻击图像已保存至: {sdedit_path}")

# 销毁被污染的 pipeline，重新加载干净的 transformer
print("🔄 重新加载干净的 Transformer 用于提取...")
del sdedit_pipe
gc.collect()
torch.cuda.empty_cache()

from diffusers import FluxTransformer2DModel
pipe.transformer = FluxTransformer2DModel.from_pretrained(
    model_path, subfolder="transformer", torch_dtype=torch.bfloat16
).to(device)

print("⏪ 对 SDEdit 0.4 图像执行 Gaussian Shading 提取...")
sdedit_score = extract_gs_score(attacked_img, pipe, device, prompt_embeds, pooled_prompt_embeds, text_ids, key_W)

print("\n" + "="*60)
print(f"Gaussian Shading 在 SDEdit 0.4 攻击后的提取分数: {sdedit_score:.4f}")
print("="*60 + "\n")

# ==========================================
# 4. 致命实验：空 Prompt 盲提取 (Blind Extraction)
# ==========================================
print("空 Prompt 盲提取测试...")
# 直接使用开头预编码好的空 Prompt embeddings
# 对原始生成图像（未 JPEG/SDEdit）做空 Prompt 盲提取
orig_img_for_blind = Image.open(orig_path)
blind_score = extract_gs_score(orig_img_for_blind, pipe, device, empty_prompt_embeds, empty_pooled_prompt_embeds, empty_text_ids, key_W)

print("\n" + "="*60)
print(f"Gaussian Shading 空 Prompt 盲提取分数: {blind_score:.4f}")
print("="*60 + "\n")