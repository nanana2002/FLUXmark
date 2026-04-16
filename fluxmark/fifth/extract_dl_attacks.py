#!/usr/bin/env python3
"""
仅补提深度学习攻击（SDXL / SDEdit / FluxFill）的 S_tamp 签名
不重复提取 S_orig 和 S_no_wm
"""


import os
import json


# 先读取配置设置 GPU（必须在 import torch 之前）
with open('config.json', 'r') as f:
    config = json.load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_id', 0))
os.environ['HF_HOME'] = config['hf_cache']


import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm


# 路径设置
base_output_dir = os.path.join(config['output_base_dir'], 'pic')
watermark_extra_dir = os.path.join(config['output_base_dir'], 'watermark_extra')
os.makedirs(watermark_extra_dir, exist_ok=True)


watermarked_dir = os.path.join(base_output_dir, 'watermarked_img')
attack_dir = os.path.join(base_output_dir, 'attack_watermarked_img')
no_watermark_dir = os.path.join(base_output_dir, 'no_watermarked_img')


# 加载摘要
print("📋 加载实验数据...")
with open(os.path.join(watermarked_dir, f'{config["experiment_name"]}_watermark_summary.json'), 'r') as f:
    watermark_summary = json.load(f)


images_info = watermark_summary['images']
print(f"   水印图像: {len(images_info)} 张")


# ==========================================
# 1. 加载模型（直接到GPU，保持常驻）
# ==========================================
print("\n🚀 加载 FLUX 模型到GPU...")
pipe = FluxPipeline.from_pretrained(
    config['model_path'],
    torch_dtype=torch.bfloat16
)


# pipe.enable_vae_tiling()
pipe.enable_attention_slicing(slice_size="auto")
pipe.to("cuda")
print(f"   模型已加载到GPU，显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


# ==========================================
# 2. 预编码所有prompts（缓存于GPU）
# ==========================================
print("🔤 预编码所有prompts到GPU缓存...")
encoded_prompts_cache = {}
unique_prompts = set()


for img_info in images_info:
    unique_prompts.add(img_info['prompt'])


no_wm_summary_path = os.path.join(no_watermark_dir, f'{config["experiment_name"]}_no_watermark_summary.json')
if os.path.exists(no_wm_summary_path):
    with open(no_wm_summary_path, 'r') as f:
        no_wm_summary = json.load(f)
    for img_info in no_wm_summary['images']:
        unique_prompts.add(img_info['prompt'])


for prompt_text in tqdm(unique_prompts, desc="编码进度"):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt_text, prompt_2=None, max_sequence_length=256
        )
    encoded_prompts_cache[prompt_text] = {
        'prompt_embeds': prompt_embeds,
        'pooled_prompt_embeds': pooled_prompt_embeds,
        'text_ids': text_ids
    }


print(f"   已编码 {len(encoded_prompts_cache)} 个唯一prompts到GPU")
print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


# ==========================================
# 3. 重建FFT密码本（常驻GPU）
# ==========================================
print("🔐 重建FFT密码本...")
secret_key = config['secret_key']
generator = torch.Generator(device='cuda').manual_seed(secret_key)
W_raw = torch.randn((1, 1024, 64), generator=generator, device='cuda', dtype=torch.float32)
W_spatial = W_raw.view(1, 32, 32, 64)


F_W = torch.fft.fftshift(torch.fft.fft2(W_spatial, dim=(1, 2)), dim=(1, 2))
h, w = 32, 32
Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
center_y, center_x = h // 2, w // 2
radius = torch.sqrt((Y - center_y)**2 + (X - center_x)**2).to('cuda')


r_inner = config['fft_radius_inner']
r_outer = config['fft_radius_outer']
mask_fft = ((radius >= r_inner) & (radius <= r_outer)).float().unsqueeze(0).unsqueeze(-1)


F_W_filtered = F_W * mask_fft
W_filtered = torch.fft.ifft2(torch.fft.ifftshift(F_W_filtered, dim=(1, 2)), dim=(1, 2)).real
W = W_filtered.view(1, 1024, 64).to(torch.bfloat16)
W = W / (torch.norm(W, dim=-1, keepdim=True) + 1e-8)


M_spatial = torch.zeros((1, 32, 32, 1), device='cuda', dtype=torch.bfloat16)
y_s, y_e, x_s, x_e = config['mask_region']
M_spatial[0, y_s:y_e, x_s:x_e, 0] = 1.0
M = M_spatial.view(1, 1024, 1)
W_anchored = W * M


print(f"   密码本已重建并常驻GPU")


# ==========================================
# 4. 提取签名函数（批量版本）
# ==========================================
def extract_signature_batch(pil_images, prompt_embeds_list, pooled_embeds_list, text_ids_list):
    """批量从图像中提取签名（高性能GPU版本）"""
    batch_size = len(pil_images)


    img_tensors = []
    for img in pil_images:
        img = img.resize((512, 512), Image.LANCZOS)
        img_tensor = T.ToTensor()(img).unsqueeze(0)
        img_tensors.append(img_tensor)
    img_batch = torch.cat(img_tensors).to("cuda", dtype=torch.bfloat16)
    img_batch = (img_batch - 0.5) * 2.0


    with torch.no_grad():
        z_0_list = []
        for img_tensor in img_batch:
            z_0 = pipe.vae.encode(img_tensor.unsqueeze(0)).latent_dist.sample()
            z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)
            z_0_list.append(z_0)


        h = w = 32
        img_ids = torch.zeros(32, 32, 3, device='cuda', dtype=torch.bfloat16)
        img_ids[..., 1] = torch.arange(32, device='cuda', dtype=torch.bfloat16).unsqueeze(1) / 31.0
        img_ids[..., 2] = torch.arange(32, device='cuda', dtype=torch.bfloat16).unsqueeze(0) / 31.0
        img_ids = img_ids.view(1, 1024, 3)


        v_pred_list = []
        for i in range(batch_size):
            v_pred = pipe.transformer(
                hidden_states=z_0_list[i],
                timestep=torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16) / 1000,
                pooled_projections=pooled_embeds_list[i],
                encoder_hidden_states=prompt_embeds_list[i],
                txt_ids=text_ids_list[i],
                img_ids=img_ids,
                return_dict=False,
            )[0]
            v_pred_list.append(v_pred)


    signatures = []
    for v_pred in v_pred_list:
        v_pred_spatial = v_pred.view(32, 32, 64)
        W_anchored_spatial = W_anchored.view(32, 32, 64)
        S = np.zeros((8, 8))


        for i in range(8):
            for j in range(8):
                W_patch = W_anchored_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
                v_patch = v_pred_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
                S[i, j] = torch.nn.functional.cosine_similarity(
                    W_patch.unsqueeze(0), v_patch.unsqueeze(0)
                ).item()
        signatures.append(S)


    return signatures


# ==========================================
# 5. 自动扫描并补提深度学习攻击的 S_tamp
# ==========================================
# 深度学习攻击的关键字（目录名包含这些即视为深度学习攻击）
DL_ATTACK_KEYWORDS = ('sdxl_', 'sdedit_', 'fluxfill_')


dl_attack_dirs = []
for d in sorted(os.listdir(attack_dir)):
    if any(kw in d for kw in DL_ATTACK_KEYWORDS):
        full_path = os.path.join(attack_dir, d)
        if os.path.isdir(full_path):
            dl_attack_dirs.append(d)


print(f"\n🎯 发现 {len(dl_attack_dirs)} 个深度学习攻击目录: {dl_attack_dirs}")


batch_size = config.get('batch_size', 4)


for attack_name in tqdm(dl_attack_dirs, desc="深度学习攻击签名提取"):
    attack_subdir = os.path.join(attack_dir, attack_name)


    for batch_start in range(0, len(images_info), batch_size):
        batch_end = min(batch_start + batch_size, len(images_info))
        batch_infos = images_info[batch_start:batch_end]


        pil_images = []
        prompt_embeds_list = []
        pooled_embeds_list = []
        text_ids_list = []
        img_ids_batch = []


        for img_info in batch_infos:
            img_id = img_info['id']
            prompt = img_info['prompt']


            attacked_img_path = os.path.join(attack_subdir, f"{img_id}.png")


            # 如果 png 不存在，试试 jpg（jpeg 攻击可能保存为 jpg）
            if not os.path.exists(attacked_img_path):
                attacked_img_path = os.path.join(attack_subdir, f"{img_id}.jpg")


            if not os.path.exists(attacked_img_path):
                continue


            S_tamp_path = os.path.join(watermark_extra_dir, f"{img_id}_{attack_name}_S_tamp.npy")
            if os.path.exists(S_tamp_path):
                continue


            try:
                img = Image.open(attacked_img_path).convert('RGB')
                pil_images.append(img)


                encoded = encoded_prompts_cache[prompt]
                prompt_embeds_list.append(encoded['prompt_embeds'])
                pooled_embeds_list.append(encoded['pooled_prompt_embeds'])
                text_ids_list.append(encoded['text_ids'])
                img_ids_batch.append(img_id)
            except Exception as e:
                print(f"\n   ❌ {img_id} {attack_name} 加载失败: {e}")
                continue


        if not pil_images:
            continue


        try:
            signatures = extract_signature_batch(pil_images, prompt_embeds_list, pooled_embeds_list, text_ids_list)


            for idx, img_id in enumerate(img_ids_batch):
                S_tamp_path = os.path.join(watermark_extra_dir, f"{img_id}_{attack_name}_S_tamp.npy")
                np.save(S_tamp_path, signatures[idx])
        except Exception as e:
            print(f"\n   ❌ {attack_name} 批量提取失败: {e}")
            continue


# ==========================================
# 6. 更新提取摘要
# ==========================================
summary_path = os.path.join(watermark_extra_dir, 'extraction_summary.json')
existing_summary = {}
if os.path.exists(summary_path):
    with open(summary_path, 'r') as f:
        existing_summary = json.load(f)


existing_summary['timestamp'] = datetime.now().isoformat()
existing_summary['dl_attacks_extracted'] = dl_attack_dirs


with open(summary_path, 'w') as f:
    json.dump(existing_summary, f, indent=2)


print(f"\n" + "="*60)
print("✅ 深度学习攻击签名补提完成！")
print("="*60)
print(f"   签名保存位置: {watermark_extra_dir}")
print(f"   当前 S_tamp 总数: {len([f for f in os.listdir(watermark_extra_dir) if f.endswith('_S_tamp.npy')])} 个")
print(f"   最终显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")




