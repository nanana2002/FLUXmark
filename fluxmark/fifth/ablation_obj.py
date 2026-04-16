#!/usr/bin/env python3
"""
消融实验：不加正交投影的水印生成（极致性能版 - 50GB显存全速运行）
对比正常版本，验证正交投影的重要性

优化策略：
- 保持所有模型常驻GPU
- 批量推理
- 不调用显存清理
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

output_dir = os.path.join(config['output_base_dir'], 'pic', 'ablation_obj_watermarked_img')
os.makedirs(output_dir, exist_ok=True)

# 加载prompts
prompts_path = os.path.join(config['output_base_dir'], 'prompts.json')
with open(prompts_path, 'r') as f:
    prompts_data = json.load(f)
prompts = prompts_data['prompts']

print(f"📋 加载了 {len(prompts)} 条 prompts")
print("🔬 消融实验：不加正交投影")

# ==========================================
# 1. 加载模型（直接到GPU，保持常驻）
# ==========================================
print("\n🚀 加载 FLUX 模型到GPU...")
pipe = FluxPipeline.from_pretrained(
    config['model_path'],
    torch_dtype=torch.bfloat16
)

# 性能优化
# pipe.enable_vae_tiling()
pipe.enable_attention_slicing(slice_size="auto")

# 直接加载到GPU并保持常驻
pipe.to("cuda")
print(f"   模型已加载到GPU，显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# 2. 预编码所有prompts（缓存于GPU）
# ==========================================
print("🔤 预编码prompts到GPU缓存...")
encoded_prompts = []
for prompt in tqdm(prompts, desc="编码进度"):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=256
        )
    # 保持在GPU上缓存
    encoded_prompts.append({
        'prompt_embeds': prompt_embeds,
        'pooled_prompt_embeds': pooled_prompt_embeds,
        'text_ids': text_ids
    })

print(f"   已编码 {len(encoded_prompts)} 个prompts到GPU")

# ==========================================
# 3. 生成FFT密码本W（正常流程）
# ==========================================
print("🔐 生成FFT密码本...")

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

# 语义掩码
M_spatial = torch.zeros((1, 32, 32, 1), device='cuda', dtype=torch.bfloat16)
y_s, y_e, x_s, x_e = config['mask_region']
M_spatial[0, y_s:y_e, x_s:x_e, 0] = 1.0
M_attn = M_spatial.view(1, 1024, 1)
W_anchored = W * M_attn

print("   ⚠️  警告：此版本未使用正交投影")
print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# 4. 非正交注入回调（关键差异）
# ==========================================
alpha = config['alpha']

def non_orthogonal_drift_callback(pipe, step_index, timestep, callback_kwargs):
    """
    非正交注入回调
    直接注入，不做正交化和能量重归一化
    """
    latents = callback_kwargs["latents"]

    # 不使用正交投影，直接注入
    sigmas = pipe.scheduler.sigmas
    dt = sigmas[step_index + 1] - sigmas[step_index]

    # 直接加法，不进行正交化和能量归一化
    latents = latents + alpha * W_anchored * abs(dt)

    callback_kwargs["latents"] = latents
    return callback_kwargs

# ==========================================
# 5. 批量生成消融实验图像（高性能模式）
# ==========================================
print(f"\n🎨 开始生成 {len(prompts)} 张消融实验图像（非正交）...\n")

results_summary = {
    'experiment_name': f"{config['experiment_name']}_ablation_obj",
    'timestamp': datetime.now().isoformat(),
    'ablation_type': 'no_orthogonal_projection',
    'total_images': len(prompts),
    'config': config,
    'images': []
}

batch_size = config.get('batch_size', 4)

# 批量生成
for batch_start in tqdm(range(0, len(prompts), batch_size), desc="生成批次"):
    batch_end = min(batch_start + batch_size, len(prompts))
    batch_indices = list(range(batch_start, batch_end))

    # 收集batch数据
    batch_prompt_embeds = torch.cat([encoded_prompts[i]['prompt_embeds'] for i in batch_indices])
    batch_pooled_embeds = torch.cat([encoded_prompts[i]['pooled_prompt_embeds'] for i in batch_indices])
    batch_text_ids = torch.cat([encoded_prompts[i]['text_ids'] for i in batch_indices])

    # 批量生成
    with torch.no_grad():
        images = pipe(
            prompt_embeds=batch_prompt_embeds,
            pooled_prompt_embeds=batch_pooled_embeds,
            num_inference_steps=config['num_inference_steps'],
            guidance_scale=config['guidance_scale'],
            height=config['height'],
            width=config['width'],
            callback_on_step_end=non_orthogonal_drift_callback,
            num_images_per_prompt=1,
        ).images

    # 批量提取签名
    for idx, img_idx in enumerate(batch_indices):
        img_id = f"ablation_obj_{img_idx:03d}"
        img_filename = f"{img_id}.png"
        img_path = os.path.join(output_dir, img_filename)
        images[idx].save(img_path)

        # 提取签名
        img_tensor = T.ToTensor()(images[idx]).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        img_tensor = (img_tensor - 0.5) * 2.0

        # 使用原始 encoded_prompts 避免维度问题
        encoded = encoded_prompts[img_idx]
        prompt_embeds = encoded['prompt_embeds']
        pooled_prompt_embeds = encoded['pooled_prompt_embeds']
        text_ids = encoded['text_ids']

        with torch.no_grad():
            z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
            z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            z_0 = pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)

            h = w = 32
            img_ids = torch.zeros(h, w, 3, device='cuda', dtype=torch.bfloat16)
            img_ids[..., 1] = torch.arange(h, device='cuda', dtype=torch.bfloat16).unsqueeze(1) / max(h - 1, 1)
            img_ids[..., 2] = torch.arange(w, device='cuda', dtype=torch.bfloat16).unsqueeze(0) / max(w - 1, 1)
            img_ids = img_ids.view(1, h * w, 3)

            t_detect = torch.tensor([pipe.scheduler.sigmas[-1]], device="cuda", dtype=torch.bfloat16)
            v_pred = pipe.transformer(
                hidden_states=z_0, timestep=t_detect / 1000,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

        # 计算签名
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

        sig_filename = f"{img_id}_signature.npy"
        sig_path = os.path.join(output_dir, sig_filename)
        np.save(sig_path, S)

        mask_8x8 = np.zeros((8, 8))
        mask_8x8[1:7, 1:7] = 1.0
        core_scores = S[mask_8x8 == 1.0]

        img_result = {
            'id': img_id,
            'prompt': prompts[img_idx],
            'image_file': img_filename,
            'signature_file': sig_filename,
            'stats': {
                'mean': float(np.mean(core_scores)),
                'std': float(np.std(core_scores)),
                'min': float(np.min(core_scores)),
                'max': float(np.max(core_scores))
            }
        }
        results_summary['images'].append(img_result)

# ==========================================
# 6. 保存摘要
# ==========================================
summary_path = os.path.join(output_dir, f'{config["experiment_name"]}_ablation_obj_summary.json')
with open(summary_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\n✅ 消融实验（非正交）完成！")
print(f"   图像保存位置: {output_dir}")
print(f"   实验摘要: {summary_path}")
print(f"\n📊 核心签名统计:")
means = [img['stats']['mean'] for img in results_summary['images']]
print(f"   平均值: {np.mean(means):.4f}")
print(f"   标准差: {np.std(means):.4f}")
print(f"   最小值: {np.min(means):.4f}")
print(f"   最大值: {np.max(means):.4f}")
print(f"\n   最终显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"   🚀 高性能模式：保持所有模型和缓存常驻GPU！")
