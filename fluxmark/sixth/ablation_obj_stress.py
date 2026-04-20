#!/usr/bin/env python3
"""
消融实验：极端 alpha 压力测试（非正交 vs 正交对比）
在多个极端 alpha 值下运行，验证正交投影机制的重要性

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

# ==========================================
# 0. 配置压力测试 alpha 列表
# ==========================================
stress_alphas = [1.0, 2.0, 3.0, 4.0]

base_output_dir = os.path.join(config['output_base_dir'], 'pic')

# 加载prompts
prompts_path = os.path.join(config['output_base_dir'], 'prompts.json')
with open(prompts_path, 'r') as f:
    prompts_data = json.load(f)
prompts = prompts_data['prompts']

# 根据 config 中的 num_samples 截断（方便小批量测试）
num_samples = config.get('num_samples', len(prompts))
prompts = prompts[:num_samples]
print(f"📋 加载了 {len(prompts)} 条 prompts（config.num_samples={num_samples}）")
print(f"🔬 极端 alpha 压力测试：alpha 列表 = {stress_alphas}")

# ==========================================
# 1. 加载模型（直接到GPU，保持常驻）
# ==========================================
print("\n🚀 加载 FLUX 模型到GPU...")
pipe = FluxPipeline.from_pretrained(
    config['model_path'],
    torch_dtype=torch.bfloat16
)

# 性能优化
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

print(f"   FFT带通: [{r_inner}, {r_outer}]")
print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# 4. 定义回调工厂函数（支持可变 alpha）
# ==========================================

def make_non_orthogonal_callback(alpha_val):
    """
    非正交注入回调（直接注入，无投影）
    """
    def non_orthogonal_drift_callback(pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        sigmas = pipe.scheduler.sigmas
        dt = sigmas[step_index + 1] - sigmas[step_index]
        # 直接加法，不进行正交化和能量归一化
        latents = latents + alpha_val * W * abs(dt)
        callback_kwargs["latents"] = latents
        return callback_kwargs
    return non_orthogonal_drift_callback


def make_orthogonal_callback(alpha_val):
    """
    正交注入回调（Token-wise 正交投影 + 能量重归一化）
    使用全图 W（与消融实验保持一致，仅对比投影机制本身）
    """
    def orthogonal_drift_callback(pipe, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]

        # Token-wise 正交化
        dot_W_x = torch.sum(W * latents, dim=-1, keepdim=True)
        dot_x_x = torch.sum(latents * latents, dim=-1, keepdim=True) + 1e-8
        W_perp = W - (dot_W_x / dot_x_x) * latents

        # 能量重归一化
        norm_W = torch.norm(W, dim=-1, keepdim=True)
        norm_W_perp = torch.norm(W_perp, dim=-1, keepdim=True) + 1e-8
        W_perp = W_perp * (norm_W / norm_W_perp)

        # 欧拉积分加权
        sigmas = pipe.scheduler.sigmas
        dt = sigmas[step_index + 1] - sigmas[step_index]
        latents = latents + alpha_val * W_perp * abs(dt)

        callback_kwargs["latents"] = latents
        return callback_kwargs
    return orthogonal_drift_callback


# ==========================================
# 5. 批量生成图像的辅助函数
# ==========================================
def generate_batch(pipe, batch_indices, callback):
    """执行一批图像生成"""
    batch_prompt_embeds = torch.cat([encoded_prompts[i]['prompt_embeds'] for i in batch_indices])
    batch_pooled_embeds = torch.cat([encoded_prompts[i]['pooled_prompt_embeds'] for i in batch_indices])
    batch_text_ids = torch.cat([encoded_prompts[i]['text_ids'] for i in batch_indices])

    with torch.no_grad():
        images = pipe(
            prompt_embeds=batch_prompt_embeds,
            pooled_prompt_embeds=batch_pooled_embeds,
            num_inference_steps=config['num_inference_steps'],
            guidance_scale=config['guidance_scale'],
            height=config['height'],
            width=config['width'],
            callback_on_step_end=callback,
            num_images_per_prompt=1,
        ).images
    return images


def extract_signature_and_save(images, batch_indices, output_dir, img_prefix):
    """提取签名并保存图像，返回结果列表"""
    results = []
    for idx, img_idx in enumerate(batch_indices):
        img_id = f"{img_prefix}_{img_idx:03d}"
        img_filename = f"{img_id}.png"
        img_path = os.path.join(output_dir, img_filename)
        images[idx].save(img_path)

        # 提取签名
        img_tensor = T.ToTensor()(images[idx]).unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        img_tensor = (img_tensor - 0.5) * 2.0

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

        # 计算 32x32 签名
        v_pred_spatial = v_pred.view(32, 32, 64)
        W_spatial = W.view(32, 32, 64)
        S = np.zeros((32, 32))

        for i in range(32):
            for j in range(32):
                W_patch = W_spatial[i, j, :].flatten().float()
                v_patch = v_pred_spatial[i, j, :].flatten().float()
                S[i, j] = torch.nn.functional.cosine_similarity(
                    W_patch.unsqueeze(0), v_patch.unsqueeze(0)
                ).item()

        sig_filename = f"{img_id}_signature.npy"
        sig_path = os.path.join(output_dir, sig_filename)
        np.save(sig_path, S)

        img_result = {
            'id': img_id,
            'prompt': prompts[img_idx],
            'image_file': img_filename,
            'signature_file': sig_filename,
            'stats': {
                'mean': float(np.mean(S)),
                'std': float(np.std(S)),
                'min': float(np.min(S)),
                'max': float(np.max(S))
            }
        }
        results.append(img_result)
    return results


# ==========================================
# 6. 主循环：对每个 alpha 运行两种方法
# ==========================================
master_summary = {
    'experiment_name': f"{config['experiment_name']}_ablation_obj_stress_multi_alpha",
    'timestamp': datetime.now().isoformat(),
    'stress_alphas': stress_alphas,
    'total_images_per_alpha': len(prompts),
    'config': config,
    'alpha_results': {}
}

batch_size = config.get('batch_size', 4)

for alpha in stress_alphas:
    print(f"\n{'='*60}")
    print(f"🧪 开始压力测试 alpha = {alpha}")
    print(f"{'='*60}")

    alpha_str = f"alpha_{alpha}"
    master_summary['alpha_results'][alpha_str] = {}

    # ---------- 6a. 非正交版本 ----------
    no_ortho_dir = os.path.join(base_output_dir, 'ablation_obj_stress_watermarked_img', alpha_str)
    os.makedirs(no_ortho_dir, exist_ok=True)
    print(f"\n🔴 [alpha={alpha}] 非正交生成 → {no_ortho_dir}")

    no_ortho_callback = make_non_orthogonal_callback(alpha)
    no_ortho_images = []

    for batch_start in tqdm(range(0, len(prompts), batch_size),
                            desc=f"非正交 alpha={alpha}", leave=False):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_indices = list(range(batch_start, batch_end))
        images = generate_batch(pipe, batch_indices, no_ortho_callback)
        no_ortho_images.extend(extract_signature_and_save(
            images, batch_indices, no_ortho_dir, "ablation_obj_stress"
        ))

    no_ortho_means = [img['stats']['mean'] for img in no_ortho_images]
    no_ortho_stds = [img['stats']['std'] for img in no_ortho_images]
    no_ortho_overall = {
        'mean': float(np.mean(no_ortho_means)),
        'std': float(np.std(no_ortho_means)),
        'min': float(np.min(no_ortho_means)),
        'max': float(np.max(no_ortho_means))
    }

    master_summary['alpha_results'][alpha_str]['non_orthogonal'] = {
        'output_dir': no_ortho_dir,
        'ablation_type': 'no_orthogonal_projection_stress',
        'alpha': alpha,
        'images': no_ortho_images,
        'overall_stats': no_ortho_overall
    }

    print(f"   ✅ 非正交完成: mean={no_ortho_overall['mean']:.4f}, std={no_ortho_overall['std']:.4f}")

    # ---------- 6b. 正交版本（Full Method） ----------
    ortho_dir = os.path.join(base_output_dir, 'ablation_ortho_stress_watermarked_img', alpha_str)
    os.makedirs(ortho_dir, exist_ok=True)
    print(f"\n🟢 [alpha={alpha}] 正交生成（Full Method） → {ortho_dir}")

    ortho_callback = make_orthogonal_callback(alpha)
    ortho_images = []

    for batch_start in tqdm(range(0, len(prompts), batch_size),
                            desc=f"正交 alpha={alpha}", leave=False):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_indices = list(range(batch_start, batch_end))
        images = generate_batch(pipe, batch_indices, ortho_callback)
        ortho_images.extend(extract_signature_and_save(
            images, batch_indices, ortho_dir, "ablation_ortho_stress"
        ))

    ortho_means = [img['stats']['mean'] for img in ortho_images]
    ortho_stds = [img['stats']['std'] for img in ortho_images]
    ortho_overall = {
        'mean': float(np.mean(ortho_means)),
        'std': float(np.std(ortho_stds)),
        'min': float(np.min(ortho_means)),
        'max': float(np.max(ortho_means))
    }

    master_summary['alpha_results'][alpha_str]['orthogonal'] = {
        'output_dir': ortho_dir,
        'ablation_type': 'orthogonal_projection_stress_full_method',
        'alpha': alpha,
        'images': ortho_images,
        'overall_stats': ortho_overall
    }

    print(f"   ✅ 正交完成: mean={ortho_overall['mean']:.4f}, std={ortho_overall['std']:.4f}")

    # ---------- 6c. 当前 alpha 对比打印 ----------
    print(f"\n📊 [alpha={alpha}] 对比统计:")
    print(f"   非正交  →  Mean: {no_ortho_overall['mean']:.4f} | Std: {no_ortho_overall['std']:.4f}")
    print(f"   正交    →  Mean: {ortho_overall['mean']:.4f} | Std: {ortho_overall['std']:.4f}")
    print(f"   💡 Std 差距: {abs(no_ortho_overall['std'] - ortho_overall['std']):.4f}")

# ==========================================
# 7. 保存总摘要
# ==========================================
summary_path = os.path.join(
    base_output_dir,
    f"{config['experiment_name']}_ablation_obj_stress_multi_alpha_summary.json"
)
with open(summary_path, 'w') as f:
    json.dump(master_summary, f, indent=2)

print(f"\n{'='*60}")
print(f"🎉 全部极端 alpha 压力测试完成！")
print(f"{'='*60}")
print(f"   总摘要: {summary_path}")
print(f"\n📈 跨 alpha 汇总:")
for alpha in stress_alphas:
    alpha_str = f"alpha_{alpha}"
    no_o = master_summary['alpha_results'][alpha_str]['non_orthogonal']['overall_stats']
    orth = master_summary['alpha_results'][alpha_str]['orthogonal']['overall_stats']
    print(f"   alpha={alpha}: 非正交(std={no_o['std']:.4f}) vs 正交(std={orth['std']:.4f})")

print(f"\n   最终显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"   🚀 高性能模式：保持所有模型和缓存常驻GPU！")
