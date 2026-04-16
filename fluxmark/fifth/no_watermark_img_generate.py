#!/usr/bin/env python3
"""
生成无水印的基准图像（极致性能版 - 50GB显存全速运行）
优化策略：
- 保持所有模型常驻GPU
- 批量推理（batch inference）
- 预编码所有prompts并缓存于GPU
- 不调用显存清理，最大化利用显存换速度

修改：从 prompts.json 读取预生成的 prompts（由 generate_prompt.py 生成）
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
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm

output_dir = os.path.join(config['output_base_dir'], 'pic', 'no_watermarked_img')
os.makedirs(output_dir, exist_ok=True)

print(f"🎮 使用 GPU: {config.get('gpu_id', 0)}")

# ==========================================
# 1. 获取 Prompts（从 prompts.json 读取）
# ==========================================
def get_prompts():
    """从 prompts.json 读取预生成的 prompts"""
    prompts_path = os.path.join(config['output_base_dir'], 'prompts.json')

    if not os.path.exists(prompts_path):
        print(f"❌ 错误: 未找到 {prompts_path}")
        print("   请先运行: python3 generate_prompt.py")
        exit(1)

    with open(prompts_path, 'r') as f:
        data = json.load(f)

    prompts = data['prompts']
    print(f"📋 从 prompts.json 加载了 {len(prompts)} 条 prompts")
    print(f"   来源: {data['source']['coco_count']} 条 COCO + {data['source']['sd_count']} 条 SD-Prompts")

    return prompts


# ==========================================
# 2. 加载模型（直接到GPU，保持常驻）
# ==========================================
print("🚀 加载 FLUX 模型到GPU...")
pipe = FluxPipeline.from_pretrained(
    config['model_path'],
    torch_dtype=torch.bfloat16,
    strict=False
)

# 性能优化：启用VAE tiling和attention slicing加速（不是显存优化）
# pipe.enable_vae_tiling()
pipe.enable_attention_slicing(slice_size="auto")

# 直接加载到GPU并保持常驻
pipe.to("cuda")
print(f"   模型已加载到GPU，显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# 3. 预编码所有prompts（缓存于GPU）
# ==========================================
print("\n📋 获取 prompts...")
prompts = get_prompts()

print(f"🔤 预编码 {len(prompts)} 个 prompts 到GPU缓存...")
encoded_prompts = []
for prompt in tqdm(prompts, desc="编码进度"):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=256
        )
    # 保持在GPU上缓存
    encoded_prompts.append({
        'prompt_embeds': prompt_embeds,  # 保持在GPU
        'pooled_prompt_embeds': pooled_prompt_embeds,
        'text_ids': text_ids
    })

print(f"   已编码 {len(encoded_prompts)} 个prompts到GPU")
print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# 4. 批量生成无水印图像（高性能模式）
# ==========================================
print(f"\n🎨 开始生成 {len(prompts)} 张无水印图像（高性能模式）...\n")

results_summary = {
    'experiment_name': config['experiment_name'],
    'timestamp': datetime.now().isoformat(),
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
            num_images_per_prompt=1,
        ).images

    # 保存图像
    for idx, img_idx in enumerate(batch_indices):
        img_id = f"no_wm_{img_idx:03d}"
        img_filename = f"{img_id}.png"
        img_path = os.path.join(output_dir, img_filename)
        images[idx].save(img_path)

        img_result = {
            'id': img_id,
            'prompt': prompts[img_idx],
            'image_file': img_filename,
            'image_path': img_path
        }
        results_summary['images'].append(img_result)

print(f"\n✅ 无水印图像生成完成！")
print(f"   图像保存位置: {output_dir}")

# ==========================================
# 5. 保存实验摘要
# ==========================================
summary_path = os.path.join(output_dir, f'{config["experiment_name"]}_no_watermark_summary.json')
with open(summary_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"   实验摘要: {summary_path}")
print(f"   最终显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"   🚀 高性能模式：保持所有模型常驻GPU，无需重复加载！")
