#!/usr/bin/env python3
"""
生成无水印的基准图像（极致性能版 - 50GB显存全速运行）
优化策略：
- 保持所有模型常驻GPU
- 批量推理（batch inference）
- 预编码所有prompts并缓存于GPU
- 不调用显存清理，最大化利用显存换速度
"""

import torch
from diffusers import FluxPipeline
from PIL import Image
import numpy as np
import os
import json
import random
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# 加载配置
with open('config.json', 'r') as f:
    config = json.load(f)

os.environ['HF_HOME'] = config['hf_cache']

output_dir = os.path.join(config['output_base_dir'], 'pic', 'no_watermarked_img')
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. 获取 Prompts
# ==========================================
def get_prompts(num_samples=200, seed=42):
    """从MS-COCO 2017和Stable-Diffusion-Prompts各取一半"""
    random.seed(seed)
    np.random.seed(seed)

    prompts = []
    sources = []

    # 尝试从 Stable-Diffusion-Prompts 加载
    try:
        print("📚 尝试加载 Stable-Diffusion-Prompts 数据集...")
        from datasets import load_dataset
        sd_dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="test", streaming=True)

        dataset_prompts = []
        for i, sample in enumerate(sd_dataset):
            if i >= num_samples:
                break
            dataset_prompts.append(sample['Prompt'])

        if len(dataset_prompts) >= num_samples // 2:
            selected_sd = random.sample(dataset_prompts, num_samples // 2)
            prompts.extend(selected_sd)
            sources.extend(['sd_prompts'] * len(selected_sd))
            print(f"   从 SD-Prompts 获取 {len(selected_sd)} 条 prompts")
        else:
            raise ValueError(f"SD-Prompts 样本不足")

    except Exception as e:
        print(f"   SD-Prompts 加载失败: {e}")

    # 尝试从 MS-COCO 2017 加载
    try:
        print("📚 尝试加载 MS-COCO 2017 验证集...")
        from datasets import load_dataset
        coco_dataset = load_dataset("yerevann/coco-karpathy", split="validation", streaming=True)

        coco_prompts = []
        for i, sample in enumerate(coco_dataset):
            if i >= num_samples:
                break
            captions = sample.get('sentences', [])
            if captions:
                coco_prompts.append(captions[0])

        if len(coco_prompts) >= num_samples // 2:
            selected_coco = random.sample(coco_prompts, num_samples // 2)
            prompts.extend(selected_coco)
            sources.extend(['coco2017'] * len(selected_coco))
            print(f"   从 MS-COCO 获取 {len(selected_coco)} 条 prompts")
        else:
            raise ValueError(f"MS-COCO 样本不足")

    except Exception as e:
        print(f"   MS-COCO 加载失败: {e}")

    # 如果数据集加载不足，使用默认prompts填充
    if len(prompts) < num_samples:
        default_prompts = [
            "a cat holding a sign that says hello world",
            "a golden retriever playing in a sunny park",
            "a red sports car on a mountain road",
            "a steaming cup of coffee on a wooden table",
            "a sunset over the ocean with palm trees",
            "a vintage bicycle leaning against a brick wall",
            "a colorful hot air balloon floating in blue sky",
            "a snowy mountain peak at sunrise",
            "a bowl of fresh strawberries on white marble",
            "an astronaut floating in space near earth",
            "a cozy library with leather chairs and books",
            "a blooming cherry blossom tree in spring",
            "a modern kitchen with stainless steel appliances",
            "a butterfly resting on a purple flower",
            "a rainy city street with neon reflections",
            "a rustic cabin in a pine forest",
            "a plate of sushi on bamboo mat",
            "a lightning storm over dark mountains",
            "a fluffy white kitten sleeping in a basket",
            "a sailboat on calm turquoise water",
            "a majestic lion portrait in golden light",
            "a medieval castle on a hill at dusk",
            "a fresh pizza straight from the oven",
            "a field of lavender under blue sky",
            "a Japanese zen garden with stone lanterns",
            "a tiger walking through the jungle",
            "a waterfall cascading down rocks",
            "a delicious stack of pancakes with syrup",
            "a vintage typewriter on a wooden desk",
            "a peaceful lake with mountains in background",
            "a basketball player making a slam dunk",
            "a beautiful garden with colorful flowers",
            "a classic muscle car in red",
            "a cat sleeping on a windowsill",
            "a mountain lake reflection at dawn",
            "a delicious chocolate cake on a plate",
            "an elephant walking across the savanna",
            "a guitar leaning against an amplifier",
            "a snowy forest with a walking path",
            "a parrot perched on a tree branch",
            "a skateboarder performing a trick",
            "a bowl of ramen with chopsticks",
            "a dramatic cliff overlooking the ocean",
            "a group of penguins on ice",
            "a vintage camera on a table",
            "a sunflower field at golden hour",
            "a wolf howling at the moon",
            "a cup of tea with steam rising",
            "a futuristic cityscape at night",
            "a dog catching a frisbee in mid-air",
        ]
        needed = num_samples - len(prompts)
        prompts.extend(default_prompts[:needed])
        sources.extend(['default'] * needed)

    return prompts[:num_samples], sources[:num_samples]


# ==========================================
# 2. 加载模型（直接到GPU，保持常驻）
# ==========================================
print("🚀 加载 FLUX 模型到GPU...")
pipe = FluxPipeline.from_pretrained(
    config['model_path'],
    torch_dtype=torch.bfloat16
)

# 性能优化：启用VAE tiling和attention slicing加速（不是显存优化）
pipe.enable_vae_tiling()
pipe.enable_attention_slicing(slice_size="auto")

# 直接加载到GPU并保持常驻
pipe.to("cuda")
print(f"   模型已加载到GPU，显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# 3. 预编码所有prompts（缓存于GPU）
# ==========================================
print("\n📋 获取 prompts...")
prompts, sources = get_prompts(config['num_samples'], config['prompt_seed'])

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

# 保存prompts供后续使用
prompts_data = {
    'experiment_name': config['experiment_name'],
    'timestamp': datetime.now().isoformat(),
    'prompts': prompts,
    'sources': sources,
    'config': config
}
prompts_path = os.path.join(config['output_base_dir'], 'prompts.json')
with open(prompts_path, 'w') as f:
    json.dump(prompts_data, f, indent=2)
print(f"   Prompts 已保存至: {prompts_path}")

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
            'source': sources[img_idx],
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
