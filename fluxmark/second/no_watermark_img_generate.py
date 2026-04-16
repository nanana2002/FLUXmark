#!/usr/bin/env python3
"""
生成无水印的基准图像（内存优化版 - 22GB显存适配）
从 MS-COCO 2017 和 Stable-Diffusion-Prompts 数据集获取 prompts
"""

import torch
from diffusers import FluxPipeline
from PIL import Image
import numpy as np
import os
import gc
import json
import random
from datetime import datetime
from tqdm import tqdm

# 加载配置
with open('config.json', 'r') as f:
    config = json.load(f)

os.environ['HF_HOME'] = config['hf_cache']

output_dir = os.path.join(config['output_base_dir'], 'pic', 'no_watermarked_img')
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. 获取 Prompts（从两个数据集各取100条）
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
            if i >= num_samples:  # 取足够样本
                break
            dataset_prompts.append(sample['Prompt'])

        if len(dataset_prompts) >= num_samples // 2:
            selected_sd = random.sample(dataset_prompts, num_samples // 2)
            prompts.extend(selected_sd)
            sources.extend(['sd_prompts'] * len(selected_sd))
            print(f"   从 SD-Prompts 获取 {len(selected_sd)} 条 prompts")
        else:
            raise ValueError(f"SD-Prompts 样本不足: {len(dataset_prompts)} < {num_samples // 2}")

    except Exception as e:
        print(f"   SD-Prompts 加载失败: {e}")
        print("   使用默认prompts补充...")

    # 尝试从 MS-COCO 2017 加载
    try:
        print("📚 尝试加载 MS-COCO 2017 验证集...")
        from datasets import load_dataset
        coco_dataset = load_dataset("yerevann/coco-karpathy", split="validation", streaming=True)

        coco_prompts = []
        for i, sample in enumerate(coco_dataset):
            if i >= num_samples:
                break
            # 使用每个图像的第一个caption
            captions = sample.get('sentences', [])
            if captions:
                coco_prompts.append(captions[0])

        if len(coco_prompts) >= num_samples // 2:
            selected_coco = random.sample(coco_prompts, num_samples // 2)
            prompts.extend(selected_coco)
            sources.extend(['coco2017'] * len(selected_coco))
            print(f"   从 MS-COCO 获取 {len(selected_coco)} 条 prompts")
        else:
            raise ValueError(f"MS-COCO 样本不足: {len(coco_prompts)} < {num_samples // 2}")

    except Exception as e:
        print(f"   MS-COCO 加载失败: {e}")
        print("   使用默认prompts补充...")

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
# 2. 加载模型（关键：先不移动到CUDA）
# ==========================================
print("🚀 加载 FLUX 模型...")
pipe = FluxPipeline.from_pretrained(
    config['model_path'],
    torch_dtype=torch.bfloat16
)

# ==========================================
# 3. 预编码所有prompts（省内存关键步骤）
# ==========================================
print("\n📋 获取 prompts...")
prompts, sources = get_prompts(config['num_samples'], config['prompt_seed'])

print(f"🔤 预编码 {len(prompts)} 个 prompts...")
encoded_prompts = []
for prompt in tqdm(prompts, desc="编码进度"):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=256
        )
    # 关键：立即移到CPU保存，避免占用GPU显存
    encoded_prompts.append({
        'prompt_embeds': prompt_embeds.cpu(),
        'pooled_prompt_embeds': pooled_prompt_embeds.cpu(),
        'text_ids': text_ids.cpu()
    })
    # 立即删除GPU张量并清理缓存
    del prompt_embeds, pooled_prompt_embeds, text_ids
    torch.cuda.empty_cache()

print(f"   已编码 {len(encoded_prompts)} 个prompts")

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
# 4. 卸载Text Encoders节省显存（关键！）
# ==========================================
print("🧹 卸载Text Encoders节省显存...")
del pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer, pipe.tokenizer_2
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()

# 现在才移动到cuda
pipe.to("cuda")
print(f"   模型已加载到GPU，当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# 5. 批量生成无水印图像
# ==========================================
print(f"\n🎨 开始生成 {len(prompts)} 张无水印图像...\n")

results_summary = {
    'experiment_name': config['experiment_name'],
    'timestamp': datetime.now().isoformat(),
    'total_images': len(prompts),
    'config': config,
    'images': []
}

for idx, prompt in enumerate(tqdm(prompts, desc="生成进度")):
    img_id = f"no_wm_{idx:03d}"

    # 使用预编码的prompts（关键：需要时移到GPU，用完立即释放）
    encoded = encoded_prompts[idx]
    prompt_embeds = encoded['prompt_embeds'].to("cuda")
    pooled_prompt_embeds = encoded['pooled_prompt_embeds'].to("cuda")
    text_ids = encoded['text_ids'].to("cuda")

    # 生成图像
    with torch.no_grad():
        image = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=config['num_inference_steps'],
            guidance_scale=config['guidance_scale'],
            height=config['height'],
            width=config['width'],
        ).images[0]

    # 保存图像
    img_filename = f"{img_id}.png"
    img_path = os.path.join(output_dir, img_filename)
    image.save(img_path)

    img_result = {
        'id': img_id,
        'prompt': prompt,
        'source': sources[idx],
        'image_file': img_filename,
        'image_path': img_path
    }
    results_summary['images'].append(img_result)

    # 关键：清理GPU显存
    del prompt_embeds, pooled_prompt_embeds, text_ids
    torch.cuda.empty_cache()

# ==========================================
# 6. 保存实验摘要
# ==========================================
summary_path = os.path.join(output_dir, f'{config["experiment_name"]}_no_watermark_summary.json')
with open(summary_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

# 最终清理
del pipe, encoded_prompts
gc.collect()
torch.cuda.empty_cache()

print(f"\n✅ 无水印图像生成完成！")
print(f"   图像保存位置: {output_dir}")
print(f"   实验摘要: {summary_path}")
print(f"   最终显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
