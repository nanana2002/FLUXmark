#!/usr/bin/env python3
"""
生成无水印的基准图像（内存优化版）
从 Stable-Diffusion-Prompts 数据集获取 prompts
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
# 1. 获取 Prompts
# ==========================================
def get_prompts(num_samples=20, seed=42):
    """获取 prompts"""
    random.seed(seed)
    np.random.seed(seed)
    
    prompts = []
    sources = []
    
    # 尝试从 Stable-Diffusion-Prompts 加载
    try:
        print("📚 尝试加载 Stable-Diffusion-Prompts 数据集...")
        from datasets import load_dataset
        sd_dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts", split="test", streaming=True)
        
        # 收集足够的样本
        dataset_prompts = []
        for i, sample in enumerate(sd_dataset):
            if i >= num_samples * 2:  # 多取一些以防重复
                break
            dataset_prompts.append(sample['Prompt'])
        
        if len(dataset_prompts) >= num_samples:
            selected = random.sample(dataset_prompts, num_samples)
            prompts.extend(selected)
            sources.extend(['sd_prompts'] * len(selected))
            print(f"   从 SD-Prompts 获取 {len(selected)} 条 prompts")
        else:
            raise ValueError(f"数据集样本不足: {len(dataset_prompts)} < {num_samples}")
            
    except Exception as e:
        print(f"   SD-Prompts 加载失败: {e}")
        print("   使用默认 prompts...")
        
        # 使用默认prompts作为fallback
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
            "a astronaut floating in space near earth",
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
            "a Japanese zen garden with stone lanterns"
        ]
        prompts = default_prompts[:num_samples]
        sources = ['default'] * len(prompts)
    
    return prompts[:num_samples], sources[:num_samples]

# ==========================================
# 2. 加载模型（先不移动到CUDA）
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
    encoded_prompts.append({
        'prompt_embeds': prompt_embeds.cpu(),  # 移到CPU保存
        'pooled_prompt_embeds': pooled_prompt_embeds.cpu(),
        'text_ids': text_ids.cpu()
    })
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
prompts_path = os.path.join(config['output_base_dir'], 'prompts_used.json')
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
    
    # 使用预编码的prompts
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
    
    # 清理GPU显存（关键！）
    del prompt_embeds, pooled_prompt_embeds, text_ids
    torch.cuda.empty_cache()

# ==========================================
# 6. 保存实验摘要
# ==========================================
summary_path = os.path.join(output_dir, f'{config["experiment_name"]}_no_watermark_summary.json')
with open(summary_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

# 最终清理
del pipe
gc.collect()
torch.cuda.empty_cache()

print(f"\n✅ 无水印图像生成完成！")
print(f"   图像保存位置: {output_dir}")
print(f"   实验摘要: {summary_path}")
print(f"   最终显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
