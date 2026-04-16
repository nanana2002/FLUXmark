#!/usr/bin/env python3
"""
语义修改攻击测试 - 使用 SDXL InstructPix2Pix
测试场景：修改图像中的文字内容、改变表情等
"""

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import os
import sys

# 模型路径（从 download_models.py 输出获取）
MODEL_PATH = "/data/daiyina/modelscope_cache/AI-ModelScope/instruct-pix2pix"

def load_model():
    """加载 InstructPix2Pix 模型"""
    print("🚀 加载 SDXL InstructPix2Pix...")
    
    # 如果没下载，尝试从 HuggingFace 镜像加载
    if not os.path.exists(MODEL_PATH):
        print("   本地模型不存在，从 ModelScope 下载...")
        from modelscope import snapshot_download
        model_path = snapshot_download("AI-ModelScope/instruct-pix2pix")
    else:
        model_path = MODEL_PATH
    
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,  # 禁用安全检查器省显存
    )
    pipe = pipe.to("cuda")
    
    # 启用内存优化
    pipe.enable_attention_slicing()
    
    return pipe


def semantic_text_change(pipe, image_path, output_path):
    """
    语义攻击1: 改变文字内容
    例如：把 "hello world" 改成 "goodbye world"
    """
    print("\n✏️ 攻击: 修改文字内容...")
    
    image = Image.open(image_path).convert("RGB")
    
    # 指令式修改
    instructions = [
        "change the text to 'goodbye world'",
        "make the sign say 'i am fake'",
        "change the text to 'stolen image'",
    ]
    
    results = []
    for i, instruction in enumerate(instructions):
        print(f"   指令 {i+1}: {instruction}")
        
        modified = pipe(
            instruction,
            image=image,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        
        save_path = output_path.replace('.png', f'_text_{i}.png')
        modified.save(save_path)
        results.append((instruction, save_path))
        print(f"   已保存: {save_path}")
    
    return results


def semantic_expression_change(pipe, image_path, output_path):
    """
    语义攻击2: 改变主体表情/状态
    """
    print("\n😺 攻击: 修改主体表情...")
    
    image = Image.open(image_path).convert("RGB")
    
    instructions = [
        "make the cat look angry",
        "make the cat smile",
        "make the cat sad and crying",
    ]
    
    results = []
    for i, instruction in enumerate(instructions):
        print(f"   指令 {i+1}: {instruction}")
        
        modified = pipe(
            instruction,
            image=image,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]
        
        save_path = output_path.replace('.png', f'_expr_{i}.png')
        modified.save(save_path)
        results.append((instruction, save_path))
        print(f"   已保存: {save_path}")
    
    return results


def semantic_style_change(pipe, image_path, output_path):
    """
    语义攻击3: 改变艺术风格
    """
    print("\n🎨 攻击: 修改艺术风格...")
    
    image = Image.open(image_path).convert("RGB")
    
    instructions = [
        "turn it into a oil painting",
        "make it look like a cartoon",
        "convert to watercolor style",
    ]
    
    results = []
    for i, instruction in enumerate(instructions):
        print(f"   指令 {i+1}: {instruction}")
        
        modified = pipe(
            instruction,
            image=image,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]
        
        save_path = output_path.replace('.png', f'_style_{i}.png')
        modified.save(save_path)
        results.append((instruction, save_path))
        print(f"   已保存: {save_path}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='语义修改攻击测试')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output-dir', type=str, default='./semantic_attacks', help='输出目录')
    parser.add_argument('--attack-type', type=str, default='all', 
                       choices=['text', 'expression', 'style', 'all'],
                       help='攻击类型')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    pipe = load_model()
    
    # 生成输出路径
    base_name = os.path.basename(args.image)
    output_path = os.path.join(args.output_dir, base_name)
    
    # 执行攻击
    all_results = []
    
    if args.attack_type in ['text', 'all']:
        results = semantic_text_change(pipe, args.image, output_path)
        all_results.extend(results)
    
    if args.attack_type in ['expression', 'all']:
        results = semantic_expression_change(pipe, args.image, output_path)
        all_results.extend(results)
    
    if args.attack_type in ['style', 'all']:
        results = semantic_style_change(pipe, args.image, output_path)
        all_results.extend(results)
    
    print(f"\n✅ 完成！所有攻击图像保存在: {args.output_dir}")
    print("\n生成的攻击图像:")
    for instruction, path in all_results:
        print(f"  - {instruction}: {path}")


if __name__ == '__main__':
    main()
