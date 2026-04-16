#!/usr/bin/env python3
"""
FLUX.1-Fill-dev 语义修改攻击测试
使用 Inpainting/Fill 方式进行局部语义修改
"""

import torch
from diffusers import FluxFillPipeline
from PIL import Image, ImageDraw
import os
import argparse
import numpy as np
from datetime import datetime

def load_flux_fill_model():
    """加载 FLUX Fill 模型"""
    model_path = "/data/daiyina/project_flux/model/flux-fill"
    
    print("🚀 加载 FLUX.1-Fill-dev...")
    print(f"   路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型未找到: {model_path}")
        print("   请先运行: bash download_models.sh")
        return None
    
    pipe = FluxFillPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    
    # 显存优化策略（关键！）
    print("   启用显存优化...")
    
    # 方法1: 顺序CPU卸载（最省显存，但稍慢）
    # 将不用的组件自动移到CPU，需要时再加载到GPU
    pipe.enable_sequential_cpu_offload()
    print("   ✓ 已启用 sequential CPU offload")
    
    # 方法2: VAE切片（处理大图像时减少显存）
    pipe.enable_vae_slicing()
    print("   ✓ 已启用 VAE slicing")
    
    # 方法3: 注意力切片
    pipe.enable_attention_slicing(slice_size="auto")
    print("   ✓ 已启用 attention slicing")
    
    # 注意：enable_sequential_cpu_offload 会自动处理 device，不需要 pipe.to("cuda")
    # 如果上面的方法还是OOM，可以尝试更激进的 enable_model_cpu_offload
    
    print("✅ 模型加载完成（带显存优化）")
    return pipe


def create_center_mask(image_size, mask_ratio=0.3):
    """创建中心区域的掩码"""
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 中心矩形
    mask_w = int(width * mask_ratio)
    mask_h = int(height * mask_ratio)
    x1 = (width - mask_w) // 2
    y1 = (height - mask_h) // 2
    x2 = x1 + mask_w
    y2 = y1 + mask_h
    
    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask, (x1, y1, x2, y2)


def create_sign_mask(image_size):
    """创建针对牌子的掩码（假设牌子在图像中心偏下）"""
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 中心偏下区域（假设牌子位置）
    mask_w = int(width * 0.5)
    mask_h = int(height * 0.25)
    x1 = (width - mask_w) // 2
    y1 = int(height * 0.55)  # 偏下
    x2 = x1 + mask_w
    y2 = y1 + mask_h
    
    draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask, (x1, y1, x2, y2)


def create_random_mask(image_size, num_regions=3):
    """创建随机掩码"""
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    for _ in range(num_regions):
        # 随机椭圆区域
        w = np.random.randint(width // 6, width // 3)
        h = np.random.randint(height // 6, height // 3)
        x = np.random.randint(0, width - w)
        y = np.random.randint(0, height - h)
        
        draw.ellipse([x, y, x + w, y + h], fill=255)
    
    return mask, None


def attack_text_change_with_mask(pipe, image, mask, output_dir, base_name):
    """攻击1: 使用 Mask 修改文字区域"""
    print("\n✏️  攻击类型: 文字区域修改（带Mask）")
    
    prompts = [
        ("a sign that says goodbye world", "fill_goodbye"),
        ("a sign that says fake image", "fill_fake"),
        ("a sign that says hello mars", "fill_mars"),
        ("an empty sign with no text", "fill_empty"),
    ]
    
    results = []
    for prompt, suffix in prompts:
        print(f"   提示词: '{prompt}'")
        
        output_path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
        
        # FLUX Fill 生成
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=28,
            guidance_scale=30.0,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        
        result.save(output_path)
        print(f"   已保存: {output_path}")
        results.append((prompt, output_path))
    
    return results


def attack_expression_change_with_mask(pipe, image, output_dir, base_name):
    """攻击2: 修改主体表情（脸部区域）"""
    print("\n😺 攻击类型: 表情修改（带Mask）")
    
    # 创建脸部区域掩码（假设主体在上半部分）
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # 上半部分区域（脸部大概位置）
    mask_w = int(width * 0.6)
    mask_h = int(height * 0.4)
    x1 = (width - mask_w) // 2
    y1 = int(height * 0.1)
    draw.rectangle([x1, y1, x1 + mask_w, y1 + mask_h], fill=255)
    
    prompts = [
        ("an angry cat face", "fill_angry_face"),
        ("a happy smiling cat", "fill_happy_face"),
        ("a sad crying cat", "fill_sad_face"),
    ]
    
    results = []
    for prompt, suffix in prompts:
        print(f"   提示词: '{prompt}'")
        
        output_path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
        
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=28,
            guidance_scale=30.0,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        
        result.save(output_path)
        print(f"   已保存: {output_path}")
        results.append((prompt, output_path))
    
    return results


def attack_style_change_full(pipe, image, output_dir, base_name):
    """攻击3: 全局风格转换（全图mask）"""
    print("\n🎨 攻击类型: 全局风格转换")
    
    # 全图掩码
    mask = Image.new("L", image.size, 255)
    
    prompts = [
        ("an oil painting of a cat holding a sign", "fill_oil_full"),
        ("a cartoon of a cat holding a sign", "fill_cartoon_full"),
        ("a watercolor painting of a cat", "fill_watercolor_full"),
        ("a pencil sketch of a cat", "fill_sketch_full"),
    ]
    
    results = []
    for prompt, suffix in prompts:
        print(f"   提示词: '{prompt}'")
        
        output_path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
        
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=28,
            guidance_scale=30.0,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        
        result.save(output_path)
        print(f"   已保存: {output_path}")
        results.append((prompt, output_path))
    
    return results


def attack_combined_modifications(pipe, image, output_dir, base_name):
    """攻击4: 多区域组合修改"""
    print("\n🔥 攻击类型: 组合修改")
    
    width, height = image.size
    
    # 组合1：改文字 + 改表情
    print("   组合1：修改文字+表情")
    mask1 = Image.new("L", (width, height), 0)
    draw1 = ImageDraw.Draw(mask1)
    # 文字区域
    draw1.rectangle([width*0.25, height*0.55, width*0.75, height*0.75], fill=255)
    # 脸部区域
    draw1.rectangle([width*0.3, height*0.1, width*0.7, height*0.4], fill=255)
    
    output_path = os.path.join(output_dir, f"{base_name}_fill_combo1.png")
    result = pipe(
        prompt="an angry cat holding a sign that says stolen",
        image=image,
        mask_image=mask1,
        num_inference_steps=28,
        guidance_scale=30.0,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    result.save(output_path)
    print(f"   已保存: {output_path}")
    
    # 组合2：大幅修改主体
    print("   组合2：改变主体类型")
    mask2 = Image.new("L", (width, height), 0)
    draw2 = ImageDraw.Draw(mask2)
    draw2.rectangle([width*0.2, height*0.2, width*0.8, height*0.8], fill=255)
    
    output_path2 = os.path.join(output_dir, f"{base_name}_fill_combo2.png")
    result2 = pipe(
        prompt="a dog holding a sign that says hello world",
        image=image,
        mask_image=mask2,
        num_inference_steps=28,
        guidance_scale=30.0,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    result2.save(output_path2)
    print(f"   已保存: {output_path2}")
    
    return [
        ("angry cat + stolen sign", output_path),
        ("dog instead of cat", output_path2)
    ]


def visualize_masks(image, output_dir, base_name):
    """可视化不同的掩码位置"""
    print("\n📐 生成掩码可视化...")
    
    # 中心掩码
    mask_center, coords = create_center_mask(image.size, 0.3)
    mask_center.save(os.path.join(output_dir, f"{base_name}_mask_center.png"))
    
    # 牌子掩码
    mask_sign, coords = create_sign_mask(image.size)
    mask_sign.save(os.path.join(output_dir, f"{base_name}_mask_sign.png"))
    
    # 随机掩码
    mask_random, _ = create_random_mask(image.size, 3)
    mask_random.save(os.path.join(output_dir, f"{base_name}_mask_random.png"))
    
    print("   掩码可视化已保存")


def main():
    parser = argparse.ArgumentParser(description='FLUX Fill 语义修改攻击测试')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output-dir', type=str, default='./flux_fill_attacks', help='输出目录')
    parser.add_argument('--attack-type', type=str, default='all',
                       choices=['text', 'expression', 'style', 'combined', 'all'],
                       help='攻击类型')
    parser.add_argument('--mask-type', type=str, default='sign',
                       choices=['center', 'sign', 'random'],
                       help='掩码类型（用于text攻击）')
    parser.add_argument('--steps', type=int, default=28, help='推理步数')
    parser.add_argument('--guidance', type=float, default=30.0, help='guidance scale')
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("🧪 FLUX.1-Fill-dev 语义攻击测试")
    print("="*60)
    print(f"输入图像: {args.image}")
    print(f"输出目录: {output_dir}")
    print(f"攻击类型: {args.attack_type}")
    print(f"掩码类型: {args.mask_type}")
    print("="*60)
    
    # 加载模型
    pipe = load_flux_fill_model()
    if pipe is None:
        return
    
    # 加载输入图像
    print(f"\n📷 加载图像: {args.image}")
    image = Image.open(args.image).convert("RGB")
    print(f"   尺寸: {image.size}")
    
    # 获取基础文件名
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # 保存原始图像副本
    image.save(os.path.join(output_dir, f"{base_name}_original.png"))
    
    # 生成并保存掩码可视化
    visualize_masks(image, output_dir, base_name)
    
    # 根据掩码类型创建掩码
    if args.mask_type == 'center':
        mask, _ = create_center_mask(image.size, 0.3)
    elif args.mask_type == 'sign':
        mask, _ = create_sign_mask(image.size)
    else:  # random
        mask, _ = create_random_mask(image.size, 3)
    
    mask.save(os.path.join(output_dir, f"{base_name}_mask_used.png"))
    
    # 执行攻击
    all_results = []
    
    if args.attack_type in ['text', 'all']:
        results = attack_text_change_with_mask(pipe, image, mask, output_dir, base_name)
        all_results.extend(results)
    
    if args.attack_type in ['expression', 'all']:
        results = attack_expression_change_with_mask(pipe, image, output_dir, base_name)
        all_results.extend(results)
    
    if args.attack_type in ['style', 'all']:
        results = attack_style_change_full(pipe, image, output_dir, base_name)
        all_results.extend(results)
    
    if args.attack_type in ['combined', 'all']:
        results = attack_combined_modifications(pipe, image, output_dir, base_name)
        all_results.extend(results)
    
    # 生成报告
    report_path = os.path.join(output_dir, "attack_report.txt")
    with open(report_path, 'w') as f:
        f.write("FLUX.1-Fill-dev 语义攻击测试报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"原始图像: {args.image}\n")
        f.write(f"生成时间: {timestamp}\n")
        f.write(f"攻击数量: {len(all_results)}\n")
        f.write(f"掩码类型: {args.mask_type}\n\n")
        
        for i, (prompt, path) in enumerate(all_results, 1):
            f.write(f"{i}. {prompt}\n")
            f.write(f"   输出: {path}\n\n")
    
    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"生成图像: {len(all_results)} 张")
    print(f"报告文件: {report_path}")
    print("="*60)
    
    print("\n生成文件列表:")
    for filename in sorted(os.listdir(output_dir)):
        print(f"  - {filename}")


if __name__ == '__main__':
    main()
