#!/usr/bin/env python3
"""
SDXL InstructPix2Pix 语义修改攻击测试
仅使用 SDXL，不需要 FLUX
"""

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import os
import argparse
from datetime import datetime

def load_sdxl_model():
    """加载 SDXL InstructPix2Pix 模型"""
    model_path = "/data/daiyina/project_flux/model/sdxl-instructpix2pix"
    
    print("🚀 加载 SDXL InstructPix2Pix...")
    print(f"   路径: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型未找到: {model_path}")
        print("   请先运行: bash download_models.sh")
        return None
    
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,  # 禁用安全检查器省显存
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()  # 启用内存优化
    
    print("✅ 模型加载完成")
    return pipe


def attack_text_change(pipe, image, output_dir, base_name):
    """攻击1: 修改文字内容"""
    print("\n✏️  攻击类型: 文字修改")
    
    instructions = [
        ("change the text to 'goodbye world'", "text_goodbye"),
        ("change the text to 'fake image'", "text_fake"),
        ("remove the text from the sign", "text_removed"),
        ("change the text to 'hello mars'", "text_mars"),
    ]
    
    results = []
    for instruction, suffix in instructions:
        print(f"   指令: '{instruction}'")
        
        output_path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
        
        # 生成修改后的图像
        edited = pipe(
            instruction,
            image=image,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        
        edited.save(output_path)
        print(f"   已保存: {output_path}")
        results.append((instruction, output_path))
    
    return results


def attack_expression_change(pipe, image, output_dir, base_name):
    """攻击2: 修改主体表情"""
    print("\n😺 攻击类型: 表情修改")
    
    instructions = [
        ("make the cat look angry", "expr_angry"),
        ("make the cat smile happily", "expr_happy"),
        ("make the cat look sad", "expr_sad"),
        ("make the cat look surprised", "expr_surprised"),
    ]
    
    results = []
    for instruction, suffix in instructions:
        print(f"   指令: '{instruction}'")
        
        output_path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
        
        edited = pipe(
            instruction,
            image=image,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        
        edited.save(output_path)
        print(f"   已保存: {output_path}")
        results.append((instruction, output_path))
    
    return results


def attack_style_change(pipe, image, output_dir, base_name):
    """攻击3: 修改艺术风格"""
    print("\n🎨 攻击类型: 风格转换")
    
    instructions = [
        ("turn it into an oil painting", "style_oil"),
        ("make it look like a cartoon", "style_cartoon"),
        ("convert to watercolor style", "style_watercolor"),
        ("make it a pencil sketch", "style_sketch"),
    ]
    
    results = []
    for instruction, suffix in instructions:
        print(f"   指令: '{instruction}'")
        
        output_path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
        
        edited = pipe(
            instruction,
            image=image,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        
        edited.save(output_path)
        print(f"   已保存: {output_path}")
        results.append((instruction, output_path))
    
    return results


def attack_combined(pipe, image, output_dir, base_name):
    """攻击4: 组合攻击（文字+风格）"""
    print("\n🔥 攻击类型: 组合攻击")
    
    instructions = [
        ("change text to 'stolen' and make it oil painting", "combo_text_oil"),
        ("make the cat angry and convert to cartoon style", "combo_angry_cartoon"),
    ]
    
    results = []
    for instruction, suffix in instructions:
        print(f"   指令: '{instruction}'")
        
        output_path = os.path.join(output_dir, f"{base_name}_{suffix}.png")
        
        edited = pipe(
            instruction,
            image=image,
            num_inference_steps=25,  # 更多步数处理复杂指令
            guidance_scale=8.0,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]
        
        edited.save(output_path)
        print(f"   已保存: {output_path}")
        results.append((instruction, output_path))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='SDXL 语义修改攻击测试')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output-dir', type=str, default='./sdxl_attacks', help='输出目录')
    parser.add_argument('--attack-type', type=str, default='all',
                       choices=['text', 'expression', 'style', 'combined', 'all'],
                       help='攻击类型')
    parser.add_argument('--steps', type=int, default=20, help='推理步数')
    parser.add_argument('--guidance', type=float, default=7.5, help='guidance scale')
    
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("🧪 SDXL InstructPix2Pix 语义攻击测试")
    print("="*60)
    print(f"输入图像: {args.image}")
    print(f"输出目录: {output_dir}")
    print(f"攻击类型: {args.attack_type}")
    print("="*60)
    
    # 加载模型
    pipe = load_sdxl_model()
    if pipe is None:
        return
    
    # 加载输入图像
    print(f"\n📷 加载图像: {args.image}")
    image = Image.open(args.image).convert("RGB")
    print(f"   尺寸: {image.size}")
    
    # 获取基础文件名
    base_name = os.path.splitext(os.path.basename(args.image))[0]
    
    # 执行攻击
    all_results = []
    
    if args.attack_type in ['text', 'all']:
        results = attack_text_change(pipe, image, output_dir, base_name)
        all_results.extend(results)
    
    if args.attack_type in ['expression', 'all']:
        results = attack_expression_change(pipe, image, output_dir, base_name)
        all_results.extend(results)
    
    if args.attack_type in ['style', 'all']:
        results = attack_style_change(pipe, image, output_dir, base_name)
        all_results.extend(results)
    
    if args.attack_type in ['combined', 'all']:
        results = attack_combined(pipe, image, output_dir, base_name)
        all_results.extend(results)
    
    # 生成报告
    report_path = os.path.join(output_dir, "attack_report.txt")
    with open(report_path, 'w') as f:
        f.write("SDXL InstructPix2Pix 语义攻击测试报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"原始图像: {args.image}\n")
        f.write(f"生成时间: {timestamp}\n")
        f.write(f"攻击数量: {len(all_results)}\n\n")
        
        for i, (instruction, path) in enumerate(all_results, 1):
            f.write(f"{i}. {instruction}\n")
            f.write(f"   输出: {path}\n\n")
    
    print("\n" + "="*60)
    print("✅ 测试完成！")
    print("="*60)
    print(f"输出目录: {output_dir}")
    print(f"生成图像: {len(all_results)} 张")
    print(f"报告文件: {report_path}")
    print("="*60)
    
    print("\n生成文件列表:")
    for instruction, path in all_results:
        filename = os.path.basename(path)
        print(f"  - {filename}")


if __name__ == '__main__':
    main()
