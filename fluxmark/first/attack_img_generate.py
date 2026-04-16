#!/usr/bin/env python3
"""
攻击图像生成脚本（内存优化版）
对水印图像进行多种攻击，生成被攻击后的图像
包括：传统攻击 + SDXL/FluxFill 语义修改攻击

优化策略：
1. 先执行所有传统攻击（无需深度学习模型）
2. 再加载SDXL，执行所有SDXL攻击，然后卸载
3. 最后加载FluxFill，执行所有FluxFill攻击，然后卸载
"""

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, FluxFillPipeline
import torchvision.transforms as T
from torchvision.transforms import functional as F_vision
from PIL import Image
import numpy as np
import io
import os
import json
from datetime import datetime
from tqdm import tqdm
import random
import gc

# 加载配置
with open('config.json', 'r') as f:
    config = json.load(f)

os.environ['HF_HOME'] = config['hf_cache']

# 基础路径
base_output_dir = os.path.join(config['output_base_dir'], 'pic', 'attack_watermarked_img')
watermarked_dir = os.path.join(config['output_base_dir'], 'pic', 'watermarked_img')

# 模型路径（用于I2I攻击）
SDXL_PATH = config.get('sdxl_path', '/data/daiyina/project_flux/model/sdxl-instructpix2pix')
FLUX_FILL_PATH = config.get('flux_fill_path', '/data/daiyina/project_flux/model/flux-fill')

# 检查模型路径是否存在
SDXL_AVAILABLE = os.path.exists(SDXL_PATH) and os.path.isdir(SDXL_PATH) and len(os.listdir(SDXL_PATH)) > 0
FLUX_FILL_AVAILABLE = os.path.exists(FLUX_FILL_PATH) and os.path.isdir(FLUX_FILL_PATH) and len(os.listdir(FLUX_FILL_PATH)) > 0

if not SDXL_AVAILABLE:
    print(f"⚠️  SDXL模型不存在或为空: {SDXL_PATH}")
    print("   将跳过SDXL攻击")
if not FLUX_FILL_AVAILABLE:
    print(f"⚠️  FluxFill模型不存在或为空: {FLUX_FILL_PATH}")
    print("   将跳过FluxFill攻击")

# 加载水印图像摘要
watermark_summary_path = os.path.join(watermarked_dir, f'{config["experiment_name"]}_watermark_summary.json')
with open(watermark_summary_path, 'r') as f:
    watermark_summary = json.load(f)

images_info = watermark_summary['images']
print(f"📋 加载了 {len(images_info)} 张水印图像信息")

# ==========================================
# 攻击函数定义
# ==========================================

def attack_jpeg(img, quality):
    """JPEG压缩攻击"""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def attack_blur(img, sigma, kernel_size=9):
    """高斯模糊攻击"""
    tensor = T.ToTensor()(img).unsqueeze(0)
    blurred = F_vision.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return T.ToPILImage()(blurred.squeeze(0))


def attack_crop_center(img, crop_ratio=0.5):
    """中心裁剪攻击"""
    w, h = img.size
    left = int(w * (1 - crop_ratio) / 2)
    top = int(h * (1 - crop_ratio) / 2)
    right = w - left
    bottom = h - top
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.Resampling.LANCZOS)


def attack_noise(img, std=0.05):
    """加高斯噪声攻击"""
    tensor = T.ToTensor()(img)
    noise = torch.randn_like(tensor) * std
    noisy = torch.clamp(tensor + noise, 0, 1)
    return T.ToPILImage()(noisy)


def attack_resize(img, scale=0.5):
    """缩放攻击"""
    w, h = img.size
    small = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    return small.resize((w, h), Image.Resampling.LANCZOS)


def attack_brightness_contrast(img, brightness=1.0, contrast=1.0):
    """亮度/对比度调整攻击"""
    tensor = T.ToTensor()(img).unsqueeze(0)
    tensor = torch.clamp(tensor * brightness, 0, 1)
    mean = tensor.mean()
    tensor = torch.clamp((tensor - mean) * contrast + mean, 0, 1)
    return T.ToPILImage()(tensor.squeeze(0))


def attack_black_block(img, block_size_ratio=0.25, position='center'):
    """添加黑色方块攻击"""
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    block_h = int(h * block_size_ratio)
    block_w = int(w * block_size_ratio)
    
    if position == 'center':
        y_start = (h - block_h) // 2
        x_start = (w - block_w) // 2
    elif position == 'random':
        y_start = random.randint(0, max(0, h - block_h))
        x_start = random.randint(0, max(0, w - block_w))
    elif position == 'corner_tl':
        y_start, x_start = 0, 0
    elif position == 'corner_tr':
        y_start, x_start = 0, w - block_w
    elif position == 'corner_bl':
        y_start, x_start = h - block_h, 0
    elif position == 'corner_br':
        y_start, x_start = h - block_h, w - block_w
    else:
        y_start = (h - block_h) // 2
        x_start = (w - block_w) // 2
    
    attacked = img_array.copy()
    attacked[y_start:y_start+block_h, x_start:x_start+block_w] = 0
    
    # 计算真实篡改掩码（8x8网格）
    true_mask = np.zeros((8, 8), dtype=bool)
    y_start_8 = int(y_start / h * 8)
    y_end_8 = int((y_start + block_h) / h * 8)
    x_start_8 = int(x_start / w * 8)
    x_end_8 = int((x_start + block_w) / w * 8)
    
    y_start_8 = max(0, min(7, y_start_8))
    y_end_8 = max(0, min(8, y_end_8))
    x_start_8 = max(0, min(7, x_start_8))
    x_end_8 = max(0, min(8, x_end_8))
    
    true_mask[y_start_8:y_end_8, x_start_8:x_end_8] = True
    
    return Image.fromarray(attacked), true_mask


def attack_sdedit_simple(img, noise_strength=0.3):
    """简化的SDEdit攻击：加噪声"""
    img_tensor = T.ToTensor()(img).unsqueeze(0)
    noise = torch.randn_like(img_tensor) * noise_strength
    noisy = torch.clamp(img_tensor + noise, 0, 1)
    return T.ToPILImage()(noisy.squeeze(0))


# ==========================================
# 深度学习攻击（分批执行）
# ==========================================

def run_sdxl_attacks(images_info, attack_configs):
    """
    执行所有SDXL攻击，然后卸载模型
    使用 StableDiffusionInstructPix2PixPipeline（非XL版本）
    """
    sdxl_configs = [c for c in attack_configs if c['type'] == 'sdxl_i2i']
    if not sdxl_configs:
        return
    
    if not SDXL_AVAILABLE:
        print(f"\n⏭️  跳过SDXL攻击（模型不可用）")
        return
    
    print(f"\n🚀 加载 SDXL InstructPix2Pix 模型...")
    print(f"   模型路径: {SDXL_PATH}")
    
    try:
        # 加载SDXL模型（使用正确的类）
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            SDXL_PATH,
            torch_dtype=torch.float16,
            safety_checker=None,  # 禁用安全检查器省显存
        )
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()  # 启用内存优化
        print("   SDXL模型加载完成")
        
        # 执行所有SDXL攻击
        for attack_config in sdxl_configs:
            attack_name = attack_config['name']
            params = attack_config['params']
            
            attack_dir = os.path.join(base_output_dir, attack_name)
            os.makedirs(attack_dir, exist_ok=True)
            
            print(f"\n   执行攻击: {attack_name}")
            
            for img_info in tqdm(images_info, desc=f"   {attack_name}", leave=False):
                img_id = img_info['id']
                img_path = os.path.join(watermarked_dir, img_info['image_file'])
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    
                    # 执行I2I转换
                    with torch.no_grad():
                        result = pipe(
                            prompt=params['instruction'],
                            image=img,
                            num_inference_steps=20,
                            guidance_scale=7.5,
                            generator=torch.Generator("cuda").manual_seed(42),
                        ).images[0]
                    
                    # 保存
                    attacked_path = os.path.join(attack_dir, f"{img_id}.png")
                    result.save(attacked_path)
                    
                    # 清理
                    del result
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"\n   ❌ {img_id} 失败: {e}")
                    continue
    except Exception as e:
        print(f"\n❌ SDXL模型加载失败: {e}")
        print("   跳过SDXL攻击")
    finally:
        # 卸载SDXL模型
        if 'pipe' in locals():
            print("\n🧹 卸载 SDXL 模型...")
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            print(f"   当前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


def run_fluxfill_attacks(images_info, attack_configs):
    """
    执行所有FluxFill攻击，然后卸载模型
    使用显存优化策略
    """
    fluxfill_configs = [c for c in attack_configs if c['type'] == 'flux_fill']
    if not fluxfill_configs:
        return
    
    if not FLUX_FILL_AVAILABLE:
        print(f"\n⏭️  跳过FluxFill攻击（模型不可用）")
        return
    
    print(f"\n🚀 加载 FluxFill 模型...")
    print(f"   模型路径: {FLUX_FILL_PATH}")
    
    try:
        # 加载FluxFill模型
        pipe = FluxFillPipeline.from_pretrained(
            FLUX_FILL_PATH,
            torch_dtype=torch.bfloat16,
        )
        
        # 显存优化策略（关键！）
        print("   启用显存优化...")
        pipe.enable_sequential_cpu_offload()  # 顺序CPU卸载
        pipe.enable_vae_slicing()  # VAE切片
        pipe.enable_attention_slicing(slice_size="auto")  # 注意力切片
        print("   ✓ 已启用 sequential CPU offload")
        print("   ✓ 已启用 VAE slicing")
        print("   ✓ 已启用 attention slicing")
        
        print("   FluxFill模型加载完成")
        
        # 执行所有FluxFill攻击
        for attack_config in fluxfill_configs:
            attack_name = attack_config['name']
            params = attack_config['params']
            
            attack_dir = os.path.join(base_output_dir, attack_name)
            os.makedirs(attack_dir, exist_ok=True)
            
            print(f"\n   执行攻击: {attack_name}")
            
            for img_info in tqdm(images_info, desc=f"   {attack_name}", leave=False):
                img_id = img_info['id']
                img_path = os.path.join(watermarked_dir, img_info['image_file'])
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    w, h = img.size
                    
                    # 创建中心mask
                    mask_ratio = params.get('mask_ratio', 0.3)
                    mask_w = int(w * mask_ratio)
                    mask_h = int(h * mask_ratio)
                    x_start = (w - mask_w) // 2
                    y_start = (h - mask_h) // 2
                    
                    mask = Image.new('L', (w, h), 0)
                    mask_draw = Image.new('L', (mask_w, mask_h), 255)
                    mask.paste(mask_draw, (x_start, y_start))
                    
                    # 执行inpainting
                    with torch.no_grad():
                        result = pipe(
                            prompt=params['prompt'],
                            image=img,
                            mask_image=mask,
                            num_inference_steps=28,
                            guidance_scale=30.0,
                            generator=torch.Generator("cuda").manual_seed(42),
                        ).images[0]
                    
                    # 保存
                    attacked_path = os.path.join(attack_dir, f"{img_id}.png")
                    result.save(attacked_path)
                    
                    # 清理
                    del result
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"\n   ❌ {img_id} 失败: {e}")
                    continue
    except Exception as e:
        print(f"\n❌ FluxFill模型加载失败: {e}")
        print("   跳过FluxFill攻击")
    finally:
        # 卸载FluxFill模型
        if 'pipe' in locals():
            print("\n🧹 卸载 FluxFill 模型...")
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            print(f"   当前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


# ==========================================
# 攻击配置
# ==========================================
ATTACK_CONFIGS = [
    # ========== 传统攻击（无需深度学习模型）==========
    # JPEG压缩
    {'name': 'jpeg_75', 'type': 'jpeg', 'params': {'quality': 75}},
    {'name': 'jpeg_50', 'type': 'jpeg', 'params': {'quality': 50}},
    {'name': 'jpeg_30', 'type': 'jpeg', 'params': {'quality': 30}},
    {'name': 'jpeg_20', 'type': 'jpeg', 'params': {'quality': 20}},
    
    # 高斯模糊
    {'name': 'blur_0.5', 'type': 'blur', 'params': {'sigma': 0.5, 'kernel_size': 5}},
    {'name': 'blur_1.0', 'type': 'blur', 'params': {'sigma': 1.0, 'kernel_size': 5}},
    {'name': 'blur_2.0', 'type': 'blur', 'params': {'sigma': 2.0, 'kernel_size': 9}},
    
    # 中心裁剪
    {'name': 'crop_0.75', 'type': 'crop', 'params': {'crop_ratio': 0.75}},
    {'name': 'crop_0.50', 'type': 'crop', 'params': {'crop_ratio': 0.50}},
    
    # 加噪声
    {'name': 'noise_0.03', 'type': 'noise', 'params': {'std': 0.03}},
    {'name': 'noise_0.05', 'type': 'noise', 'params': {'std': 0.05}},
    {'name': 'noise_0.10', 'type': 'noise', 'params': {'std': 0.10}},
    
    # 缩放
    {'name': 'resize_0.75', 'type': 'resize', 'params': {'scale': 0.75}},
    {'name': 'resize_0.50', 'type': 'resize', 'params': {'scale': 0.50}},
    {'name': 'resize_0.25', 'type': 'resize', 'params': {'scale': 0.25}},
    
    # 亮度/对比度
    {'name': 'brightness_0.8', 'type': 'brightness_contrast', 'params': {'brightness': 0.8, 'contrast': 1.0}},
    {'name': 'brightness_1.2', 'type': 'brightness_contrast', 'params': {'brightness': 1.2, 'contrast': 1.0}},
    {'name': 'contrast_0.8', 'type': 'brightness_contrast', 'params': {'brightness': 1.0, 'contrast': 0.8}},
    {'name': 'contrast_1.2', 'type': 'brightness_contrast', 'params': {'brightness': 1.0, 'contrast': 1.2}},
    
    # 黑色方块（篡改定位测试）
    {'name': 'black_block_center', 'type': 'black_block', 'params': {'block_size_ratio': 0.25, 'position': 'center'}},
    {'name': 'black_block_random', 'type': 'black_block', 'params': {'block_size_ratio': 0.25, 'position': 'random'}},
    {'name': 'black_block_corner', 'type': 'black_block', 'params': {'block_size_ratio': 0.20, 'position': 'corner_tl'}},
    
    # SDEdit（简化版扩散再生攻击）
    {'name': 'sdedit_0.3', 'type': 'sdedit', 'params': {'noise_strength': 0.3}},
    
    # ========== 深度学习攻击（分批执行）==========
    # 注：SDXL 和 FluxFill 攻击运行时间较长，暂时注释掉，可单独运行
    # SDXL I2I 语义修改攻击
    # {'name': 'sdxl_painting', 'type': 'sdxl_i2i', 'params': {'instruction': 'make it look like an oil painting'}},
    # {'name': 'sdxl_watercolor', 'type': 'sdxl_i2i', 'params': {'instruction': 'make it look like a watercolor painting'}},
    # {'name': 'sdxl_sketch', 'type': 'sdxl_i2i', 'params': {'instruction': 'make it look like a pencil sketch'}},
    
    # FluxFill Inpainting 攻击
    # {'name': 'fluxfill_center', 'type': 'flux_fill', 'params': {'prompt': 'a beautiful scenic view', 'mask_ratio': 0.3}},
]

# 分离攻击类型
TRADITIONAL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] not in ['sdxl_i2i', 'flux_fill']]
SDXL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] == 'sdxl_i2i']
FLUXFILL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] == 'flux_fill']

# ==========================================
# 主循环：生成攻击图像
# ==========================================
print(f"\n🔬 开始生成攻击图像...")
print(f"   传统攻击: {len(TRADITIONAL_ATTACKS)} 种")
print(f"   SDXL攻击: {len(SDXL_ATTACKS)} 种")
print(f"   FluxFill攻击: {len(FLUXFILL_ATTACKS)} 种")

# 设置随机种子
random.seed(config['prompt_seed'])
np.random.seed(config['prompt_seed'])

attack_summary = {
    'experiment_name': config['experiment_name'],
    'timestamp': datetime.now().isoformat(),
    'config': config,
    'attacks': [a['name'] for a in ATTACK_CONFIGS],
    'images': []
}

# ==========================================
# Phase 1: 传统攻击（无需深度学习模型）
# ==========================================
print("\n" + "="*60)
print("Phase 1: 传统攻击（无需深度学习模型）")
print("="*60)

for img_info in tqdm(images_info, desc="传统攻击进度"):
    img_id = img_info['id']
    prompt = img_info['prompt']
    
    # 加载原图
    img_path = os.path.join(watermarked_dir, img_info['image_file'])
    img = Image.open(img_path).convert('RGB')
    
    img_attacks = {
        'id': img_id,
        'prompt': prompt,
        'attacks': {}
    }
    
    # 应用每种传统攻击
    for attack_config in TRADITIONAL_ATTACKS:
        attack_name = attack_config['name']
        attack_type = attack_config['type']
        params = attack_config['params']
        
        # 创建攻击类型目录
        attack_dir = os.path.join(base_output_dir, attack_name)
        os.makedirs(attack_dir, exist_ok=True)
        
        # 执行攻击
        try:
            if attack_type == 'jpeg':
                attacked_img = attack_jpeg(img, params['quality'])
                save_ext = 'jpg'
            elif attack_type == 'blur':
                attacked_img = attack_blur(img, params['sigma'], params.get('kernel_size', 9))
                save_ext = 'png'
            elif attack_type == 'crop':
                attacked_img = attack_crop_center(img, params['crop_ratio'])
                save_ext = 'png'
            elif attack_type == 'noise':
                attacked_img = attack_noise(img, params['std'])
                save_ext = 'png'
            elif attack_type == 'resize':
                attacked_img = attack_resize(img, params['scale'])
                save_ext = 'png'
            elif attack_type == 'brightness_contrast':
                attacked_img = attack_brightness_contrast(img, params['brightness'], params['contrast'])
                save_ext = 'png'
            elif attack_type == 'black_block':
                attacked_img, true_mask = attack_black_block(img, params['block_size_ratio'], params['position'])
                mask_filename = f"{img_id}_mask.npy"
                mask_path = os.path.join(attack_dir, mask_filename)
                np.save(mask_path, true_mask)
                img_attacks['attacks'][attack_name] = {
                    'image_file': f"{img_id}.png",
                    'mask_file': mask_filename
                }
                save_ext = 'png'
            elif attack_type == 'sdedit':
                attacked_img = attack_sdedit_simple(img, params['noise_strength'])
                save_ext = 'png'
            else:
                continue
            
            # 保存攻击后的图像
            attacked_filename = f"{img_id}.{save_ext}"
            attacked_path = os.path.join(attack_dir, attacked_filename)
            attacked_img.save(attacked_path)
            
            if attack_type != 'black_block':
                img_attacks['attacks'][attack_name] = {'image_file': attacked_filename}
                
        except Exception as e:
            print(f"\n❌ 传统攻击 {attack_name} 失败: {e}")
            continue
    
    attack_summary['images'].append(img_attacks)

print(f"\n✅ 传统攻击完成")
print(f"   当前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# Phase 2: SDXL 攻击
# ==========================================
if SDXL_ATTACKS:
    print("\n" + "="*60)
    print("Phase 2: SDXL I2I 攻击")
    print("="*60)
    run_sdxl_attacks(images_info, ATTACK_CONFIGS)

# ==========================================
# Phase 3: FluxFill 攻击
# ==========================================
if FLUXFILL_ATTACKS:
    print("\n" + "="*60)
    print("Phase 3: FluxFill Inpainting 攻击")
    print("="*60)
    run_fluxfill_attacks(images_info, ATTACK_CONFIGS)

# ==========================================
# 保存攻击摘要
# ==========================================
summary_path = os.path.join(base_output_dir, f'{config["experiment_name"]}_attack_summary.json')
with open(summary_path, 'w') as f:
    json.dump(attack_summary, f, indent=2)

# 最终清理
gc.collect()
torch.cuda.empty_cache()

print(f"\n" + "="*60)
print("✅ 所有攻击图像生成完成！")
print("="*60)
print(f"   攻击图像保存位置: {base_output_dir}")
print(f"   攻击摘要: {summary_path}")
print(f"\n📊 攻击类型统计:")
for attack in ATTACK_CONFIGS:
    attack_dir = os.path.join(base_output_dir, attack['name'])
    if os.path.exists(attack_dir):
        num_files = len([f for f in os.listdir(attack_dir) if f.endswith(('.png', '.jpg'))])
        print(f"   - {attack['name']}: {num_files} 张")
print(f"\n   最终显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
