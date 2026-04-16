#!/usr/bin/env python3
"""
攻击图像生成脚本（极致性能版 - 50GB显存全速运行）
对水印图像进行多种攻击，生成被攻击后的图像

优化策略：
- SDXL和FluxFill可同时加载常驻GPU
- 批量处理攻击图像
- 并行执行传统攻击
- 不调用显存清理，最大化利用显存换速度
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

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
# 攻击函数定义（传统攻击）
# ==========================================

def attack_jpeg(img, quality):
    """JPEG压缩攻击"""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def attack_blur(img, sigma, kernel_size=9):
    """高斯模糊攻击"""
    tensor = T.ToTensor()(img).unsqueeze(0).to('cuda')
    blurred = F_vision.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return T.ToPILImage()(blurred.squeeze(0).cpu())


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
    tensor = T.ToTensor()(img).unsqueeze(0).to('cuda')
    noise = torch.randn_like(tensor) * std
    noisy = torch.clamp(tensor + noise, 0, 1)
    return T.ToPILImage()(noisy.squeeze(0).cpu())


def attack_resize(img, scale=0.5):
    """缩放攻击"""
    w, h = img.size
    small = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    return small.resize((w, h), Image.Resampling.LANCZOS)


def attack_brightness_contrast(img, brightness=1.0, contrast=1.0):
    """亮度/对比度调整攻击"""
    tensor = T.ToTensor()(img).unsqueeze(0).to('cuda')
    tensor = torch.clamp(tensor * brightness, 0, 1)
    mean = tensor.mean()
    tensor = torch.clamp((tensor - mean) * contrast + mean, 0, 1)
    return T.ToPILImage()(tensor.squeeze(0).cpu())


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
    img_tensor = T.ToTensor()(img).unsqueeze(0).to('cuda')
    noise = torch.randn_like(img_tensor) * noise_strength
    noisy = torch.clamp(img_tensor + noise, 0, 1)
    return T.ToPILImage()(noisy.squeeze(0).cpu())


# ==========================================
# 深度学习攻击（高性能版本）
# ==========================================

class DeepLearningAttacker:
    """深度学习攻击器（高性能版，模型常驻GPU）"""

    def __init__(self):
        self.sdxl_pipe = None
        self.fluxfill_pipe = None
        self._load_models()

    def _load_models(self):
        """加载所有深度学习模型到GPU"""
        # 加载SDXL
        if SDXL_AVAILABLE:
            print("\n🚀 加载 SDXL InstructPix2Pix 模型...")
            self.sdxl_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                SDXL_PATH,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            self.sdxl_pipe = self.sdxl_pipe.to("cuda")
            self.sdxl_pipe.enable_attention_slicing()
            print("   SDXL模型已加载到GPU")

        # 加载FluxFill
        if FLUX_FILL_AVAILABLE:
            print("\n🚀 加载 FluxFill 模型...")
            self.fluxfill_pipe = FluxFillPipeline.from_pretrained(
                FLUX_FILL_PATH,
                torch_dtype=torch.bfloat16,
            )
            # 50GB显存可以同时加载SDXL和FluxFill
            self.fluxfill_pipe = self.fluxfill_pipe.to("cuda")
            self.fluxfill_pipe.enable_vae_slicing()
            self.fluxfill_pipe.enable_attention_slicing()
            print("   FluxFill模型已加载到GPU")

        print(f"\n   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def sdxl_attack_batch(self, images_info, attack_config, attack_dir):
        """批量SDXL攻击"""
        attack_name = attack_config['name']
        params = attack_config['params']
        os.makedirs(attack_dir, exist_ok=True)

        print(f"\n   执行攻击: {attack_name}")

        for img_info in tqdm(images_info, desc=f"   {attack_name}", leave=False):
            img_id = img_info['id']
            img_path = os.path.join(watermarked_dir, img_info['image_file'])
            output_path = os.path.join(attack_dir, f"{img_id}.png")

            if os.path.exists(output_path):
                continue

            try:
                img = Image.open(img_path).convert('RGB')

                with torch.no_grad():
                    result = self.sdxl_pipe(
                        prompt=params['instruction'],
                        image=img,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        generator=torch.Generator("cuda").manual_seed(42),
                    ).images[0]

                result.save(output_path)

            except Exception as e:
                print(f"\n   ❌ {img_id} 失败: {e}")
                continue

    def fluxfill_attack_batch(self, images_info, attack_config, attack_dir):
        """批量FluxFill攻击"""
        attack_name = attack_config['name']
        params = attack_config['params']
        os.makedirs(attack_dir, exist_ok=True)

        print(f"\n   执行攻击: {attack_name}")

        for img_info in tqdm(images_info, desc=f"   {attack_name}", leave=False):
            img_id = img_info['id']
            img_path = os.path.join(watermarked_dir, img_info['image_file'])
            output_path = os.path.join(attack_dir, f"{img_id}.png")

            if os.path.exists(output_path):
                continue

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

                with torch.no_grad():
                    result = self.fluxfill_pipe(
                        prompt=params['prompt'],
                        image=img,
                        mask_image=mask,
                        num_inference_steps=28,
                        guidance_scale=30.0,
                        generator=torch.Generator("cuda").manual_seed(42),
                    ).images[0]

                result.save(output_path)

            except Exception as e:
                print(f"\n   ❌ {img_id} 失败: {e}")
                continue


# ==========================================
# 攻击配置
# ==========================================
ATTACK_CONFIGS = [
    # ========== 传统攻击（无需深度学习模型）==========
    # JPEG压缩
    {'name': 'jpeg_75', 'type': 'jpeg', 'params': {'quality': 75}},
    {'name': 'jpeg_50', 'type': 'jpeg', 'params': {'quality': 50}},
    {'name': 'jpeg_30', 'type': 'jpeg', 'params': {'quality': 30}},

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

    # 亮度/对比度
    {'name': 'brightness_0.8', 'type': 'brightness_contrast', 'params': {'brightness': 0.8, 'contrast': 1.0}},
    {'name': 'brightness_1.2', 'type': 'brightness_contrast', 'params': {'brightness': 1.2, 'contrast': 1.0}},
    {'name': 'contrast_0.8', 'type': 'brightness_contrast', 'params': {'brightness': 1.0, 'contrast': 0.8}},
    {'name': 'contrast_1.2', 'type': 'brightness_contrast', 'params': {'brightness': 1.0, 'contrast': 1.2}},

    # 黑色方块（篡改定位测试）
    {'name': 'black_block_center', 'type': 'black_block', 'params': {'block_size_ratio': 0.25, 'position': 'center'}},
    {'name': 'black_block_random', 'type': 'black_block', 'params': {'block_size_ratio': 0.25, 'position': 'random'}},

    # SDEdit（简化版扩散再生攻击）
    {'name': 'sdedit_0.3', 'type': 'sdedit', 'params': {'noise_strength': 0.3}},

    # ========== 深度学习攻击（高性能模式可同时加载）==========
    # SDXL I2I 语义修改攻击
    # {'name': 'sdxl_painting', 'type': 'sdxl_i2i', 'params': {'instruction': 'make it look like an oil painting'}},

    # FluxFill Inpainting 攻击
    # {'name': 'fluxfill_center', 'type': 'flux_fill', 'params': {'prompt': 'a beautiful scenic view', 'mask_ratio': 0.3}},
]

# 分离攻击类型
TRADITIONAL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] not in ['sdxl_i2i', 'flux_fill']]
SDXL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] == 'sdxl_i2i']
FLUXFILL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] == 'flux_fill']

# ==========================================
# 主循环：生成攻击图像（高性能模式）
# ==========================================
print(f"\n🔬 开始生成攻击图像（高性能模式）...")
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
# Phase 1: 传统攻击（并行执行）
# ==========================================
def process_traditional_attacks_for_image(img_info):
    """处理单个图像的所有传统攻击"""
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

        # 检查是否已存在
        output_path = os.path.join(attack_dir, f"{img_id}.png")
        if os.path.exists(output_path):
            if attack_type == 'black_block':
                mask_path = os.path.join(attack_dir, f"{img_id}_mask.npy")
                if os.path.exists(mask_path):
                    img_attacks['attacks'][attack_name] = {
                        'image_file': f"{img_id}.png",
                        'mask_file': f"{img_id}_mask.npy"
                    }
                else:
                    img_attacks['attacks'][attack_name] = {'image_file': f"{img_id}.png"}
            else:
                img_attacks['attacks'][attack_name] = {'image_file': f"{img_id}.png"}
            continue

        # 执行攻击
        try:
            if attack_type == 'jpeg':
                attacked_img = attack_jpeg(img, params['quality'])
                save_ext = 'jpg'
                attacked_path = os.path.join(attack_dir, f"{img_id}.{save_ext}")
                attacked_img.save(attacked_path)
                img_attacks['attacks'][attack_name] = {'image_file': f"{img_id}.{save_ext}"}

            elif attack_type == 'blur':
                attacked_img = attack_blur(img, params['sigma'], params.get('kernel_size', 9))
                save_ext = 'png'
                attacked_path = os.path.join(attack_dir, f"{img_id}.{save_ext}")
                attacked_img.save(attacked_path)
                img_attacks['attacks'][attack_name] = {'image_file': f"{img_id}.{save_ext}"}

            elif attack_type == 'crop':
                attacked_img = attack_crop_center(img, params['crop_ratio'])
                save_ext = 'png'
                attacked_path = os.path.join(attack_dir, f"{img_id}.{save_ext}")
                attacked_img.save(attacked_path)
                img_attacks['attacks'][attack_name] = {'image_file': f"{img_id}.{save_ext}"}

            elif attack_type == 'noise':
                attacked_img = attack_noise(img, params['std'])
                save_ext = 'png'
                attacked_path = os.path.join(attack_dir, f"{img_id}.{save_ext}")
                attacked_img.save(attacked_path)
                img_attacks['attacks'][attack_name] = {'image_file': f"{img_id}.{save_ext}"}

            elif attack_type == 'resize':
                attacked_img = attack_resize(img, params['scale'])
                save_ext = 'png'
                attacked_path = os.path.join(attack_dir, f"{img_id}.{save_ext}")
                attacked_img.save(attacked_path)
                img_attacks['attacks'][attack_name] = {'image_file': f"{img_id}.{save_ext}"}

            elif attack_type == 'brightness_contrast':
                attacked_img = attack_brightness_contrast(img, params['brightness'], params['contrast'])
                save_ext = 'png'
                attacked_path = os.path.join(attack_dir, f"{img_id}.{save_ext}")
                attacked_img.save(attacked_path)
                img_attacks['attacks'][attack_name] = {'image_file': f"{img_id}.{save_ext}"}

            elif attack_type == 'black_block':
                attacked_img, true_mask = attack_black_block(img, params['block_size_ratio'], params['position'])
                mask_filename = f"{img_id}_mask.npy"
                mask_path = os.path.join(attack_dir, mask_filename)
                np.save(mask_path, true_mask)
                img_attacks['attacks'][attack_name] = {
                    'image_file': f"{img_id}.png",
                    'mask_file': mask_filename
                }
                attacked_path = os.path.join(attack_dir, f"{img_id}.png")
                attacked_img.save(attacked_path)

            elif attack_type == 'sdedit':
                attacked_img = attack_sdedit_simple(img, params['noise_strength'])
                save_ext = 'png'
                attacked_path = os.path.join(attack_dir, f"{img_id}.{save_ext}")
                attacked_img.save(attacked_path)
                img_attacks['attacks'][attack_name] = {'image_file': f"{img_id}.{save_ext}"}

        except Exception as e:
            print(f"\n❌ 传统攻击 {attack_name} 失败: {e}")
            continue

    return img_attacks


print("\n" + "="*60)
print("Phase 1: 传统攻击（高性能并行模式）")
print("="*60)

# 使用多线程并行处理传统攻击
num_workers = config.get('num_workers', 4)
print(f"   使用 {num_workers} 个并行工作线程")

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    results = list(tqdm(
        executor.map(process_traditional_attacks_for_image, images_info),
        total=len(images_info),
        desc="传统攻击进度"
    ))

attack_summary['images'] = results

print(f"\n✅ 传统攻击完成")
print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ==========================================
# Phase 2: 深度学习攻击（模型常驻GPU）
# ==========================================
if SDXL_ATTACKS or FLUXFILL_ATTACKS:
    print("\n" + "="*60)
    print("Phase 2: 深度学习攻击（模型常驻GPU）")
    print("="*60)

    # 初始化攻击器（同时加载所有深度学习模型）
    attacker = DeepLearningAttacker()

    # SDXL攻击
    for attack_config in SDXL_ATTACKS:
        attack_dir = os.path.join(base_output_dir, attack_config['name'])
        attacker.sdxl_attack_batch(images_info, attack_config, attack_dir)

    # FluxFill攻击
    for attack_config in FLUXFILL_ATTACKS:
        attack_dir = os.path.join(base_output_dir, attack_config['name'])
        attacker.fluxfill_attack_batch(images_info, attack_config, attack_dir)

# ==========================================
# 保存攻击摘要
# ==========================================
summary_path = os.path.join(base_output_dir, f'{config["experiment_name"]}_attack_summary.json')
with open(summary_path, 'w') as f:
    json.dump(attack_summary, f, indent=2)

print(f"\n" + "="*60)
print("✅ 所有攻击图像生成完成（高性能模式）！")
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
print(f"   🚀 高性能模式：SDXL和FluxFill可同时常驻GPU！")
