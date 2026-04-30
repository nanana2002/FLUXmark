#!/usr/bin/env python3
"""
公共攻击工具模块
从 attack_img_generate.py 提取并重构，供 baseline 等脚本复用
"""

import os
import io
import json
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F_vision
from tqdm import tqdm


# ==========================================
# 传统攻击函数定义（纯函数，无外部依赖）
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

    # 计算真实篡改掩码（32x32网格，与潜空间原生分辨率一致）
    true_mask = np.zeros((32, 32), dtype=bool)
    y_start_32 = int(y_start / h * 32)
    y_end_32 = int((y_start + block_h) / h * 32)
    x_start_32 = int(x_start / w * 32)
    x_end_32 = int((x_start + block_w) / w * 32)

    y_start_32 = max(0, min(31, y_start_32))
    y_end_32 = max(0, min(32, y_end_32))
    x_start_32 = max(0, min(31, x_start_32))
    x_end_32 = max(0, min(32, x_end_32))

    true_mask[y_start_32:y_end_32, x_start_32:x_end_32] = True

    return Image.fromarray(attacked), true_mask


def attack_copy_move(img, block_size_ratio=0.2):
    """Copy-Move 攻击：从图中复制一个块粘贴到另一位置"""
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    block_h = int(h * block_size_ratio)
    block_w = int(w * block_size_ratio)

    # 随机选择源块位置
    src_y = random.randint(0, max(0, h - block_h))
    src_x = random.randint(0, max(0, w - block_w))

    # 随机选择目标位置（确保不与源块完全重叠）
    for _ in range(10):
        dst_y = random.randint(0, max(0, h - block_h))
        dst_x = random.randint(0, max(0, w - block_w))
        if abs(dst_y - src_y) > block_h // 2 or abs(dst_x - src_x) > block_w // 2:
            break

    attacked = img_array.copy()
    attacked[dst_y:dst_y+block_h, dst_x:dst_x+block_w] = img_array[src_y:src_y+block_h, src_x:src_x+block_w]

    true_mask = np.zeros((32, 32), dtype=bool)
    dst_y_start_32 = int(dst_y / h * 32)
    dst_y_end_32 = int((dst_y + block_h) / h * 32)
    dst_x_start_32 = int(dst_x / w * 32)
    dst_x_end_32 = int((dst_x + block_w) / w * 32)
    dst_y_start_32 = max(0, min(31, dst_y_start_32))
    dst_y_end_32 = max(0, min(32, dst_y_end_32))
    dst_x_start_32 = max(0, min(31, dst_x_start_32))
    dst_x_end_32 = max(0, min(32, dst_x_end_32))
    true_mask[dst_y_start_32:dst_y_end_32, dst_x_start_32:dst_x_end_32] = True

    return Image.fromarray(attacked), true_mask


def attack_splicing(img, donor_img, block_size_ratio=0.2):
    """Splicing 攻击：从 donor 图像中取块粘贴到当前图像"""
    img_array = np.array(img)
    donor_array = np.array(donor_img.resize(img.size, Image.LANCZOS))
    h, w = img_array.shape[:2]
    block_h = int(h * block_size_ratio)
    block_w = int(w * block_size_ratio)

    # 随机选择源块位置（donor）
    src_y = random.randint(0, max(0, h - block_h))
    src_x = random.randint(0, max(0, w - block_w))

    # 随机选择目标位置
    dst_y = random.randint(0, max(0, h - block_h))
    dst_x = random.randint(0, max(0, w - block_w))

    attacked = img_array.copy()
    attacked[dst_y:dst_y+block_h, dst_x:dst_x+block_w] = donor_array[src_y:src_y+block_h, src_x:src_x+block_w]

    true_mask = np.zeros((32, 32), dtype=bool)
    dst_y_start_32 = int(dst_y / h * 32)
    dst_y_end_32 = int((dst_y + block_h) / h * 32)
    dst_x_start_32 = int(dst_x / w * 32)
    dst_x_end_32 = int((dst_x + block_w) / w * 32)
    dst_y_start_32 = max(0, min(31, dst_y_start_32))
    dst_y_end_32 = max(0, min(32, dst_y_end_32))
    dst_x_start_32 = max(0, min(31, dst_x_start_32))
    dst_x_end_32 = max(0, min(32, dst_x_end_32))
    true_mask[dst_y_start_32:dst_y_end_32, dst_x_start_32:dst_x_end_32] = True

    return Image.fromarray(attacked), true_mask


def attack_sdedit_simple(img, pipe, noise_strength=0.3, num_inference_steps=28):
    """
    真正的SDEdit攻击：使用 FluxImg2ImgPipeline 进行扩散再生
    """
    result = pipe(
        prompt="",
        image=img,
        strength=noise_strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    return result


# ==========================================
# 攻击配置（完整复刻 attack_img_generate.py）
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

    # 手工篡改（Copy-Move / Splicing）
    {'name': 'copy_move', 'type': 'copy_move', 'params': {'block_size_ratio': 0.2}},
    {'name': 'splicing', 'type': 'splicing', 'params': {'block_size_ratio': 0.2}},

    # SDEdit（扩散再生攻击）
    {'name': 'sdedit_0.3', 'type': 'sdedit', 'params': {'noise_strength': 0.3}},

    # ========== 深度学习攻击（高性能模式可同时加载）==========
    # SDXL I2I 语义修改攻击
    {'name': 'sdxl_style_oil_painting', 'type': 'sdxl_i2i', 'params': {'instruction': 'transform into an oil painting style', 'modification': 'oil_painting', 'use_modified_prompt': True}},
    {'name': 'sdxl_style_sketch', 'type': 'sdxl_i2i', 'params': {'instruction': 'convert to pencil sketch style', 'modification': 'sketch', 'use_modified_prompt': True}},

    # FluxFill Inpainting 攻击 - 局部重绘
    {'name': 'fluxfill_center', 'type': 'flux_fill', 'params': {'prompt': 'seamless continuation', 'mask_ratio': 0.3, 'modification': 'inpaint_center', 'use_modified_prompt': True}},
    {'name': 'fluxfill_random', 'type': 'flux_fill', 'params': {'prompt': 'seamless continuation', 'mask_ratio': 0.25, 'position': 'random', 'modification': 'inpaint_random', 'use_modified_prompt': True}},

    # ========== 崩溃边缘分析 (Breakdown Point Analysis) 极端攻击 ==========
    # JPEG 更极端压缩
    {'name': 'jpeg_20', 'type': 'jpeg', 'params': {'quality': 20}},
    {'name': 'jpeg_10', 'type': 'jpeg', 'params': {'quality': 10}},

    # 高斯模糊更极端
    {'name': 'blur_3.0', 'type': 'blur', 'params': {'sigma': 3.0, 'kernel_size': 9}},
    {'name': 'blur_4.0', 'type': 'blur', 'params': {'sigma': 4.0, 'kernel_size': 11}},

    # SDEdit 更极端
    {'name': 'sdedit_0.4', 'type': 'sdedit', 'params': {'noise_strength': 0.4}},
    {'name': 'sdedit_0.5', 'type': 'sdedit', 'params': {'noise_strength': 0.5}},

    # 噪声更极端
    {'name': 'noise_0.15', 'type': 'noise', 'params': {'std': 0.15}},
    {'name': 'noise_0.20', 'type': 'noise', 'params': {'std': 0.20}},

    # 缩放更极端
    {'name': 'resize_0.25', 'type': 'resize', 'params': {'scale': 0.25}},
]


# ==========================================
# 深度学习攻击器（重构为可复用，去除全局路径硬编码）
# ==========================================

class DeepLearningAttacker:
    """深度学习攻击器（逐个加载版本，避免OOM）"""

    def __init__(self, config, watermarked_dir, flux_pipe=None):
        self.config = config
        self.watermarked_dir = watermarked_dir
        self.sdxl_pipe = None
        self.fluxfill_pipe = None
        self.flux_pipe = flux_pipe  # 用于SDEdit攻击
        # 存储语义修改信息
        self.modified_prompts = {}

        # 模型路径
        self.SDXL_PATH = config.get('sdxl_path', '/data/daiyina/project_flux/model/sdxl-instructpix2pix')
        self.FLUX_FILL_PATH = config.get('flux_fill_path', '/data/daiyina/project_flux/model/flux-fill')

        # 检查模型路径是否存在
        self.SDXL_AVAILABLE = os.path.exists(self.SDXL_PATH) and os.path.isdir(self.SDXL_PATH) and len(os.listdir(self.SDXL_PATH)) > 0
        self.FLUX_FILL_AVAILABLE = os.path.exists(self.FLUX_FILL_PATH) and os.path.isdir(self.FLUX_FILL_PATH) and len(os.listdir(self.FLUX_FILL_PATH)) > 0

        if not self.SDXL_AVAILABLE:
            print(f"⚠️  SDXL模型不存在或为空: {self.SDXL_PATH}")
            print("   将跳过SDXL攻击")
        if not self.FLUX_FILL_AVAILABLE:
            print(f"⚠️  FluxFill模型不存在或为空: {self.FLUX_FILL_PATH}")
            print("   将跳过FluxFill攻击")

        # 加载 prompts_modified.json（如果存在）
        self.PROMPTS_MODIFIED = None
        PROMPTS_MODIFIED_PATH = os.path.join(config.get('output_base_dir', '.'), 'prompts_modified.json')
        if os.path.exists(PROMPTS_MODIFIED_PATH):
            with open(PROMPTS_MODIFIED_PATH, 'r') as f:
                self.PROMPTS_MODIFIED = json.load(f)
            print(f"📋 已加载 prompts_modified.json")
        else:
            print(f"⚠️  未找到 prompts_modified.json，将使用默认风格迁移")
            print(f"   预期路径: {PROMPTS_MODIFIED_PATH}")

    def _load_sdxl(self):
        """加载SDXL模型"""
        if self.sdxl_pipe is None and self.SDXL_AVAILABLE:
            print("\n🚀 加载 SDXL InstructPix2Pix 模型...")
            from diffusers import StableDiffusionXLInstructPix2PixPipeline
            self.sdxl_pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                self.SDXL_PATH,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            self.sdxl_pipe = self.sdxl_pipe.to("cuda")
            self.sdxl_pipe.enable_attention_slicing()
            print("   SDXL模型已加载到GPU")
            print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def _unload_sdxl(self):
        """卸载SDXL模型释放显存"""
        if self.sdxl_pipe is not None:
            print("\n🧹 卸载 SDXL 模型释放显存...")
            del self.sdxl_pipe
            self.sdxl_pipe = None
            torch.cuda.empty_cache()
            print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def _load_fluxfill(self):
        """加载FluxFill模型"""
        if self.fluxfill_pipe is None and self.FLUX_FILL_AVAILABLE:
            print("\n🚀 加载 FluxFill 模型...")
            from diffusers import FluxFillPipeline
            self.fluxfill_pipe = FluxFillPipeline.from_pretrained(
                self.FLUX_FILL_PATH,
                torch_dtype=torch.bfloat16,
            )
            self.fluxfill_pipe = self.fluxfill_pipe.to("cuda")
            self.fluxfill_pipe.enable_vae_slicing()
            self.fluxfill_pipe.enable_attention_slicing()
            print("   FluxFill模型已加载到GPU")
            print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def _unload_fluxfill(self):
        """卸载FluxFill模型释放显存"""
        if self.fluxfill_pipe is not None:
            print("\n🧹 卸载 FluxFill 模型释放显存...")
            del self.fluxfill_pipe
            self.fluxfill_pipe = None
            torch.cuda.empty_cache()
            print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def sdxl_attack_batch(self, images_info, attack_config, attack_dir):
        """批量SDXL攻击 - 语义修改攻击"""
        attack_name = attack_config['name']
        params = attack_config['params']
        os.makedirs(attack_dir, exist_ok=True)

        # 加载SDXL模型
        self._load_sdxl()
        if self.sdxl_pipe is None:
            print(f"   ⚠️  SDXL模型不可用，跳过 {attack_name}")
            return

        print(f"\n   执行攻击: {attack_name}")
        print(f"   风格指令: {params['instruction']}")

        for img_info in tqdm(images_info, desc=f"   {attack_name}", leave=False):
            img_id = img_info['id']
            original_prompt = img_info['prompt']
            img_path = os.path.join(self.watermarked_dir, img_info['image_file'])
            output_path = os.path.join(attack_dir, f"{img_id}.png")

            if os.path.exists(output_path):
                # 记录已存在的修改prompt
                modified_prompt = f"[{params.get('modification', attack_name)}] {original_prompt}"
                if img_id not in self.modified_prompts:
                    self.modified_prompts[img_id] = {}
                self.modified_prompts[img_id][attack_name] = {
                    'original_prompt': original_prompt,
                    'modified_prompt': modified_prompt,
                    'instruction': params['instruction'],
                    'modification_type': params.get('modification', attack_name)
                }
                continue

            # 获取修改后的 prompt
            modified_prompt_for_attack = None
            if params.get('use_modified_prompt', False) and self.PROMPTS_MODIFIED:
                img_modified = self.PROMPTS_MODIFIED.get('modified_prompts', {}).get(img_id, {})
                for attack_key, mod_info in img_modified.items():
                    if mod_info.get('modification_type') == params.get('modification'):
                        modified_prompt_for_attack = mod_info.get('modified_prompt', original_prompt)
                        break

            try:
                img = Image.open(img_path).convert('RGB')

                with torch.no_grad():
                    # SDXL InstructPix2Pix 使用 instruction 进行风格迁移
                    result = self.sdxl_pipe(
                        prompt=params['instruction'],
                        image=img,
                        num_inference_steps=20,
                        guidance_scale=7.5,
                        image_guidance_scale=1.5,
                        generator=torch.Generator("cuda").manual_seed(42),
                    ).images[0]

                result.save(output_path)

                # 构建最终使用的 modified_prompt
                if modified_prompt_for_attack:
                    final_modified_prompt = f"[{params.get('modification', attack_name)}] {modified_prompt_for_attack}"
                else:
                    final_modified_prompt = f"[{params.get('modification', attack_name)}] {original_prompt} (style: {params['instruction']})"

                if img_id not in self.modified_prompts:
                    self.modified_prompts[img_id] = {}
                self.modified_prompts[img_id][attack_name] = {
                    'original_prompt': original_prompt,
                    'modified_prompt': final_modified_prompt,
                    'instruction': params['instruction'],
                    'modification_type': params.get('modification', attack_name),
                    'source': 'prompts_modified.json' if modified_prompt_for_attack else 'default'
                }

            except Exception as e:
                print(f"\n   ❌ {img_id} 失败: {e}")
                continue

        # 所有SDXL攻击完成后卸载模型
        self._unload_sdxl()

    def sdedit_attack_batch(self, images_info, attack_config, attack_dir):
        """批量SDEdit攻击 - 真正的扩散再生攻击"""
        attack_name = attack_config['name']
        params = attack_config['params']
        os.makedirs(attack_dir, exist_ok=True)

        # 检查FLUX pipe是否已加载
        if self.flux_pipe is None:
            print(f"   ⚠️  FLUX pipe未加载，跳过 {attack_name}")
            return

        print(f"\n   执行攻击: {attack_name}")
        print(f"   噪声强度: {params['noise_strength']}")
        print(f"   使用FLUX进行真正的扩散再生...")

        for img_info in tqdm(images_info, desc=f"   {attack_name}", leave=False):
            img_id = img_info['id']
            original_prompt = img_info['prompt']
            img_path = os.path.join(self.watermarked_dir, img_info['image_file'])
            output_path = os.path.join(attack_dir, f"{img_id}.png")

            if os.path.exists(output_path):
                # 记录已存在的修改prompt
                modified_prompt = f"[sdedit_regeneration] {original_prompt} (regenerated with noise strength {params['noise_strength']})"
                if img_id not in self.modified_prompts:
                    self.modified_prompts[img_id] = {}
                self.modified_prompts[img_id][attack_name] = {
                    'original_prompt': original_prompt,
                    'modified_prompt': modified_prompt,
                    'modification_type': 'sdedit_regeneration',
                    'noise_strength': params['noise_strength']
                }
                continue

            try:
                img = Image.open(img_path).convert('RGB')

                # 使用真正的SDEdit攻击（扩散再生）
                attacked_img = attack_sdedit_simple(
                    img, self.flux_pipe,
                    noise_strength=params['noise_strength'],
                    num_inference_steps=28
                )
                attacked_img.save(output_path)

                # 记录修改后的prompt
                modified_prompt = f"[sdedit_regeneration] {original_prompt} (regenerated with noise strength {params['noise_strength']})"
                if img_id not in self.modified_prompts:
                    self.modified_prompts[img_id] = {}
                self.modified_prompts[img_id][attack_name] = {
                    'original_prompt': original_prompt,
                    'modified_prompt': modified_prompt,
                    'modification_type': 'sdedit_regeneration',
                    'noise_strength': params['noise_strength']
                }

            except Exception as e:
                print(f"\n   ❌ {img_id} 失败: {e}")
                continue

    def fluxfill_attack_batch(self, images_info, attack_config, attack_dir):
        """批量FluxFill攻击 - 局部重绘攻击"""
        attack_name = attack_config['name']
        params = attack_config['params']
        os.makedirs(attack_dir, exist_ok=True)

        # 加载FluxFill模型
        self._load_fluxfill()
        if self.fluxfill_pipe is None:
            print(f"   ⚠️  FluxFill模型不可用，跳过 {attack_name}")
            return

        print(f"\n   执行攻击: {attack_name}")
        print(f"   重绘prompt: {params['prompt']}")

        for img_info in tqdm(images_info, desc=f"   {attack_name}", leave=False):
            img_id = img_info['id']
            original_prompt = img_info['prompt']
            img_path = os.path.join(self.watermarked_dir, img_info['image_file'])
            output_path = os.path.join(attack_dir, f"{img_id}.png")
            mask_path = os.path.join(attack_dir, f"{img_id}_mask.npy")

            if os.path.exists(output_path):
                # 记录已存在的修改prompt
                modified_prompt = f"[{params.get('modification', attack_name)}] {original_prompt} (inpainted: {params['prompt']})"
                if img_id not in self.modified_prompts:
                    self.modified_prompts[img_id] = {}
                self.modified_prompts[img_id][attack_name] = {
                    'original_prompt': original_prompt,
                    'modified_prompt': modified_prompt,
                    'inpaint_prompt': params['prompt'],
                    'modification_type': params.get('modification', attack_name),
                    'mask_ratio': params.get('mask_ratio', 0.3)
                }
                continue

            # 获取修改后的 inpaint prompt
            inpaint_prompt = params['prompt']
            if params.get('use_modified_prompt', False) and self.PROMPTS_MODIFIED:
                img_modified = self.PROMPTS_MODIFIED.get('modified_prompts', {}).get(img_id, {})
                found_prompt = False
                for attack_key, mod_info in img_modified.items():
                    if mod_info.get('modification_type') == params.get('modification'):
                        inpaint_prompt = mod_info.get('inpaint_prompt', params['prompt'])
                        found_prompt = True
                        break
                # fallback: fluxfill_random 若找不到 inpaint_random，则复用 inpaint_center 的 prompt
                if not found_prompt and params.get('modification') == 'inpaint_random':
                    for attack_key, mod_info in img_modified.items():
                        if mod_info.get('modification_type') == 'inpaint_center':
                            inpaint_prompt = mod_info.get('inpaint_prompt', params['prompt'])
                            break

            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size

                # 创建mask
                mask_ratio = params.get('mask_ratio', 0.3)
                position = params.get('position', 'center')
                mask_w = int(w * mask_ratio)
                mask_h = int(h * mask_ratio)

                if position == 'center':
                    x_start = (w - mask_w) // 2
                    y_start = (h - mask_h) // 2
                elif position == 'random':
                    x_start = random.randint(0, max(0, w - mask_w))
                    y_start = random.randint(0, max(0, h - mask_h))
                else:
                    x_start = (w - mask_w) // 2
                    y_start = (h - mask_h) // 2

                mask = Image.new('L', (w, h), 0)
                mask_draw = Image.new('L', (mask_w, mask_h), 255)
                mask.paste(mask_draw, (x_start, y_start))

                # 计算并保存32x32篡改掩码
                true_mask = np.zeros((32, 32), dtype=bool)
                y_start_32 = int(y_start / h * 32)
                y_end_32 = int((y_start + mask_h) / h * 32)
                x_start_32 = int(x_start / w * 32)
                x_end_32 = int((x_start + mask_w) / w * 32)
                y_start_32 = max(0, min(31, y_start_32))
                y_end_32 = max(0, min(32, y_end_32))
                x_start_32 = max(0, min(31, x_start_32))
                x_end_32 = max(0, min(32, x_end_32))
                true_mask[y_start_32:y_end_32, x_start_32:x_end_32] = True
                np.save(mask_path, true_mask)

                with torch.no_grad():
                    result = self.fluxfill_pipe(
                        prompt=inpaint_prompt,
                        image=img,
                        mask_image=mask,
                        num_inference_steps=28,
                        guidance_scale=30.0,
                        generator=torch.Generator("cuda").manual_seed(42),
                    ).images[0]

                result.save(output_path)

                # 记录修改后的prompt
                final_modified_prompt = f"[{params.get('modification', attack_name)}] {original_prompt} (inpainted: {inpaint_prompt})"
                if img_id not in self.modified_prompts:
                    self.modified_prompts[img_id] = {}
                self.modified_prompts[img_id][attack_name] = {
                    'original_prompt': original_prompt,
                    'modified_prompt': final_modified_prompt,
                    'inpaint_prompt': inpaint_prompt,
                    'modification_type': params.get('modification', attack_name),
                    'mask_ratio': mask_ratio,
                    'mask_position': position,
                    'source': 'prompts_modified.json' if inpaint_prompt != params['prompt'] else 'default'
                }

            except Exception as e:
                print(f"\n   ❌ {img_id} 失败: {e}")
                continue

        # 所有FluxFill攻击完成后卸载模型
        self._unload_fluxfill()
