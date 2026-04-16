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

import os
import json
import io

# 先读取配置设置 GPU（必须在 import torch 之前）
with open('config.json', 'r') as f:
    config = json.load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_id', 0))
os.environ['HF_HOME'] = config['hf_cache']

import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline, FluxFillPipeline
import torchvision.transforms as T
from torchvision.transforms import functional as F_vision
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

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

print(f"🎮 使用 GPU: {config.get('gpu_id', 0)}")

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

    true_mask = np.zeros((8, 8), dtype=bool)
    dst_y_start_8 = int(dst_y / h * 8)
    dst_y_end_8 = int((dst_y + block_h) / h * 8)
    dst_x_start_8 = int(dst_x / w * 8)
    dst_x_end_8 = int((dst_x + block_w) / w * 8)
    dst_y_start_8 = max(0, min(7, dst_y_start_8))
    dst_y_end_8 = max(0, min(8, dst_y_end_8))
    dst_x_start_8 = max(0, min(7, dst_x_start_8))
    dst_x_end_8 = max(0, min(8, dst_x_end_8))
    true_mask[dst_y_start_8:dst_y_end_8, dst_x_start_8:dst_x_end_8] = True

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

    true_mask = np.zeros((8, 8), dtype=bool)
    dst_y_start_8 = int(dst_y / h * 8)
    dst_y_end_8 = int((dst_y + block_h) / h * 8)
    dst_x_start_8 = int(dst_x / w * 8)
    dst_x_end_8 = int((dst_x + block_w) / w * 8)
    dst_y_start_8 = max(0, min(7, dst_y_start_8))
    dst_y_end_8 = max(0, min(8, dst_y_end_8))
    dst_x_start_8 = max(0, min(7, dst_x_start_8))
    dst_x_end_8 = max(0, min(8, dst_x_end_8))
    true_mask[dst_y_start_8:dst_y_end_8, dst_x_start_8:dst_x_end_8] = True

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
# 深度学习攻击（高性能版本）
# ==========================================

class DeepLearningAttacker:
    """深度学习攻击器（逐个加载版本，避免OOM）"""

    def __init__(self, flux_pipe=None):
        self.sdxl_pipe = None
        self.fluxfill_pipe = None
        self.flux_pipe = flux_pipe  # 用于SDEdit攻击
        # 存储语义修改信息
        self.modified_prompts = {}

    def _load_sdxl(self):
        """加载SDXL模型"""
        if self.sdxl_pipe is None and SDXL_AVAILABLE:
            print("\n🚀 加载 SDXL InstructPix2Pix 模型...")
            self.sdxl_pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
                SDXL_PATH,
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
        if self.fluxfill_pipe is None and FLUX_FILL_AVAILABLE:
            print("\n🚀 加载 FluxFill 模型...")
            self.fluxfill_pipe = FluxFillPipeline.from_pretrained(
                FLUX_FILL_PATH,
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
        """批量SDXL攻击 - 语义修改攻击（支持 prompts_modified.json）"""
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
            img_path = os.path.join(watermarked_dir, img_info['image_file'])
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

            # 获取修改后的 prompt（如果 prompts_modified.json 存在且配置要求使用）
            modified_prompt_for_attack = None
            if params.get('use_modified_prompt', False) and PROMPTS_MODIFIED:
                # 尝试从 prompts_modified.json 获取对应图像的修改后 prompt
                img_modified = PROMPTS_MODIFIED.get('modified_prompts', {}).get(img_id, {})
                # 查找对应修改类型的 prompt
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
            img_path = os.path.join(watermarked_dir, img_info['image_file'])
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
        """批量FluxFill攻击 - 局部重绘攻击（支持 prompts_modified.json）"""
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
            img_path = os.path.join(watermarked_dir, img_info['image_file'])
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

            # 获取修改后的 inpaint prompt（如果 prompts_modified.json 存在且配置要求使用）
            inpaint_prompt = params['prompt']
            if params.get('use_modified_prompt', False) and PROMPTS_MODIFIED:
                img_modified = PROMPTS_MODIFIED.get('modified_prompts', {}).get(img_id, {})
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

                # 计算并保存8x8篡改掩码
                true_mask = np.zeros((8, 8), dtype=bool)
                y_start_8 = int(y_start / h * 8)
                y_end_8 = int((y_start + mask_h) / h * 8)
                x_start_8 = int(x_start / w * 8)
                x_end_8 = int((x_start + mask_w) / w * 8)
                y_start_8 = max(0, min(7, y_start_8))
                y_end_8 = max(0, min(8, y_end_8))
                x_start_8 = max(0, min(7, x_start_8))
                x_end_8 = max(0, min(8, x_end_8))
                true_mask[y_start_8:y_end_8, x_start_8:x_end_8] = True
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


# ==========================================
# 加载 prompts_modified.json（如果存在）
# ==========================================
PROMPTS_MODIFIED_PATH = os.path.join(config['output_base_dir'], 'prompts_modified.json')
PROMPTS_MODIFIED = None
if os.path.exists(PROMPTS_MODIFIED_PATH):
    with open(PROMPTS_MODIFIED_PATH, 'r') as f:
        PROMPTS_MODIFIED = json.load(f)
    print(f"📋 已加载 prompts_modified.json")
else:
    print(f"⚠️  未找到 prompts_modified.json，将使用默认风格迁移")
    print(f"   预期路径: {PROMPTS_MODIFIED_PATH}")

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

    # 手工篡改（Copy-Move / Splicing）
    {'name': 'copy_move', 'type': 'copy_move', 'params': {'block_size_ratio': 0.2}},
    {'name': 'splicing', 'type': 'splicing', 'params': {'block_size_ratio': 0.2}},

    # SDEdit（扩散再生攻击）
    {'name': 'sdedit_0.3', 'type': 'sdedit', 'params': {'noise_strength': 0.3}},

    # ========== 深度学习攻击（高性能模式可同时加载）==========
    # SDXL I2I 语义修改攻击 - 使用 prompts_modified.json 中的修改后 prompt
    # 同时添加全局风格迁移
    {'name': 'sdxl_style_oil_painting', 'type': 'sdxl_i2i', 'params': {'instruction': 'transform into an oil painting style', 'modification': 'oil_painting', 'use_modified_prompt': True}},
    {'name': 'sdxl_style_sketch', 'type': 'sdxl_i2i', 'params': {'instruction': 'convert to pencil sketch style', 'modification': 'sketch', 'use_modified_prompt': True}},
    {'name': 'sdxl_style_watercolor', 'type': 'sdxl_i2i', 'params': {'instruction': 'transform into watercolor painting', 'modification': 'watercolor', 'use_modified_prompt': True}},
    {'name': 'sdxl_style_cyberpunk', 'type': 'sdxl_i2i', 'params': {'instruction': 'convert to cyberpunk neon style', 'modification': 'cyberpunk', 'use_modified_prompt': True}},
    {'name': 'sdxl_style_anime', 'type': 'sdxl_i2i', 'params': {'instruction': 'transform into anime style', 'modification': 'anime', 'use_modified_prompt': True}},

    # FluxFill Inpainting 攻击 - 局部重绘
    {'name': 'fluxfill_center', 'type': 'flux_fill', 'params': {'prompt': 'seamless continuation', 'mask_ratio': 0.3, 'modification': 'inpaint_center', 'use_modified_prompt': True}},
    {'name': 'fluxfill_random', 'type': 'flux_fill', 'params': {'prompt': 'seamless continuation', 'mask_ratio': 0.25, 'position': 'random', 'modification': 'inpaint_random', 'use_modified_prompt': True}},
]

# 分离攻击类型
TRADITIONAL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] not in ['sdxl_i2i', 'flux_fill', 'sdedit']]
SDXL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] == 'sdxl_i2i']
FLUXFILL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] == 'flux_fill']
SDEDIT_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] == 'sdedit']

num_samples = len(images_info)

def count_images_in_dir(dir_path):
    """统计目录中已生成的图片数量"""
    if not os.path.exists(dir_path):
        return 0
    return len([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

def filter_completed_attacks(attack_list, phase_name):
    """过滤掉已经完成的攻击类型"""
    remaining = []
    for cfg in attack_list:
        attack_dir = os.path.join(base_output_dir, cfg['name'])
        existing = count_images_in_dir(attack_dir)
        if existing >= num_samples:
            print(f"   ⏭️  {phase_name} 跳过已完成: {cfg['name']} ({existing} 张)")
        else:
            remaining.append(cfg)
    return remaining

TRADITIONAL_ATTACKS = filter_completed_attacks(TRADITIONAL_ATTACKS, "传统攻击")
SDXL_ATTACKS = filter_completed_attacks(SDXL_ATTACKS, "SDXL")
FLUXFILL_ATTACKS = filter_completed_attacks(FLUXFILL_ATTACKS, "FluxFill")
SDEDIT_ATTACKS = filter_completed_attacks(SDEDIT_ATTACKS, "SDEdit")

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

            elif attack_type == 'copy_move':
                attacked_img, true_mask = attack_copy_move(img, params['block_size_ratio'])
                mask_filename = f"{img_id}_mask.npy"
                mask_path = os.path.join(attack_dir, mask_filename)
                np.save(mask_path, true_mask)
                img_attacks['attacks'][attack_name] = {
                    'image_file': f"{img_id}.png",
                    'mask_file': mask_filename
                }
                attacked_path = os.path.join(attack_dir, f"{img_id}.png")
                attacked_img.save(attacked_path)

            elif attack_type == 'splicing':
                # 随机选择同批次另一张图作为 donor
                donor_idx = random.randint(0, len(images_info) - 1)
                donor_path = os.path.join(watermarked_dir, images_info[donor_idx]['image_file'])
                donor_img = Image.open(donor_path).convert('RGB')
                attacked_img, true_mask = attack_splicing(img, donor_img, params['block_size_ratio'])
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
                # SDEdit 现在由 DeepLearningAttacker 处理（需要 FLUX pipe）
                # 跳过，后面会统一处理
                continue

        except Exception as e:
            print(f"\n❌ 传统攻击 {attack_name} 失败: {e}")
            continue

    return img_attacks


print("\n" + "="*60)
print("Phase 1: 传统攻击（高性能并行模式）")
print("="*60)

if TRADITIONAL_ATTACKS:
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
    attacker = None  # 初始化，后面深度学习攻击会赋值

    print(f"\n✅ 传统攻击完成")
    print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
else:
    print("   所有传统攻击均已完成，跳过 Phase 1")
    # 尝试恢复已有的 attack_summary 中的 images 字段，避免覆盖丢失
    existing_summary_path = os.path.join(base_output_dir, f'{config["experiment_name"]}_attack_summary.json')
    if os.path.exists(existing_summary_path):
        with open(existing_summary_path, 'r') as f:
            old_summary = json.load(f)
        attack_summary['images'] = old_summary.get('images', [])
    else:
        attack_summary['images'] = []
    attacker = None

# ==========================================
# Phase 2: 深度学习攻击（模型逐个加载，避免OOM）
# ==========================================
if SDXL_ATTACKS or FLUXFILL_ATTACKS or SDEDIT_ATTACKS:
    print("\n" + "="*60)
    print("Phase 2: 深度学习攻击（模型逐个加载）")
    print("="*60)

    # 初始化攻击器（不预先加载模型）
    attacker = DeepLearningAttacker(flux_pipe=None)

    # SDXL攻击（语义修改）- 逐个加载SDXL模型
    if SDXL_ATTACKS:
        print("\n🎨 执行 SDXL 全局风格迁移攻击...")
        for attack_config in SDXL_ATTACKS:
            attack_dir = os.path.join(base_output_dir, attack_config['name'])
            attacker.sdxl_attack_batch(images_info, attack_config, attack_dir)
            # 每次攻击后清理显存
            torch.cuda.empty_cache()

    # FluxFill攻击（局部重绘）- 逐个加载FluxFill模型
    if FLUXFILL_ATTACKS:
        print("\n🎨 执行 FluxFill 局部重绘攻击...")
        for attack_config in FLUXFILL_ATTACKS:
            attack_dir = os.path.join(base_output_dir, attack_config['name'])
            attacker.fluxfill_attack_batch(images_info, attack_config, attack_dir)
            # 每次攻击后清理显存
            torch.cuda.empty_cache()

    # SDEdit攻击（扩散再生）- 使用FLUX进行真正的扩散再生
    if SDEDIT_ATTACKS:
        print("\n🎨 执行 SDEdit 扩散再生攻击...")
        # 加载FLUX用于SDEdit攻击
        print("\n🚀 加载 FLUX 用于SDEdit攻击...")
        from diffusers import FluxImg2ImgPipeline
        flux_pipe = FluxImg2ImgPipeline.from_pretrained(
            config['model_path'],
            torch_dtype=torch.bfloat16
        )
        flux_pipe = flux_pipe.to("cuda")
        flux_pipe.enable_attention_slicing()
        print("   FLUX已加载到GPU")
        print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        attacker.flux_pipe = flux_pipe

        for attack_config in SDEDIT_ATTACKS:
            attack_dir = os.path.join(base_output_dir, attack_config['name'])
            attacker.sdedit_attack_batch(images_info, attack_config, attack_dir)

        # SDEdit完成后卸载FLUX
        print("\n🧹 卸载 FLUX 模型释放显存...")
        del flux_pipe
        attacker.flux_pipe = None
        torch.cuda.empty_cache()
        print(f"   当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # 保存 prompts_modified.json（记录实际使用的修改prompts）
    if attacker.modified_prompts:
        prompts_modified_used_path = os.path.join(config['output_base_dir'], 'prompts_modified_used.json')
        with open(prompts_modified_used_path, 'w') as f:
            json.dump({
                'experiment_name': config['experiment_name'],
                'timestamp': datetime.now().isoformat(),
                'description': '实际使用的语义修改prompts（来自prompts_modified.json或默认）',
                'modified_prompts': attacker.modified_prompts,
                'attack_types': {
                    'sdxl': [a['name'] for a in SDXL_ATTACKS],
                    'fluxfill': [a['name'] for a in FLUXFILL_ATTACKS],
                    'sdedit': [a['name'] for a in SDEDIT_ATTACKS]
                }
            }, f, indent=2)
        print(f"\n📝 已保存实际使用的修改prompts: {prompts_modified_used_path}")
        print(f"   共 {len(attacker.modified_prompts)} 张图像的语义修改记录")

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
if attacker and attacker.modified_prompts:
    prompts_modified_used_path = os.path.join(config['output_base_dir'], 'prompts_modified_used.json')
    print(f"   实际使用的语义修改记录: {prompts_modified_used_path}")
    print(f"   （原始 prompts_modified.json 由用户生成，未被修改）")
print(f"\n📊 攻击类型统计:")
for attack in ATTACK_CONFIGS:
    attack_dir = os.path.join(base_output_dir, attack['name'])
    if os.path.exists(attack_dir):
        num_files = len([f for f in os.listdir(attack_dir) if f.endswith(('.png', '.jpg'))])
        print(f"   - {attack['name']}: {num_files} 张")
print(f"\n   攻击类型说明:")
print(f"   - 传统攻击: JPEG压缩、模糊、裁剪、噪声、缩放、亮度/对比度调整、黑块")
print(f"   - SDXL全局风格迁移: 油画风格、素描风格、水彩风格、赛博朋克、动漫风格")
print(f"     （支持使用 prompts_modified.json 中的语义修改prompt）")
print(f"   - FluxFill局部重绘: 中心区域、随机区域重绘（使用修改后prompt）")
print(f"   - SDEdit扩散再生: 真正的扩散模型重生成（使用FLUX）")
print(f"\n   最终显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"   🚀 高性能模式：SDXL和FluxFill可同时常驻GPU！")
