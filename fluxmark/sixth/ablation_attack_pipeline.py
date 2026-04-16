#!/usr/bin/env python3
"""
消融实验攻击+篡改定位一体化脚本
【优化版：加载一个模型 → 攻击 → 卸载 → 再加载下一个】
"""

import os
import sys
import json
import io
import argparse
import shutil

# 先读取配置设置 GPU（必须在 import torch 之前）
with open('config.json', 'r') as f:
    config = json.load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_id', 0))
os.environ['HF_HOME'] = config['hf_cache']

import torch
from diffusers import FluxPipeline, StableDiffusionXLInstructPix2PixPipeline
import torchvision.transforms as T
from torchvision.transforms import functional as F_vision
from PIL import Image
import numpy as np
from datetime import datetime
from tqdm import tqdm
import random
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import ImageDraw
import gc

# ==========================================
# 0. 命令行参数
# ==========================================
parser = argparse.ArgumentParser(description='消融实验攻击+篡改定位一体化脚本')
parser.add_argument('--input_dir', type=str, required=True, help='输入图片目录路径')
parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
parser.add_argument('--img_id', type=str, default=None, help='只处理指定图片ID（不含扩展名）')
parser.add_argument('--skip_dl', action='store_true', help='跳过深度学习攻击（SDXL/FluxFill/SDEdit）')
parser.add_argument('--prompt_file', type=str, default=None, help='自定义 prompts JSON 路径')
args = parser.parse_args()

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

ATTACK_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'attacks')
WATERMARK_EXTRA_DIR = os.path.join(OUTPUT_DIR, 'watermark_extra')
PIC_RESULT_DIR = os.path.join(OUTPUT_DIR, 'tamper_viz')
os.makedirs(ATTACK_OUTPUT_DIR, exist_ok=True)
os.makedirs(WATERMARK_EXTRA_DIR, exist_ok=True)
os.makedirs(PIC_RESULT_DIR, exist_ok=True)

# ==========================================
# 1. 发现输入图片
# ==========================================
all_images = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
if args.img_id:
    all_images = [f for f in all_images if os.path.splitext(f)[0] == args.img_id]
    if not all_images:
        print(f"❌ 未找到图片: {args.img_id}")
        exit(1)

print(f"📋 发现 {len(all_images)} 张输入图片")

# 加载 prompts_modified.json（用于 SDXL / FluxFill 攻击）
PROMPTS_MODIFIED_PATH = os.path.join(config['output_base_dir'], 'prompts_modified.json')
PROMPTS_MODIFIED = None
if os.path.exists(PROMPTS_MODIFIED_PATH):
    with open(PROMPTS_MODIFIED_PATH, 'r') as f:
        PROMPTS_MODIFIED = json.load(f)
    print(f"   已加载 prompts_modified.json")
else:
    print(f"   未找到 prompts_modified.json，将使用默认 prompt")

# ==========================================
# 2. 加载 prompts
# ==========================================
prompts_map = {}
if args.prompt_file and os.path.exists(args.prompt_file):
    with open(args.prompt_file, 'r') as f:
        pd = json.load(f)
    if 'prompts' in pd:
        for i, p in enumerate(pd['prompts']):
            prompts_map[f"wm_{i:03d}"] = p
    elif isinstance(pd, dict):
        prompts_map = pd
elif os.path.exists(os.path.join(INPUT_DIR, 'prompts.json')):
    with open(os.path.join(INPUT_DIR, 'prompts.json'), 'r') as f:
        pd = json.load(f)
    if 'prompts' in pd:
        for i, p in enumerate(pd['prompts']):
            prompts_map[f"wm_{i:03d}"] = p
elif os.path.exists(os.path.join(os.path.dirname(INPUT_DIR), 'prompts.json')):
    with open(os.path.join(os.path.dirname(INPUT_DIR), 'prompts.json'), 'r') as f:
        pd = json.load(f)
    if 'prompts' in pd:
        for i, p in enumerate(pd['prompts']):
            prompts_map[f"wm_{i:03d}"] = p

summary_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_summary.json')]
for sf in summary_files:
    with open(os.path.join(INPUT_DIR, sf), 'r') as f:
        sdata = json.load(f)
    for img_info in sdata.get('images', []):
        prompts_map[img_info['id']] = img_info.get('prompt', '')

images_info = []
for idx, img_file in enumerate(all_images):
    img_id = os.path.splitext(img_file)[0]
    prompt = prompts_map.get(img_id, '')
    images_info.append({
        'id': img_id,
        'prompt': prompt,
        'image_file': img_file,
        'input_path': os.path.join(INPUT_DIR, img_file)
    })

# ==========================================
# 3. 攻击配置
# ==========================================
ATTACK_CONFIGS = [
    {'name': 'jpeg_75', 'type': 'jpeg', 'params': {'quality': 75}},
    {'name': 'jpeg_50', 'type': 'jpeg', 'params': {'quality': 50}},
    {'name': 'jpeg_30', 'type': 'jpeg', 'params': {'quality': 30}},
    {'name': 'blur_0.5', 'type': 'blur', 'params': {'sigma': 0.5, 'kernel_size': 5}},
    {'name': 'blur_1.0', 'type': 'blur', 'params': {'sigma': 1.0, 'kernel_size': 5}},
    {'name': 'blur_2.0', 'type': 'blur', 'params': {'sigma': 2.0, 'kernel_size': 9}},
    {'name': 'crop_0.75', 'type': 'crop', 'params': {'crop_ratio': 0.75}},
    {'name': 'crop_0.50', 'type': 'crop', 'params': {'crop_ratio': 0.50}},
    {'name': 'noise_0.03', 'type': 'noise', 'params': {'std': 0.03}},
    {'name': 'noise_0.05', 'type': 'noise', 'params': {'std': 0.05}},
    {'name': 'noise_0.10', 'type': 'noise', 'params': {'std': 0.10}},
    {'name': 'resize_0.75', 'type': 'resize', 'params': {'scale': 0.75}},
    {'name': 'resize_0.50', 'type': 'resize', 'params': {'scale': 0.50}},
    {'name': 'brightness_0.8', 'type': 'brightness_contrast', 'params': {'brightness': 0.8, 'contrast': 1.0}},
    {'name': 'brightness_1.2', 'type': 'brightness_contrast', 'params': {'brightness': 1.2, 'contrast': 1.0}},
    {'name': 'contrast_0.8', 'type': 'brightness_contrast', 'params': {'brightness': 1.0, 'contrast': 0.8}},
    {'name': 'contrast_1.2', 'type': 'brightness_contrast', 'params': {'brightness': 1.0, 'contrast': 1.2}},
    {'name': 'black_block_center', 'type': 'black_block', 'params': {'block_size_ratio': 0.25, 'position': 'center'}},
    {'name': 'black_block_random', 'type': 'black_block', 'params': {'block_size_ratio': 0.25, 'position': 'random'}},
    {'name': 'sdedit_0.3', 'type': 'sdedit', 'params': {'noise_strength': 0.3}},
    {'name': 'sdxl_style_sketch', 'type': 'sdxl_i2i', 'params': {'instruction': 'convert to pencil sketch style', 'modification': 'sketch'}},
    {'name': 'sdxl_style_oil_painting', 'type': 'sdxl_i2i', 'params': {'instruction': 'transform into an oil painting style', 'modification': 'oil_painting'}},
    {'name': 'fluxfill_center', 'type': 'flux_fill', 'params': {'prompt': 'seamless continuation', 'mask_ratio': 0.3, 'position': 'center'}},
    {'name': 'fluxfill_random', 'type': 'flux_fill', 'params': {'prompt': 'seamless continuation', 'mask_ratio': 0.25, 'position': 'random'}},
]

if args.skip_dl:
    ATTACK_CONFIGS = [c for c in ATTACK_CONFIGS if c['type'] not in ['sdxl_i2i', 'flux_fill', 'sdedit']]

TRADITIONAL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] not in ['sdxl_i2i', 'flux_fill', 'sdedit']]
DL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] in ['sdxl_i2i', 'flux_fill', 'sdedit']]

# ==========================================
# 4. 攻击函数
# ==========================================
def attack_jpeg(img, quality):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def attack_blur(img, sigma, kernel_size=9):
    tensor = T.ToTensor()(img).unsqueeze(0).to('cuda')
    blurred = F_vision.gaussian_blur(tensor, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return T.ToPILImage()(blurred.squeeze(0).cpu())

def attack_crop_center(img, crop_ratio=0.5):
    w, h = img.size
    left = int(w * (1 - crop_ratio) / 2)
    top = int(h * (1 - crop_ratio) / 2)
    right = w - left
    bottom = h - top
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.Resampling.LANCZOS)

def attack_noise(img, std=0.05):
    tensor = T.ToTensor()(img).unsqueeze(0).to('cuda')
    noise = torch.randn_like(tensor) * std
    noisy = torch.clamp(tensor + noise, 0, 1)
    return T.ToPILImage()(noisy.squeeze(0).cpu())

def attack_resize(img, scale=0.5):
    w, h = img.size
    small = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    return small.resize((w, h), Image.Resampling.LANCZOS)

def attack_brightness_contrast(img, brightness=1.0, contrast=1.0):
    tensor = T.ToTensor()(img).unsqueeze(0).to('cuda')
    tensor = torch.clamp(tensor * brightness, 0, 1)
    mean = tensor.mean()
    tensor = torch.clamp((tensor - mean) * contrast + mean, 0, 1)
    return T.ToPILImage()(tensor.squeeze(0).cpu())

def attack_black_block(img, block_size_ratio=0.25, position='center'):
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    block_h = int(h * block_size_ratio)
    block_w = int(w * block_size_ratio)
    true_mask = np.zeros((8, 8), dtype=int)

    if position == 'center':
        y_start = (h - block_h) // 2
        x_start = (w - block_w) // 2
    elif position == 'random':
        y_start = random.randint(0, h - block_h)
        x_start = random.randint(0, w - block_w)
    else:
        y_start = (h - block_h) // 2
        x_start = (w - block_w) // 2

    attacked = img_array.copy()
    attacked[y_start:y_start+block_h, x_start:x_start+block_w] = 0

    mask_y_start = max(0, min(7, y_start * 8 // h))
    mask_y_end = max(0, min(8, (y_start + block_h) * 8 // h))
    mask_x_start = max(0, min(7, x_start * 8 // w))
    mask_x_end = max(0, min(8, (x_start + block_w) * 8 // w))
    true_mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end] = 1

    return Image.fromarray(attacked), true_mask

def attack_sdedit_simple(img, pipe_sdedit, noise_strength=0.3, num_inference_steps=28):
    result = pipe_sdedit(
        prompt="", image=img, strength=noise_strength,
        num_inference_steps=num_inference_steps, guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    return result

# ==========================================
# 工具函数：卸载模型 + 清空显存
# ==========================================
def unload_model(model, model_name):
    if model is not None:
        mem_before = torch.cuda.memory_allocated() / 1024**3
        # 先移到 CPU 再删除，确保 CUDA 张量引用被释放
        try:
            if hasattr(model, 'to'):
                model = model.to('cpu')
        except Exception:
            pass
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / 1024**3
        print(f"✅ 已卸载 {model_name} 模型，显存: {mem_before:.2f} GB → {mem_after:.2f} GB")

# ==========================================
# 签名提取函数（延迟加载 pipe，调用时必须已加载）
# ==========================================
def extract_signature_batch(pil_images, prompt_embeds_list, pooled_embeds_list, text_ids_list):
    img_tensors = []
    valid_indices = []
    for i, img in enumerate(pil_images):
        if img is None or prompt_embeds_list[i] is None: continue
        valid_indices.append(i)
        img = img.resize((512, 512), Image.LANCZOS)
        img_tensors.append(T.ToTensor()(img).unsqueeze(0))
    
    if not img_tensors: return [None]*len(pil_images)
    img_batch = torch.cat(img_tensors).to("cuda", dtype=torch.bfloat16)
    img_batch = (img_batch - 0.5) * 2.0

    z_0_list = []
    for t in img_batch:
        z0 = pipe.vae.encode(t.unsqueeze(0)).latent_dist.sample()
        z0 = (z0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z0 = pipe._pack_latents(z0,1,16,64,64)
        z_0_list.append(z0)

    img_ids = torch.zeros(32,32,3,device='cuda',dtype=torch.bfloat16)
    img_ids[...,1] = torch.arange(32,device='cuda',dtype=torch.bfloat16).unsqueeze(1)/31.0
    img_ids[...,2] = torch.arange(32,device='cuda',dtype=torch.bfloat16).unsqueeze(0)/31.0
    img_ids = img_ids.view(1,1024,3)

    sigs = [None]*len(pil_images)
    for idx, i in enumerate(valid_indices):
        v = pipe.transformer(
            hidden_states=z_0_list[idx],
            timestep=torch.tensor([pipe.scheduler.sigmas[-1]],device='cuda',dtype=torch.bfloat16)/1000,
            pooled_projections=pooled_embeds_list[i],
            encoder_hidden_states=prompt_embeds_list[i],
            txt_ids=text_ids_list[i], img_ids=img_ids, return_dict=False)[0]
        
        s = np.zeros((8,8))
        vv = v.view(32,32,64)
        ww = W.view(32,32,64)
        for a in range(8):
            for b in range(8):
                wp = ww[a*4:(a+1)*4, b*4:(b+1)*4].flatten().float()
                vp = vv[a*4:(a+1)*4, b*4:(b+1)*4].flatten().float()
                s[a,b] = torch.nn.functional.cosine_similarity(wp[None],vp[None]).item()
        sigs[i] = s
    return sigs

# ==========================================
# 先跑传统攻击（无模型）
# ==========================================
print("\n【1/4】执行传统攻击...")
for ac in TRADITIONAL_ATTACKS:
    name = ac['name']
    sub = os.path.join(ATTACK_OUTPUT_DIR, name)
    os.makedirs(sub, exist_ok=True)
    print(f"🔨 {name}")
    for info in tqdm(images_info, desc=name):
        img = Image.open(info['input_path']).convert('RGB')
        mask = None
        if ac['type']=='jpeg': res=attack_jpeg(img,**ac['params'])
        elif ac['type']=='blur': res=attack_blur(img,**ac['params'])
        elif ac['type']=='crop': res=attack_crop_center(img,**ac['params'])
        elif ac['type']=='noise': res=attack_noise(img,**ac['params'])
        elif ac['type']=='resize': res=attack_resize(img,**ac['params'])
        elif ac['type']=='brightness_contrast': res=attack_brightness_contrast(img,**ac['params'])
        elif ac['type']=='black_block': res,mask=attack_black_block(img,**ac['params'])
        else: continue
        res.save(os.path.join(sub,f"{info['id']}.png"))
        if mask is not None: np.save(os.path.join(sub,f"{info['id']}_mask.npy"),mask)

# ==========================================
# 【核心优化】逐个加载/攻击/卸载 DL 模型
# ==========================================
SDXL_PATH = config.get('sdxl_path', '/data/daiyina/project_flux/model/sdxl-instructpix2pix')
FLUX_FILL_PATH = config.get('flux_fill_path', '/data/daiyina/project_flux/model/flux-fill')
FLUX_MODEL_PATH = config['model_path']

# --------------------
# A. 处理 SDXL
# --------------------
if any(c['type']=='sdxl_i2i' for c in DL_ATTACKS) and not args.skip_dl:
    print("\n【2/4】加载 SDXL 模型并执行攻击...")
    sdxl = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(SDXL_PATH, torch_dtype=torch.float16).to("cuda")
    for ac in [c for c in DL_ATTACKS if c['type']=='sdxl_i2i']:
        name = ac['name']
        sub = os.path.join(ATTACK_OUTPUT_DIR,name)
        os.makedirs(sub, exist_ok=True)
        print(f"🔨 {name}")
        for info in tqdm(images_info, desc=name):
            img = Image.open(info['input_path']).convert('RGB')
            p = f"[{ac['params']['modification']}] {info['prompt']}"
            # 尝试从 prompts_modified.json 获取修改后的 prompt
            if PROMPTS_MODIFIED:
                img_modified = PROMPTS_MODIFIED.get('modified_prompts', {}).get(info['id'], {})
                for attack_key, mod_info in img_modified.items():
                    if mod_info.get('modification_type') == ac['params'].get('modification'):
                        p = mod_info.get('modified_prompt', p)
                        break
            out = sdxl(prompt=p, image=img, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7.0).images[0]
            out.save(os.path.join(sub,f"{info['id']}.png"))
    unload_model(sdxl, "SDXL")

# --------------------
# B. 处理 FluxFill（终极修复：强制 1024 + 正确 mask 缩放 + latent 对齐）
# --------------------
if any(c['type']=='flux_fill' for c in DL_ATTACKS) and not args.skip_dl:
    print("\n【3/4】加载 FluxFill 模型并执行攻击...")
    from diffusers import FluxFillPipeline
    fluxfill = FluxFillPipeline.from_pretrained(FLUX_FILL_PATH, torch_dtype=torch.bfloat16).to("cuda")
    fluxfill.enable_vae_slicing()
    fluxfill.enable_attention_slicing(slice_size="auto")

    for ac in [c for c in DL_ATTACKS if c['type']=='flux_fill']:
        name = ac['name']
        sub = os.path.join(ATTACK_OUTPUT_DIR,name)
        os.makedirs(sub, exist_ok=True)
        print(f"🔨 {name}")
        for info in tqdm(images_info, desc=name):
            img = Image.open(info['input_path']).convert('RGB')
            w, h = img.size
            
            # 获取 inpaint prompt（优先从 prompts_modified.json）
            inpaint_prompt = ac['params']['prompt']
            if PROMPTS_MODIFIED:
                img_modified = PROMPTS_MODIFIED.get('modified_prompts', {}).get(info['id'], {})
                found_prompt = False
                for attack_key, mod_info in img_modified.items():
                    if mod_info.get('modification_type') == ac['params'].get('modification'):
                        inpaint_prompt = mod_info.get('inpaint_prompt', inpaint_prompt)
                        found_prompt = True
                        break
                # fallback: fluxfill_random 若找不到 inpaint_random，则复用 inpaint_center 的 prompt
                if not found_prompt and ac['params'].get('modification') == 'inpaint_random':
                    for attack_key, mod_info in img_modified.items():
                        if mod_info.get('modification_type') == 'inpaint_center':
                            inpaint_prompt = mod_info.get('inpaint_prompt', inpaint_prompt)
                            break
            
            # 生成 mask
            mr = ac['params']['mask_ratio']
            bw, bh = int(w * mr), int(h * mr)
            if ac['params']['position'] == 'center':
                x, y = (w - bw) // 2, (h - bh) // 2
            else:
                x, y = random.randint(0, max(0, w - bw)), random.randint(0, max(0, h - bh))
            
            mask = Image.new('L', (w, h), 0)
            mask_draw = Image.new('L', (bw, bh), 255)
            mask.paste(mask_draw, (x, y))

            with torch.no_grad():
                out = fluxfill(
                    prompt=inpaint_prompt,
                    image=img,
                    mask_image=mask,
                    num_inference_steps=28,
                    guidance_scale=30.0,
                    generator=torch.Generator("cuda").manual_seed(42),
                ).images[0]

            out.save(os.path.join(sub, f"{info['id']}.png"))

            # 保存 mask
            m = np.zeros((8,8), dtype=int)
            my1 = max(0, min(7, y*8//h))
            my2 = max(0, min(8, (y+bh)*8//h))
            mx1 = max(0, min(7, x*8//w))
            mx2 = max(0, min(8, (x+bw)*8//w))
            m[my1:my2, mx1:mx2] = 1
            np.save(os.path.join(sub, f"{info['id']}_mask.npy"), m)

    unload_model(fluxfill, "FluxFill")
# --------------------
# C. 处理 SDEdit
# --------------------
if any(c['type']=='sdedit' for c in DL_ATTACKS) and not args.skip_dl:
    print("\n【4/4】加载 FluxImg2Img 执行 SDEdit...")
    from diffusers import FluxImg2ImgPipeline
    sdedit = FluxImg2ImgPipeline.from_pretrained(FLUX_MODEL_PATH, torch_dtype=torch.bfloat16).to("cuda")
    for ac in [c for c in DL_ATTACKS if c['type']=='sdedit']:
        name = ac['name']
        sub = os.path.join(ATTACK_OUTPUT_DIR,name)
        os.makedirs(sub, exist_ok=True)
        print(f"🔨 {name}")
        for info in tqdm(images_info, desc=name):
            img = Image.open(info['input_path']).convert('RGB')
            out = attack_sdedit_simple(img,sdedit,**ac['params'])
            out.save(os.path.join(sub,f"{info['id']}.png"))
    unload_model(sdedit, "SDEdit/FluxImg2Img")

# ==========================================
# 所有攻击完成后，再加载 FLUX 模型做签名提取
# ==========================================
print("\n🚀 加载 FLUX 模型用于签名提取...")
pipe = FluxPipeline.from_pretrained(config['model_path'], torch_dtype=torch.bfloat16)
pipe.enable_attention_slicing(slice_size="auto")
pipe.to("cuda")
print(f"   模型已加载，显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

print("🔤 预编码 prompts...")
encoded_prompts_cache = {}
for img_info in tqdm(images_info, desc="编码进度"):
    prompt = img_info['prompt']
    if prompt in encoded_prompts_cache: continue
    try:
        pe, ppe, tid = pipe.encode_prompt(prompt=prompt, prompt_2=None, max_sequence_length=256)
        encoded_prompts_cache[prompt] = {'prompt_embeds': pe, 'pooled_prompt_embeds': ppe, 'text_ids': tid}
    except:
        encoded_prompts_cache[prompt] = None

secret_key = config.get('secret_key', 42)
generator = torch.Generator(device='cuda').manual_seed(secret_key)
W_raw = torch.randn((1, 1024, 64), generator=generator, device='cuda', dtype=torch.float32)
W_spatial = W_raw.view(1, 32, 32, 64)
F_W = torch.fft.fftshift(torch.fft.fft2(W_spatial, dim=(1, 2)), dim=(1, 2))
h, w = 32, 32
Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
center_y, center_x = h // 2, w // 2
radius = torch.sqrt((Y - center_y)**2 + (X - center_x)**2).to('cuda')
r_inner = config['fft_radius_inner']
r_outer = config['fft_radius_outer']
mask_fft = ((radius >= r_inner) & (radius <= r_outer)).float().unsqueeze(0).unsqueeze(-1)
F_W_filtered = F_W * mask_fft
W_filtered = torch.fft.ifft2(torch.fft.ifftshift(F_W_filtered, dim=(1, 2)), dim=(1, 2)).real
W = W_filtered.view(1, 1024, 64).to(torch.bfloat16)
W = W / (torch.norm(W, dim=-1, keepdim=True) + 1e-8)
print("   密码本已重建（全图统一提取）")

# ==========================================
# 提取签名 + 可视化
# ==========================================
print("\n开始提取签名 & 生成可视化...")
batch_size=4

def visualize_tamper_heatmap(img_orig, img_attacked, S_orig, S_tamp, true_mask, save_path, attack_name, metrics=None):
    img_size = img_orig.size
    diff = S_orig - S_tamp
    heatmap_8x8 = np.abs(diff)
    smooth_heatmap = cv2.resize(heatmap_8x8, img_size, interpolation=cv2.INTER_CUBIC)
    smooth_heatmap = cv2.GaussianBlur(smooth_heatmap,(21,21),0)
    th = np.mean(heatmap_8x8)+1.5*np.std(heatmap_8x8)
    th = max(th,0.04)
    raw_mask = smooth_heatmap>th
    kernel=np.ones((5,5),np.uint8)
    cleaned_mask = cv2.morphologyEx(raw_mask.astype(np.uint8),cv2.MORPH_OPEN,kernel).astype(bool)

    fig,axes=plt.subplots(2,4,figsize=(20,10))
    axes[0,0].imshow(img_orig); axes[0,0].set_title('Original'); axes[0,0].axis('off')
    axes[0,1].imshow(img_attacked); axes[0,1].set_title(f'Attacked: {attack_name}'); axes[0,1].axis('off')
    im=axes[0,2].imshow(heatmap_8x8,cmap='hot'); axes[0,2].set_title('8x8 Heatmap'); axes[0,2].axis('off'); plt.colorbar(im,ax=axes[0,2],fraction=0.046)
    if true_mask is not None: axes[0,3].imshow(true_mask,cmap='Reds')
    axes[0,3].set_title('GT Mask'); axes[0,3].axis('off')
    im2=axes[1,0].imshow(smooth_heatmap,cmap='hot'); axes[1,0].set_title('High-Res Heatmap'); axes[1,0].axis('off'); plt.colorbar(im2,ax=axes[1,0],fraction=0.046)
    axes[1,1].imshow(img_attacked.resize(img_size),alpha=0.6); axes[1,1].imshow(smooth_heatmap,alpha=0.5,cmap='hot'); axes[1,1].set_title('Overlay'); axes[1,1].axis('off')
    axes[1,2].imshow(cleaned_mask,cmap='Blues'); axes[1,2].set_title('Detected'); axes[1,2].axis('off')
    comp=np.zeros((*cleaned_mask.shape,3))
    if true_mask is not None:
        gt=cv2.resize(true_mask.astype(np.uint8),img_size,interpolation=cv2.INTER_NEAREST).astype(bool)
        comp[np.logical_and(cleaned_mask,gt)]=[0,1,0]
        comp[np.logical_and(cleaned_mask,~gt)]=[1,1,0]
        comp[np.logical_and(~cleaned_mask,gt)]=[1,0,0]
    axes[1,3].imshow(comp); axes[1,3].set_title('Result'); axes[1,3].axis('off')
    plt.tight_layout(); plt.savefig(save_path,dpi=150,bbox_inches='tight'); plt.close()

# 提取原始签名
for s in range(0,len(images_info),batch_size):
    b=images_info[s:s+batch_size]
    imgs,pes,pse,tis,skips=[],[],[],[],[]
    for info in b:
        p=os.path.join(WATERMARK_EXTRA_DIR,f"{info['id']}_S_orig.npy")
        e=os.path.join(INPUT_DIR,f"{info['id']}_signature.npy")
        if os.path.exists(p): skips.append(True); continue
        if os.path.exists(e) and not os.path.exists(p): shutil.copy(e,p); skips.append(True); continue
        skips.append(False)
        imgs.append(Image.open(info['input_path']).convert('RGB'))
        enc=encoded_prompts_cache[info['prompt']]
        pes.append(enc['prompt_embeds']); pse.append(enc['pooled_prompt_embeds']); tis.append(enc['text_ids'])
    if all(skips):continue
    sigs=extract_signature_batch(imgs,pes,pse,tis)
    for i,info in enumerate(b):
        if skips[i]:continue
        np.save(os.path.join(WATERMARK_EXTRA_DIR,f"{info['id']}_S_orig.npy"),sigs[i])

# 提取攻击后签名
for ac in ATTACK_CONFIGS:
    an=ac['name']
    sub=os.path.join(ATTACK_OUTPUT_DIR,an)
    if not os.path.exists(sub):continue
    os.makedirs(os.path.join(WATERMARK_EXTRA_DIR,an),exist_ok=True)
    for s in range(0,len(images_info),batch_size):
        b=images_info[s:s+batch_size]
        imgs,pes,pse,tis,skips=[],[],[],[],[]
        for info in b:
            tp=os.path.join(WATERMARK_EXTRA_DIR,an,f"{info['id']}_{an}_S_tamp.npy")
            ai=os.path.join(sub,f"{info['id']}.png")
            if os.path.exists(tp) or not os.path.exists(ai): skips.append(True); continue
            skips.append(False)
            imgs.append(Image.open(ai).convert('RGB'))
            enc=encoded_prompts_cache[info['prompt']]
            pes.append(enc['prompt_embeds']); pse.append(enc['pooled_prompt_embeds']); tis.append(enc['text_ids'])
        if all(skips):continue
        sigs=extract_signature_batch(imgs,pes,pse,tis)
        for i,info in enumerate(b):
            if skips[i]:continue
            np.save(os.path.join(WATERMARK_EXTRA_DIR,an,f"{info['id']}_{an}_S_tamp.npy"),sigs[i])

# 生成可视化
cnt=0
for ac in ATTACK_CONFIGS:
    an=ac['name']
    sub=os.path.join(ATTACK_OUTPUT_DIR,an)
    sts=os.path.join(WATERMARK_EXTRA_DIR,an)
    if not os.path.exists(sub):continue
    for info in images_info:
        id=info['id']
        so=os.path.join(WATERMARK_EXTRA_DIR,f"{id}_S_orig.npy")
        st=os.path.join(sts,f"{id}_{an}_S_tamp.npy")
        if not os.path.exists(so)or not os.path.exists(st):continue
        ai=os.path.join(sub,f"{id}.png")
        if not os.path.exists(ai):continue
        So=np.load(so); St=np.load(st)
        I=Image.open(info['input_path']).convert('RGB')
        A=Image.open(ai).convert('RGB')
        tm=None
        tmp=os.path.join(sub,f"{id}_mask.npy")
        if os.path.exists(tmp): tm=np.load(tmp)
        visualize_tamper_heatmap(I,A,So,St,tm,os.path.join(PIC_RESULT_DIR,f"{an}_{id}_heatmap.png"),an)
        cnt+=1

print(f"\n🎉 全部完成！生成 {cnt} 张图")