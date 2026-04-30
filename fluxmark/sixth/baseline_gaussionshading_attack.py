#!/usr/bin/env python3
"""
对 baseline_gaussionshading_flux.py 生成的带水印图像施加所有攻击，
并进行 GaussianShading 水印提取与鲁棒性分析。

支持断点续跑：只要攻击图片存在就跳过攻击生成；JSON 没分数则只补提取。
"""

import os
import json
import random

# 必须在 import torch 之前设置 GPU
with open('config.json', 'r') as f:
    config = json.load(f)

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.get('gpu_id', 0))
os.environ['HF_HOME'] = config.get('hf_cache', '/home/daiyn/hf_cache')

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from diffusers import FluxPipeline, FluxImg2ImgPipeline
import gc

from attack_utils import (
    attack_jpeg, attack_blur, attack_crop_center, attack_noise,
    attack_resize, attack_brightness_contrast, attack_black_block,
    attack_copy_move, attack_splicing,
    ATTACK_CONFIGS, DeepLearningAttacker
)

# ==========================================
# 基础配置
# ==========================================
model_path = config.get('model_path', '/home/daiyn/project_flux/model/flux-schnell')
gpu_id = config.get('gpu_id', 0)
device = 'cuda:0'

seed = config.get('prompt_seed', 42)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

BASELINE_DIR = "pic/baseline_gaussionshading_img"
ORIG_IMG_PATH = os.path.join(BASELINE_DIR, "baseline_gaussionshading_flux.png")
KEY_W_PATH = os.path.join(BASELINE_DIR, "baseline_gaussionshading_key_W.npy")
OUTPUT_DIR = os.path.join(BASELINE_DIR, "attacks")
RESULT_JSON = os.path.join(BASELINE_DIR, "baseline_gaussionshading_robustness.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载 key_W
if not os.path.exists(KEY_W_PATH):
    print(f"❌ 未找到 key_W: {KEY_W_PATH}")
    print("   请先运行 baseline_gaussionshading_flux.py")
    exit(1)

key_W = torch.from_numpy(np.load(KEY_W_PATH)).to(device, dtype=torch.float32)
print(f"✅ 已加载 key_W, shape: {key_W.shape}")

# 加载原始图像
if not os.path.exists(ORIG_IMG_PATH):
    print(f"❌ 未找到原始图像: {ORIG_IMG_PATH}")
    exit(1)

orig_img = Image.open(ORIG_IMG_PATH).convert("RGB")
print(f"✅ 已加载原始图像: {ORIG_IMG_PATH}")

# ==========================================
# 加载 FLUX 模型
# ==========================================
print("\n🚀 加载 FLUX 模型用于水印提取...")
pipe = FluxPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
).to(device)

prompt = "Elderly gray haired man in a suit scowling into the camera."
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=256
    )

pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.tokenizer = None
pipe.tokenizer_2 = None
gc.collect()
torch.cuda.empty_cache()
print("✅ FLUX 模型已精简（保留 VAE + Transformer + Scheduler）")

# ==========================================
# 提取函数
# ==========================================
def pack_latents(latents):
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latents(latents, h=64, w=64):
    b, _, c4 = latents.shape
    c = c4 // 4
    latents = latents.view(b, h // 2, w // 2, c, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, c, h, w)


def extract_gs_score(image, key_W):
    # 统一 resize 到 512x512，避免 1024x1024 攻击图像提取过慢
    if image.size != (512, 512):
        image = image.resize((512, 512), Image.LANCZOS)
    img_tensor = T.ToTensor()(image).unsqueeze(0).to(device, dtype=torch.bfloat16)
    img_tensor = (img_tensor - 0.5) * 2.0
    _, _, img_h, img_w = img_tensor.shape
    latent_h, latent_w = img_h // 8, img_w // 8
    with torch.no_grad():
        z_0 = pipe.vae.encode(img_tensor).latent_dist.sample()
        z_0 = (z_0 - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        z_t_packed = pack_latents(z_0)
        b, seq_len, c4 = z_t_packed.shape
        grid_h = latent_h // 2
        grid_w = latent_w // 2
        img_ids = torch.zeros(grid_h, grid_w, 3, device=device, dtype=torch.bfloat16)
        img_ids[..., 1] = torch.arange(grid_h, device=device, dtype=torch.bfloat16).unsqueeze(1) / max(grid_h - 1.0, 1.0)
        img_ids[..., 2] = torch.arange(grid_w, device=device, dtype=torch.bfloat16).unsqueeze(0) / max(grid_w - 1.0, 1.0)
        img_ids = img_ids.view(seq_len, 3)
        txt_ids_2d = text_ids if text_ids.dim() == 2 else text_ids.squeeze(0)
        sigmas = pipe.scheduler.sigmas
        for i in range(len(sigmas) - 1, 0, -1):
            t_curr = sigmas[i]
            t_next = sigmas[i-1]
            v_pred = pipe.transformer(
                hidden_states=z_t_packed, timestep=torch.tensor([t_curr], device=device, dtype=torch.bfloat16) / 1000.0,
                pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids_2d, img_ids=img_ids, return_dict=False,
            )[0]
            dt = t_next - t_curr
            z_t_packed = z_t_packed + v_pred * dt
        z_1_hat = unpack_latents(z_t_packed, h=latent_h, w=latent_w).to(torch.float32)
    orig_key_flat = key_W.flatten()
    extracted_flat = z_1_hat.flatten()
    score = torch.nn.functional.cosine_similarity(orig_key_flat.unsqueeze(0), extracted_flat.unsqueeze(0)).item()
    return score


# ==========================================
# 断点续跑：加载已有结果
# ==========================================
if os.path.exists(RESULT_JSON):
    with open(RESULT_JSON, 'r') as f:
        results = json.load(f)
    print(f"📂 已加载已有结果 ({len(results.get('attacks', {}))} 条记录)")
else:
    results = {
        'experiment': 'baseline_gaussionshading_robustness',
        'model': 'flux-schnell',
        'prompt': prompt,
        'original_score': None,
        'attacks': {}
    }


def save_results():
    with open(RESULT_JSON, 'w') as f:
        json.dump(results, f, indent=2)


def reload_pipe_for_extraction():
    """SDEdit 会污染 pipe，完整重建最保险"""
    global pipe, prompt_embeds, pooled_prompt_embeds, text_ids
    print("   🔄 完整重建 FLUX 提取 pipeline...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    pipe = FluxPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    ).to(device)

    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=256
        )

    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    pipe.tokenizer = None
    pipe.tokenizer_2 = None
    gc.collect()
    torch.cuda.empty_cache()
    print("   ✅ FLUX 提取 pipeline 重建完成")


# ==========================================
# 基准提取
# ==========================================
if results.get('original_score') is None:
    print("\n" + "="*60)
    print("基准提取（原始水印图像）")
    print("="*60)
    orig_score = extract_gs_score(orig_img, key_W)
    results['original_score'] = float(orig_score)
    save_results()
    print(f"原始图像提取分数: {orig_score:.4f}")
else:
    orig_score = results['original_score']
    print(f"\n✅ 基准提取已存在: {orig_score:.4f}")

images_info = [{'id': 'baseline', 'prompt': prompt, 'image_file': 'baseline_gaussionshading_flux.png'}]

TRADITIONAL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] not in ['sdxl_i2i', 'flux_fill', 'sdedit']]
SDXL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] == 'sdxl_i2i']
FLUXFILL_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] == 'flux_fill']
SDEDIT_ATTACKS = [c for c in ATTACK_CONFIGS if c['type'] == 'sdedit']


# ==========================================
# Phase 1: 传统攻击
# ==========================================
print("\n" + "="*60)
print("Phase 1: 传统攻击 + 提取")
print("="*60)

for attack_config in TRADITIONAL_ATTACKS:
    attack_name = attack_config['name']
    attack_type = attack_config['type']
    params = attack_config['params']
    save_ext = 'jpg' if attack_type == 'jpeg' else 'png'
    attacked_path = os.path.join(OUTPUT_DIR, f"{attack_name}.{save_ext}")

    # 完全跳过
    if os.path.exists(attacked_path) and 'score' in results.get('attacks', {}).get(attack_name, {}):
        print(f"\n⏭️  {attack_name} 已存在且已有分数，跳过")
        continue

    # 图片存在但无分数 -> 只提取
    if os.path.exists(attacked_path) and 'score' not in results.get('attacks', {}).get(attack_name, {}):
        print(f"\n🔨 {attack_name} 图像已存在，仅提取")
        try:
            attacked_img = Image.open(attacked_path).convert("RGB")
            score = extract_gs_score(attacked_img, key_W)
            results['attacks'][attack_name] = {
                'score': float(score), 'type': attack_type,
                'params': params, 'image_file': f"{attack_name}.{save_ext}"
            }
            save_results()
            print(f"   ✅ 分数: {score:.4f}")
        except Exception as e:
            print(f"   ❌ 提取失败: {e}")
            results['attacks'][attack_name] = {'error': str(e), 'type': attack_type, 'params': params}
            save_results()
        continue

    # 图片不存在 -> 攻击 + 提取
    print(f"\n🔨 {attack_name}")
    try:
        if attack_type == 'jpeg':
            attacked_img = attack_jpeg(orig_img, params['quality'])
        elif attack_type == 'blur':
            attacked_img = attack_blur(orig_img, params['sigma'], params.get('kernel_size', 9))
        elif attack_type == 'crop':
            attacked_img = attack_crop_center(orig_img, params['crop_ratio'])
        elif attack_type == 'noise':
            attacked_img = attack_noise(orig_img, params['std'])
        elif attack_type == 'resize':
            attacked_img = attack_resize(orig_img, params['scale'])
        elif attack_type == 'brightness_contrast':
            attacked_img = attack_brightness_contrast(orig_img, params['brightness'], params['contrast'])
        elif attack_type == 'black_block':
            attacked_img, true_mask = attack_black_block(orig_img, params['block_size_ratio'], params['position'])
            np.save(os.path.join(OUTPUT_DIR, f"{attack_name}_mask.npy"), true_mask)
        elif attack_type == 'copy_move':
            attacked_img, true_mask = attack_copy_move(orig_img, params['block_size_ratio'])
            np.save(os.path.join(OUTPUT_DIR, f"{attack_name}_mask.npy"), true_mask)
        elif attack_type == 'splicing':
            attacked_img, true_mask = attack_splicing(orig_img, orig_img, params['block_size_ratio'])
            np.save(os.path.join(OUTPUT_DIR, f"{attack_name}_mask.npy"), true_mask)
        else:
            print(f"   ⚠️ 未知攻击类型: {attack_type}")
            continue

        attacked_img.save(attacked_path)
        print(f"   💾 攻击图像已保存: {attacked_path}")

        score = extract_gs_score(attacked_img, key_W)
        results['attacks'][attack_name] = {
            'score': float(score), 'type': attack_type,
            'params': params, 'image_file': f"{attack_name}.{save_ext}"
        }
        save_results()
        print(f"   ✅ 分数: {score:.4f}")
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        results['attacks'][attack_name] = {'error': str(e), 'type': attack_type, 'params': params}
        save_results()


# ==========================================
# Phase 2: SDEdit 攻击
# ==========================================
if SDEDIT_ATTACKS:
    print("\n" + "="*60)
    print("Phase 2: SDEdit 攻击 + 提取")
    print("="*60)

    need_generate = []
    need_extract = []

    for attack_config in SDEDIT_ATTACKS:
        attack_name = attack_config['name']
        attacked_path = os.path.join(OUTPUT_DIR, f"{attack_name}.png")
        if os.path.exists(attacked_path):
            if 'score' in results.get('attacks', {}).get(attack_name, {}):
                print(f"⏭️  {attack_name} 已存在且已有分数，跳过")
            else:
                need_extract.append(attack_config)
        else:
            need_generate.append(attack_config)

    if not need_generate and not need_extract:
        print("   所有 SDEdit 攻击均已完成，跳过")
    else:
        # 生成未完成的攻击图
        if need_generate:
            print(f"   待生成: {[c['name'] for c in need_generate]}")
            for attack_config in need_generate:
                attack_name = attack_config['name']
                params = attack_config['params']
                print(f"\n🔨 {attack_name} (noise_strength={params['noise_strength']})")
                try:
                    print("   🚀 加载 FluxImg2ImgPipeline...")
                    sdedit_pipe = FluxImg2ImgPipeline.from_pretrained(
                        model_path, torch_dtype=torch.bfloat16
                    ).to(device)

                    attacked_img = sdedit_pipe(
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        image=orig_img,
                        strength=params['noise_strength'],
                        num_inference_steps=4,
                        guidance_scale=0.0,
                        height=512, width=512,
                        generator=torch.Generator(device=device).manual_seed(42),
                    ).images[0]

                    attacked_path = os.path.join(OUTPUT_DIR, f"{attack_name}.png")
                    attacked_img.save(attacked_path)
                    print(f"   💾 攻击图像已保存: {attacked_path}")

                    del sdedit_pipe
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"   ❌ 攻击失败: {e}")
                    results['attacks'][attack_name] = {'error': str(e), 'type': 'sdedit', 'params': params}
                    save_results()

        # 需要提取时重建 pipe
        if need_generate or need_extract:
            reload_pipe_for_extraction()

            for attack_config in SDEDIT_ATTACKS:
                attack_name = attack_config['name']
                params = attack_config['params']
                if 'score' in results.get('attacks', {}).get(attack_name, {}):
                    continue
                if 'error' in results.get('attacks', {}).get(attack_name, {}):
                    continue
                attacked_path = os.path.join(OUTPUT_DIR, f"{attack_name}.png")
                if not os.path.exists(attacked_path):
                    continue

                print(f"\n⏪ 提取 {attack_name} ...")
                try:
                    attacked_img = Image.open(attacked_path).convert("RGB")
                    score = extract_gs_score(attacked_img, key_W)
                    results['attacks'][attack_name] = {
                        'score': float(score), 'type': 'sdedit',
                        'params': params, 'image_file': f"{attack_name}.png"
                    }
                    save_results()
                    print(f"   ✅ 分数: {score:.4f}")
                except Exception as e:
                    print(f"   ❌ 提取失败: {e}")
                    results['attacks'][attack_name] = {'error': str(e), 'type': 'sdedit', 'params': params}
                    save_results()


# ==========================================
# Phase 3: SDXL 攻击
# ==========================================
if SDXL_ATTACKS:
    print("\n" + "="*60)
    print("Phase 3: SDXL 攻击 + 提取")
    print("="*60)

    need_attack = []
    need_extract_only = []
    for attack_config in SDXL_ATTACKS:
        attack_name = attack_config['name']
        attacked_path = os.path.join(OUTPUT_DIR, attack_name, "baseline.png")
        if os.path.exists(attacked_path):
            if 'score' in results.get('attacks', {}).get(attack_name, {}):
                print(f"⏭️  {attack_name} 已存在且已有分数，跳过")
            else:
                need_extract_only.append(attack_config)
        else:
            need_attack.append(attack_config)

    if not need_attack and not need_extract_only:
        print("   所有 SDXL 攻击均已完成，跳过")
    else:
        attacker = DeepLearningAttacker(config, BASELINE_DIR)

        for attack_config in need_attack:
            attack_name = attack_config['name']
            attack_dir = os.path.join(OUTPUT_DIR, attack_name)
            os.makedirs(attack_dir, exist_ok=True)
            print(f"\n🔨 {attack_name}")
            attacker.sdxl_attack_batch(images_info, attack_config, attack_dir)

            attacked_path = os.path.join(attack_dir, "baseline.png")
            if os.path.exists(attacked_path):
                try:
                    attacked_img = Image.open(attacked_path).convert("RGB")
                    score = extract_gs_score(attacked_img, key_W)
                    results['attacks'][attack_name] = {
                        'score': float(score), 'type': 'sdxl_i2i',
                        'params': attack_config['params'],
                        'image_file': os.path.join(attack_name, "baseline.png")
                    }
                    save_results()
                    print(f"   ✅ 分数: {score:.4f}")
                except Exception as e:
                    print(f"   ❌ 提取失败: {e}")
                    results['attacks'][attack_name] = {
                        'error': str(e), 'type': 'sdxl_i2i',
                        'params': attack_config['params']
                    }
                    save_results()
            else:
                print(f"   ⚠️ 未找到攻击后图像")
                results['attacks'][attack_name] = {
                    'error': 'attack image not generated', 'type': 'sdxl_i2i',
                    'params': attack_config['params']
                }
                save_results()

        for attack_config in need_extract_only:
            attack_name = attack_config['name']
            attacked_path = os.path.join(OUTPUT_DIR, attack_name, "baseline.png")
            print(f"\n🔨 {attack_name} 图像已存在，仅提取")
            try:
                attacked_img = Image.open(attacked_path).convert("RGB")
                score = extract_gs_score(attacked_img, key_W)
                results['attacks'][attack_name] = {
                    'score': float(score), 'type': 'sdxl_i2i',
                    'params': attack_config['params'],
                    'image_file': os.path.join(attack_name, "baseline.png")
                }
                save_results()
                print(f"   ✅ 分数: {score:.4f}")
            except Exception as e:
                print(f"   ❌ 提取失败: {e}")
                results['attacks'][attack_name] = {
                    'error': str(e), 'type': 'sdxl_i2i',
                    'params': attack_config['params']
                }
                save_results()

        attacker._unload_sdxl()
        gc.collect()
        torch.cuda.empty_cache()


# ==========================================
# Phase 4: FluxFill 攻击
# ==========================================
if FLUXFILL_ATTACKS:
    print("\n" + "="*60)
    print("Phase 4: FluxFill 攻击 + 提取")
    print("="*60)

    need_attack = []
    need_extract_only = []
    for attack_config in FLUXFILL_ATTACKS:
        attack_name = attack_config['name']
        attacked_path = os.path.join(OUTPUT_DIR, attack_name, "baseline.png")
        if os.path.exists(attacked_path):
            if 'score' in results.get('attacks', {}).get(attack_name, {}):
                print(f"⏭️  {attack_name} 已存在且已有分数，跳过")
            else:
                need_extract_only.append(attack_config)
        else:
            need_attack.append(attack_config)

    if not need_attack and not need_extract_only:
        print("   所有 FluxFill 攻击均已完成，跳过")
    else:
        attacker = DeepLearningAttacker(config, BASELINE_DIR)

        for attack_config in need_attack:
            attack_name = attack_config['name']
            attack_dir = os.path.join(OUTPUT_DIR, attack_name)
            os.makedirs(attack_dir, exist_ok=True)
            print(f"\n🔨 {attack_name}")
            attacker.fluxfill_attack_batch(images_info, attack_config, attack_dir)

            attacked_path = os.path.join(attack_dir, "baseline.png")
            if os.path.exists(attacked_path):
                try:
                    attacked_img = Image.open(attacked_path).convert("RGB")
                    score = extract_gs_score(attacked_img, key_W)
                    results['attacks'][attack_name] = {
                        'score': float(score), 'type': 'flux_fill',
                        'params': attack_config['params'],
                        'image_file': os.path.join(attack_name, "baseline.png")
                    }
                    save_results()
                    print(f"   ✅ 分数: {score:.4f}")
                except Exception as e:
                    print(f"   ❌ 提取失败: {e}")
                    results['attacks'][attack_name] = {
                        'error': str(e), 'type': 'flux_fill',
                        'params': attack_config['params']
                    }
                    save_results()
            else:
                print(f"   ⚠️ 未找到攻击后图像")
                results['attacks'][attack_name] = {
                    'error': 'attack image not generated', 'type': 'flux_fill',
                    'params': attack_config['params']
                }
                save_results()

        for attack_config in need_extract_only:
            attack_name = attack_config['name']
            attacked_path = os.path.join(OUTPUT_DIR, attack_name, "baseline.png")
            print(f"\n🔨 {attack_name} 图像已存在，仅提取")
            try:
                attacked_img = Image.open(attacked_path).convert("RGB")
                score = extract_gs_score(attacked_img, key_W)
                results['attacks'][attack_name] = {
                    'score': float(score), 'type': 'flux_fill',
                    'params': attack_config['params'],
                    'image_file': os.path.join(attack_name, "baseline.png")
                }
                save_results()
                print(f"   ✅ 分数: {score:.4f}")
            except Exception as e:
                print(f"   ❌ 提取失败: {e}")
                results['attacks'][attack_name] = {
                    'error': str(e), 'type': 'flux_fill',
                    'params': attack_config['params']
                }
                save_results()

        attacker._unload_fluxfill()
        gc.collect()
        torch.cuda.empty_cache()


# ==========================================
# 最终汇总
# ==========================================
print("\n" + "="*70)
print(" GaussianShading 鲁棒性分析汇总")
print("="*70)
print(f"{'攻击类型':<35} {'分数':>10} {'状态':>12}")
print("-" * 70)
print(f"{'原始图像 (baseline)':<35} {orig_score:>10.4f} {'OK':>12}")

for attack_config in ATTACK_CONFIGS:
    name = attack_config['name']
    info = results.get('attacks', {}).get(name, {})
    if 'score' in info:
        print(f"{name:<35} {info['score']:>10.4f} {'OK':>12}")
    elif 'error' in info:
        print(f"{name:<35} {'--':>10} {'FAIL':>12}")
    else:
        print(f"{name:<35} {'--':>10} {'PENDING':>12}")

print("="*70)
total = len(ATTACK_CONFIGS)
success = sum(1 for c in ATTACK_CONFIGS if 'score' in results.get('attacks', {}).get(c['name'], {}))
failed = sum(1 for c in ATTACK_CONFIGS if 'error' in results.get('attacks', {}).get(c['name'], {}))
print(f"📊 总攻击数: {total} | 成功: {success} | 失败: {failed} | 待完成: {total - success - failed}")
print(f"📁 结果: {RESULT_JSON}")
print("="*70)
