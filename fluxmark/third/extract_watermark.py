#!/usr/bin/env python3
"""
水印提取模块（极致性能版 - 50GB显存全速运行）
提供提取8x8签名的函数，供其他脚本调用

优化策略：
- 保持模型常驻GPU
- 批量提取签名
- 预编码prompts缓存于GPU
"""

import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import json

# 加载配置
with open('config.json', 'r') as f:
    config = json.load(f)

os.environ['HF_HOME'] = config['hf_cache']


class WatermarkExtractor:
    """水印提取器类（高性能版）"""

    def __init__(self, pipe=None, keep_models_on_gpu=True):
        """
        初始化提取器
        Args:
            pipe: 可选，传入已加载的pipeline
            keep_models_on_gpu: 是否保持模型常驻GPU（高性能模式）
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._encoded_cache = {}
        self._keep_on_gpu = keep_models_on_gpu

        if pipe is None:
            print("🚀 加载 FLUX 模型...")
            self.pipe = FluxPipeline.from_pretrained(
                config['model_path'],
                torch_dtype=torch.bfloat16
            )
            print("   加载到GPU（高性能模式）...")
            self.pipe.to(self.device)
            self.pipe.enable_vae_tiling()
            self.pipe.enable_attention_slicing()
        else:
            print("📦 使用传入的pipeline...")
            self.pipe = pipe

        self._rebuild_watermark_codebook()

    def _rebuild_watermark_codebook(self):
        """重建FFT密码本"""
        print("🔐 重建FFT密码本...")

        secret_key = config['secret_key']
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

        self.W = W / (torch.norm(W, dim=-1, keepdim=True) + 1e-8)

        M_spatial = torch.zeros((1, 32, 32, 1), device='cuda', dtype=torch.bfloat16)
        y_s, y_e, x_s, x_e = config['mask_region']
        M_spatial[0, y_s:y_e, x_s:x_e, 0] = 1.0
        M_attn = M_spatial.view(1, 1024, 1)
        self.W_anchored = self.W * M_attn

        self.mask_8x8 = np.zeros((8, 8))
        self.mask_8x8[1:7, 1:7] = 1.0

        print(f"   密码本已创建（GPU）")

    def encode_prompt(self, prompt):
        """编码prompt（带缓存）"""
        if prompt in self._encoded_cache:
            return self._encoded_cache[prompt]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = self.pipe.encode_prompt(
                prompt=prompt, prompt_2=None, max_sequence_length=256
            )

        result = {
            'prompt_embeds': prompt_embeds,
            'pooled_prompt_embeds': pooled_prompt_embeds,
            'text_ids': text_ids
        }
        self._encoded_cache[prompt] = result
        return result

    def extract_signature(self, image, prompt_embeds=None, pooled_prompt_embeds=None, text_ids=None, prompt=None):
        """从图像中提取8x8签名"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        if prompt_embeds is None and prompt is not None:
            encoded = self.encode_prompt(prompt)
            prompt_embeds = encoded['prompt_embeds']
            pooled_prompt_embeds = encoded['pooled_prompt_embeds']
            text_ids = encoded['text_ids']

        img_tensor = T.ToTensor()(image).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        img_tensor = (img_tensor - 0.5) * 2.0

        with torch.no_grad():
            z_0 = self.pipe.vae.encode(img_tensor).latent_dist.sample()
            z_0 = (z_0 - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
            z_0 = self.pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)

            h = w = 32
            img_ids = torch.zeros(h, w, 3, device=self.device, dtype=torch.bfloat16)
            img_ids[..., 1] = torch.arange(h, device=self.device, dtype=torch.bfloat16).unsqueeze(1) / max(h - 1, 1)
            img_ids[..., 2] = torch.arange(w, device=self.device, dtype=torch.bfloat16).unsqueeze(0) / max(w - 1, 1)
            img_ids = img_ids.view(1, h * w, 3)

            t_detect = torch.tensor([self.pipe.scheduler.sigmas[-1]], device=self.device, dtype=torch.bfloat16)
            v_pred = self.pipe.transformer(
                hidden_states=z_0, timestep=t_detect / 1000,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

        v_pred_spatial = v_pred.view(32, 32, 64)
        W_anchored_spatial = self.W_anchored.view(32, 32, 64)
        S = np.zeros((8, 8))

        for i in range(8):
            for j in range(8):
                W_patch = W_anchored_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
                v_patch = v_pred_spatial[i*4:(i+1)*4, j*4:(j+1)*4, :].flatten().float()
                S[i, j] = torch.nn.functional.cosine_similarity(
                    W_patch.unsqueeze(0), v_patch.unsqueeze(0)
                ).item()

        return S

    def extract_signature_batch(self, images, prompts):
        """批量提取签名（高性能版）"""
        signatures = []
        for img, prompt in zip(images, prompts):
            encoded = self._encoded_cache.get(prompt)
            if encoded is None:
                encoded = self.encode_prompt(prompt)
            S = self.extract_signature(
                img,
                prompt_embeds=encoded['prompt_embeds'],
                pooled_prompt_embeds=encoded['pooled_prompt_embeds'],
                text_ids=encoded['text_ids']
            )
            signatures.append(S)

        return signatures

    def compute_core_stats(self, S):
        """计算核心区域统计信息"""
        core_scores = S[self.mask_8x8 == 1.0]
        return {
            'mean': float(np.mean(core_scores)),
            'std': float(np.std(core_scores)),
            'min': float(np.min(core_scores)),
            'max': float(np.max(core_scores))
        }


def extract_signature_from_file(image_path, signature_output_path, prompt=None,
                                 prompt_embeds=None, pooled_prompt_embeds=None, text_ids=None):
    """从图像文件提取签名并保存（高性能版）"""
    extractor = WatermarkExtractor(keep_models_on_gpu=True)

    if prompt is not None and prompt_embeds is None:
        encoded = extractor.encode_prompt(prompt)
        prompt_embeds = encoded['prompt_embeds']
        pooled_prompt_embeds = encoded['pooled_prompt_embeds']
        text_ids = encoded['text_ids']

    S = extractor.extract_signature(
        image_path,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        text_ids=text_ids
    )

    np.save(signature_output_path, S)
    stats = extractor.compute_core_stats(S)

    print(f"✅ 签名已保存至: {signature_output_path}")
    print(f"   核心区域均值: {stats['mean']:.4f}")

    return S, stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='提取图像水印签名（高性能版）')
    parser.add_argument('--image', '-i', required=True, help='输入图像路径')
    parser.add_argument('--output', '-o', required=True, help='签名输出路径(.npy)')
    parser.add_argument('--prompt', '-p', help='图像对应的prompt')

    args = parser.parse_args()

    extract_signature_from_file(args.image, args.output, prompt=args.prompt)
