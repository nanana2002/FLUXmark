#!/usr/bin/env python3
"""
水印提取模块（内存优化版 - 22GB显存适配）
提供提取8x8签名的函数，供其他脚本调用
"""

import torch
from diffusers import FluxPipeline
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import gc
import json

# 加载配置
with open('config.json', 'r') as f:
    config = json.load(f)

os.environ['HF_HOME'] = config['hf_cache']


class WatermarkExtractor:
    """水印提取器类（内存优化版）"""

    def __init__(self, pipe=None, enable_cpu_offload=False):
        """
        初始化提取器
        Args:
            pipe: 可选，传入已加载的pipeline（推荐，节省显存）
            enable_cpu_offload: 是否启用CPU卸载（提取时建议False，保证精度）
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._encoded_cache = {}  # prompt编码缓存
        self._cpu_offload = enable_cpu_offload

        # 加载模型（如果未提供）
        if pipe is None:
            print("🚀 加载 FLUX 模型...")
            self.pipe = FluxPipeline.from_pretrained(
                config['model_path'],
                torch_dtype=torch.bfloat16
            )

            # 提取时建议不使用CPU offload，保证精度
            if self._cpu_offload:
                print("   启用显存优化（CPU offload）...")
                self.pipe.enable_sequential_cpu_offload()
            else:
                # 标准模式：加载到GPU
                print("   加载到GPU（保证提取精度）...")
                self.pipe.to(self.device)
                # 启用VAE切片节省显存
                self.pipe.enable_vae_slicing()
        else:
            print("📦 使用传入的pipeline...")
            self.pipe = pipe

        # 重建FFT密码本W（在GPU上直接创建）
        self._rebuild_watermark_codebook()

    def _rebuild_watermark_codebook(self):
        """重建FFT密码本"""
        print("🔐 重建FFT密码本...")

        secret_key = config['secret_key']
        device = self.device if not self._cpu_offload else 'cuda'

        # 在目标设备上生成
        generator = torch.Generator(device=device).manual_seed(secret_key)
        W_raw = torch.randn((1, 1024, 64), generator=generator, device=device, dtype=torch.float32)
        W_spatial = W_raw.view(1, 32, 32, 64)

        # FFT变换
        F_W = torch.fft.fftshift(torch.fft.fft2(W_spatial, dim=(1, 2)), dim=(1, 2))
        h, w = 32, 32
        Y, X = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        center_y, center_x = h // 2, w // 2
        radius = torch.sqrt((Y - center_y)**2 + (X - center_x)**2)

        # 带通滤波器
        r_inner = config['fft_radius_inner']
        r_outer = config['fft_radius_outer']
        mask_fft = ((radius >= r_inner) & (radius <= r_outer)).float().unsqueeze(0).unsqueeze(-1)

        F_W_filtered = F_W * mask_fft
        W_filtered = torch.fft.ifft2(torch.fft.ifftshift(F_W_filtered, dim=(1, 2)), dim=(1, 2)).real
        W = W_filtered.view(1, 1024, 64).to(torch.bfloat16)

        # L2归一化
        self.W = W / (torch.norm(W, dim=-1, keepdim=True) + 1e-8)

        # 语义掩码M
        M_spatial = torch.zeros((1, 32, 32, 1), device=device, dtype=torch.bfloat16)
        y_s, y_e, x_s, x_e = config['mask_region']
        M_spatial[0, y_s:y_e, x_s:x_e, 0] = 1.0
        M_attn = M_spatial.view(1, 1024, 1)
        self.W_anchored = self.W * M_attn

        # 8x8核心掩码
        self.mask_8x8 = np.zeros((8, 8))
        self.mask_8x8[1:7, 1:7] = 1.0

        print(f"   密码本已创建（{device}）")

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
        """
        从图像中提取8x8签名

        Args:
            image: PIL.Image或图像路径
            prompt_embeds: 预编码的prompt embeddings
            pooled_prompt_embeds: 预编码的pooled embeddings
            text_ids: 预编码的text ids
            prompt: 如果embeddings为None，提供prompt进行编码

        Returns:
            S: 8x8签名数组
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        # 编码prompt（如果未提供）
        if prompt_embeds is None and prompt is not None:
            encoded = self.encode_prompt(prompt)
            prompt_embeds = encoded['prompt_embeds']
            pooled_prompt_embeds = encoded['pooled_prompt_embeds']
            text_ids = encoded['text_ids']

        # 图像预处理
        img_tensor = T.ToTensor()(image).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
        img_tensor = (img_tensor - 0.5) * 2.0  # [0,1] -> [-1,1]

        with torch.no_grad():
            # VAE编码
            z_0 = self.pipe.vae.encode(img_tensor).latent_dist.sample()
            z_0 = (z_0 - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
            z_0 = self.pipe._pack_latents(z_0, batch_size=1, num_channels_latents=16, height=64, width=64)

            # 构造img_ids
            h = w = 32
            img_ids = torch.zeros(h, w, 3, device=self.device, dtype=torch.bfloat16)
            img_ids[..., 1] = torch.arange(h, device=self.device, dtype=torch.bfloat16).unsqueeze(1) / max(h - 1, 1)
            img_ids[..., 2] = torch.arange(w, device=self.device, dtype=torch.bfloat16).unsqueeze(0) / max(w - 1, 1)
            img_ids = img_ids.view(1, h * w, 3)

            # Transformer提取v_pred
            t_detect = torch.tensor([self.pipe.scheduler.sigmas[-1]], device=self.device, dtype=torch.bfloat16)
            v_pred = self.pipe.transformer(
                hidden_states=z_0, timestep=t_detect / 1000,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

        # 计算8x8签名
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

        # 清理中间变量
        del z_0, v_pred, img_tensor
        torch.cuda.empty_cache()

        return S

    def extract_signature_batch(self, images, prompts):
        """
        批量提取签名（内存优化版）

        Args:
            images: PIL.Image列表或路径列表
            prompts: prompt列表

        Returns:
            signatures: 签名列表
        """
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
        """
        计算核心区域统计信息

        Args:
            S: 8x8签名

        Returns:
            stats: 包含mean, std, min, max的字典
        """
        core_scores = S[self.mask_8x8 == 1.0]
        return {
            'mean': float(np.mean(core_scores)),
            'std': float(np.std(core_scores)),
            'min': float(np.min(core_scores)),
            'max': float(np.max(core_scores))
        }


def extract_signature_from_file(image_path, signature_output_path, prompt=None,
                                 prompt_embeds=None, pooled_prompt_embeds=None, text_ids=None):
    """
    从图像文件提取签名并保存（便捷函数，内存优化版）

    Args:
        image_path: 输入图像路径
        signature_output_path: 签名输出路径(.npy)
        prompt: 图像对应的prompt（如果embeddings未提供）
        prompt_embeds: 预编码embeddings
        pooled_prompt_embeds: 预编码pooled embeddings
        text_ids: 预编码text ids
    """
    # 提取时禁用CPU offload保证精度
    extractor = WatermarkExtractor(enable_cpu_offload=False)

    # 如果有prompt但没有预编码embeddings
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


# 主函数（命令行使用）
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='提取图像水印签名')
    parser.add_argument('--image', '-i', required=True, help='输入图像路径')
    parser.add_argument('--output', '-o', required=True, help='签名输出路径(.npy)')
    parser.add_argument('--prompt', '-p', help='图像对应的prompt')

    args = parser.parse_args()

    extract_signature_from_file(args.image, args.output, prompt=args.prompt)
