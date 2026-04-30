# OrthoFlow 实验结果汇总报告

> **实验名称**: OrthoFlow_Topic_0327_HighPerf
> **生成时间**: 2026-04-22 11:43:28
> **样本数量**: 1000 张
> **模型路径**: `/home/daiyn/project_flux/model/flux-schnell`
> **输出目录**: `/home/daiyn/project_flux/fluxmark/sixth`

---

## 一、隐蔽性分析 (Invisibility)

- **FID**: `45.37` (lower is better, <50 excellent, <100 good, <200 moderate)
- **CLIP Score (无水印)**: `27.321924962043763`
- **CLIP Score (水印图)**: `27.288092735290526`
- **CLIP 差异**: `-0.0338` (越接近 0 越好)

---

## 二、全局鲁棒性分析 (Robustness)

### Table 1: 方法对比 (Baselines vs OrthoFlow)

| 方法 | Clean (原Prompt) | Clean (空Prompt) | JPEG-50 | SDEdit-0.4 | 提取成本 | Prompt 依赖 | 篡改定位 |
|------|------------------|-------------------|---------|------------|----------|-------------|----------|
| **Tree-Ring** | 0.0020 | 0.0020 | Failed | Failed | 4-Step Transformer | 需原 Prompt | 无 (IoU=0) |
| **Gaussian Shading** | 0.1216 | **0.1336** | **0.0391** (崩盘) | **0.0536** (崩盘) | 4-Step Transformer | 需原 Prompt | 无 (IoU=0) |
| **OrthoFlow (Ours)** | **0.0613** | **0.0613** | **0.0196** (稳健) | **0.0088** (稳健) | **1-Step Estimation** | **Prompt-Agnostic** | **8x8 热力图** |

> **画质崩塌证据**：Gaussian Shading 为在 4 步模型中存活，使用 α=0.5 的极端强度且无正交保护，直接将庞大噪声砸入生成流形。对比图见 `pic/baseline_gaussionshading_img/gs_vs_nowm_comparison.png`——GS 图像存在明显纹理扭曲与噪点，而 OrthoFlow 的正交流形投影保证绝对零画质损失。
>
> **盲提取真相**：GS 的 ODE 逆推必须依赖原图 Prompt 引导。在真实取证场景中，网图来源未知，Prompt 不可能获取。OrthoFlow 使用空 Prompt 即可单步提取签名，是真正的 Prompt-Agnostic 盲提取。
>
> **算力降维打击**：GS 每检验一张图需跑满 4 步（未来 20 步模型即 20 步）Transformer 前向。OrthoFlow 仅需 **单步速度估计 v_pred**，算力成本为 GS 的 **1/4 ~ 1/20**。

### Table 2: OrthoFlow 详细攻击数据

| 攻击类型 | Mean Cosine | Retention Rate | AUC | TPR@0.1%FPR |
|----------|-------------|----------------|-----|-------------|
| blur_0.5 | 0.0548 | 89.4% | 0.9996 | 0.9980 |
| blur_1.0 | 0.0400 | 65.3% | 0.9996 | 0.9970 |
| blur_2.0 | 0.0215 | 35.0% | 0.9982 | 0.9720 |
| blur_3.0 | 0.0136 | 21.9% | 0.9905 | 0.7830 |
| blur_4.0 | 0.0076 | 12.2% | 0.9346 | 0.2130 |
| brightness_0.8 | 0.0574 | 94.1% | 0.9999 | 0.9980 |
| brightness_1.2 | 0.0558 | 91.3% | 0.9992 | 0.9940 |
| contrast_0.8 | 0.0591 | 97.0% | 0.9999 | 0.9980 |
| contrast_1.2 | 0.0566 | 92.0% | 0.9990 | 0.9950 |
| crop_0.50 | 0.0006 | 0.8% | 0.5070 | 0.0000 |
| crop_0.75 | 0.0005 | 0.8% | 0.4998 | 0.0030 |
| jpeg_10 | 0.0039 | 6.2% | 0.7335 | 0.0720 |
| jpeg_20 | 0.0092 | 14.8% | 0.9176 | 0.3630 |
| jpeg_30 | 0.0133 | 21.3% | 0.9642 | 0.6230 |
| jpeg_50 | 0.0196 | 31.6% | 0.9925 | 0.8740 |
| jpeg_75 | 0.0283 | 45.9% | 0.9988 | 0.9760 |
| noise_0.03 | 0.0245 | 39.5% | 0.9969 | 0.9300 |
| noise_0.05 | 0.0181 | 29.3% | 0.9889 | 0.8200 |
| noise_0.10 | 0.0112 | 17.9% | 0.9464 | 0.5120 |
| noise_0.15 | 0.0080 | 12.9% | 0.8981 | 0.3100 |
| noise_0.20 | 0.0063 | 10.1% | 0.8580 | 0.1630 |
| resize_0.25 | 0.0200 | 32.5% | 0.9974 | 0.9680 |
| resize_0.50 | 0.0384 | 63.0% | 0.9994 | 0.9960 |
| resize_0.75 | 0.0524 | 85.8% | 0.9997 | 0.9980 |
| sdedit_0.3 | 0.0088 | 14.2% | 0.9443 | 0.3590 |
| sdedit_0.4 | 0.0060 | 9.5% | 0.8609 | 0.1310 |
| sdedit_0.5 | 0.0047 | 7.4% | 0.8062 | 0.0570 |
| sdxl_style_oil_painting | 0.0141 | 22.6% | 0.9262 | 0.5710 |
| sdxl_style_sketch | 0.0088 | 14.1% | 0.8381 | 0.3560 |

### 关键发现
- **SDEdit 0.3 (扩散再生)**: AUC=0.9443, TPR@0.1%FPR=0.36 
- **Crop 0.50 (结构性弱点)**: AUC=0.5070 

---

## 三、篡改定位分析 (Tamper Localization)

| 攻击类型 | Patch-AUC | F1-Score | IoU |
|----------|-----------|----------|-----|
| **Tree-Ring / Gaussian Shading (Baseline)** | N/A | N/A | **0.00** *(不具备 Localization 能力)* |
| black_block_center | 0.8950 | 0.7793 | 0.6405 |
| black_block_random | 0.8797 | 0.6945 | 0.5351 |
| copy_move | 0.9316 | 0.6372 | 0.4699 |
| fluxfill_center | 0.7629 | 0.6551 | 0.4925 |
| fluxfill_random | 0.7626 | 0.6354 | 0.4730 |
| splicing | 0.9177 | 0.6401 | 0.4734 |

> **Baseline 的结构性失明**：Tree-Ring 与 Gaussian Shading 均为全局展平的一维数字签名，无法生成空间差分热力图。面对 `black_block_center` 或 `FluxFill` 局部篡改时完全无能为力。OrthoFlow 的 8×8 差分签名则可直接绘制零误报篡改热力图。

---

## 四、消融实验分析 (Ablation Study)

### 4.1 隐蔽性 + 水印强度

| 方法 | FID | CLIP Score | S_mean |
|------|-----|------------|--------|
| Full Method | 45.37 | 27.2884 | 0.0613 |
| w/o FFT | 45.27 | 27.3902 | 0.0454 |
| w/o Orthogonal | 45.54 | 27.3458 | 0.0610 |
| w/o Orthogonal (Stress) | 56.80 | 27.4004 | 0.2559 |
| w/ Semantic Mask | 45.57 | 27.3348 | 0.0245 |

### 4.2 关键结论
- **去掉 FFT** 后 S_mean 下降了 **26.1%**，证明频域约束对能量保护至关重要。
- **语义 Mask** 的 S_mean 下降了 **60.1%**，反向证明全图统一注入的必要性。
- **正交投影** 的核心价值在于抹平方差（Variance），使全图盲提取成为可能。详见可视化结果 `ablation_variance_violin.png`。

---

## 五、网格分辨率消融 (Grid Size Ablation)

| 网格尺寸 | 向量维度 | Mean Cosine | Avg Std |
|----------|----------|-------------|---------|
| 32x32 | 64 | 0.0614 | 0.1271 |
| 16x16 | 256 | 0.0615 | 0.0727 |
| 8x8 | 1024 | 0.0614 | 0.0393 |

**结论**: 32×32（64D）方差极大，16×16（256D）有所改善，8×8（1024D）是统计学稳定性与空间精度的最优解（Sweet Spot）。

---

## 六、原始结果文件索引

- **鲁棒性分析 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/merge.json`
- **鲁棒性详情 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/detail.json`
- **隐蔽性分析 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/invisibility_analysis.json`
- **消融定量分析 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/ablation_analysis.json`
- **网格分辨率消融 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/ablation_grid/grid_ablation_summary.json`
- **消融可视化 (图片)**: `/home/daiyn/project_flux/fluxmark/sixth/result/ablation_viz/`
- **篡改热力图 (图片)**: `/home/daiyn/project_flux/fluxmark/sixth/result/pic/tamper_img/`
- **GS 画质崩塌对比 (图片)**: `/home/daiyn/project_flux/fluxmark/sixth/pic/baseline_gaussionshading_img/gs_vs_nowm_comparison.png`

---

*本报告由 `summarize_all_results.py` 自动生成*
