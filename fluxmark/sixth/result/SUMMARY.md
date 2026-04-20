# OrthoFlow 实验结果汇总报告

> **实验名称**: OrthoFlow_Topic_0327_HighPerf
> **生成时间**: 2026-04-20 17:39:03
> **样本数量**: 20 张
> **模型路径**: `/home/daiyn/project_flux/model/flux-schnell`
> **输出目录**: `/home/daiyn/project_flux/fluxmark/sixth`

---

## 一、隐蔽性分析 (Invisibility)

- **FID**: `205.60` (lower is better, <50 excellent, <100 good, <200 moderate)
- **CLIP Score (无水印)**: `31.517475128173828`
- **CLIP Score (水印图)**: `31.382282257080078`
- **CLIP 差异**: `-0.1352` (越接近 0 越好)

## 二、全局鲁棒性分析 (Robustness)

| 攻击类型 | Mean Cosine | Retention Rate | AUC | TPR@0.1%FPR |
|----------|-------------|----------------|-----|-------------|
| blur_0.5 | 0.0549 | 87.4% | 1.0000 | 1.0000 |
| blur_1.0 | 0.0416 | 66.3% | 1.0000 | 1.0000 |
| blur_2.0 | 0.0212 | 33.7% | 1.0000 | 1.0000 |
| brightness_0.8 | 0.0578 | 92.3% | 1.0000 | 1.0000 |
| brightness_1.2 | 0.0586 | 94.0% | 1.0000 | 1.0000 |
| contrast_0.8 | 0.0599 | 96.0% | 1.0000 | 1.0000 |
| contrast_1.2 | 0.0573 | 91.5% | 1.0000 | 1.0000 |
| crop_0.50 | 0.0006 | 0.9% | 0.4450 | 0.0500 |
| crop_0.75 | 0.0013 | 2.2% | 0.5075 | 0.0000 |
| jpeg_30 | 0.0133 | 20.5% | 0.8950 | 0.6500 |
| jpeg_50 | 0.0198 | 30.8% | 0.9725 | 0.8500 |
| jpeg_75 | 0.0284 | 44.6% | 1.0000 | 1.0000 |
| noise_0.03 | 0.0241 | 37.4% | 1.0000 | 1.0000 |
| noise_0.05 | 0.0173 | 27.2% | 0.9975 | 0.9500 |
| noise_0.10 | 0.0111 | 17.5% | 0.9575 | 0.8500 |
| resize_0.50 | 0.0407 | 65.4% | 1.0000 | 1.0000 |
| resize_0.75 | 0.0537 | 86.2% | 1.0000 | 1.0000 |
| sdedit_0.3 | 0.0083 | 13.3% | 0.8950 | 0.6000 |
| sdxl_style_oil_painting | 0.0171 | 27.4% | 1.0000 | 1.0000 |
| sdxl_style_sketch | 0.0118 | 18.9% | 0.9225 | 0.8000 |

### 关键发现
- **AUC = 1.0 的攻击**: jpeg_75, blur_0.5, blur_1.0, blur_2.0, noise_0.03, resize_0.75, resize_0.50, brightness_0.8, brightness_1.2, contrast_0.8, contrast_1.2, sdxl_style_oil_painting
- **SDEdit 0.3 (扩散再生)**: AUC=0.8950, TPR@0.1%FPR=0.60 — 这是最难的攻击，表现优异
- **Crop 0.50 (结构性弱点)**: AUC=0.4450 — 这是 VAE + DiT 导致的已知局限性

## 三、篡改定位分析 (Tamper Localization)

| 攻击类型 | Patch-AUC | F1-Score | IoU |
|----------|-----------|----------|-----|
| black_block_center | 0.8942 | 0.7179 | 0.5659 |
| black_block_random | 0.8657 | 0.6184 | 0.4547 |
| copy_move | 0.9274 | 0.6227 | 0.4572 |
| fluxfill_center | 0.7725 | 0.5003 | 0.3442 |
| fluxfill_random | 0.7602 | 0.4416 | 0.2987 |
| splicing | 0.9036 | 0.5517 | 0.3877 |

## 四、消融实验分析 (Ablation Study)

### 4.1 隐蔽性 + 水印强度

| 方法 | FID ↓ | CLIP Score ↑ | S_mean |
|------|-------|--------------|--------|
| Full Method | 205.60 | 31.2282 | 0.0624 |
| w/o FFT | 216.46 | 30.9388 | 0.0495 |
| w/o Orthogonal | 213.27 | 31.0691 | 0.0640 |
| w/o Orthogonal (Stress) | 227.11 | 31.4884 | 0.2329 |
| w/ Semantic Mask | 206.57 | 30.8677 | 0.0252 |

### 4.2 关键结论
- **去掉 FFT** 后 S_mean 下降了 **20.6%**，证明频域约束对能量保护至关重要。
- **语义 Mask** 的 S_mean 下降了 **59.6%**，反向证明全图统一注入的必要性。
- **正交投影** 的核心价值在于抹平方差（Variance），使全图盲提取成为可能。详见可视化结果 `ablation_variance_violin.png`。

## 五、网格分辨率消融 (Grid Size Ablation)

| 网格尺寸 | 向量维度 | Mean Cosine | Avg Std |
|----------|----------|-------------|---------|
| 32x32 | 64 | 0.0624 | 0.1253 |
| 16x16 | 256 | 0.0624 | 0.0725 |
| 8x8 | 1024 | 0.0624 | 0.0392 |

**结论**: 32×32（64D）方差极大，16×16（256D）有所改善，8×8（1024D）是统计学稳定性与空间精度的最优解（Sweet Spot）。

## 六、原始结果文件索引

- ✅ **鲁棒性分析 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/merge.json`
- ✅ **鲁棒性详情 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/detail.json`
- ✅ **隐蔽性分析 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/invisibility_analysis.json`
- ✅ **消融定量分析 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/ablation_analysis.json`
- ✅ **网格分辨率消融 (JSON)**: `/home/daiyn/project_flux/fluxmark/sixth/result/ablation_grid/grid_ablation_summary.json`
- ✅ **消融可视化 (图片)**: `/home/daiyn/project_flux/fluxmark/sixth/result/ablation_viz/`
- ✅ **篡改热力图 (图片)**: `/home/daiyn/project_flux/fluxmark/sixth/result/pic/tamper_img/`

---

*本报告由 `summarize_all_results.py` 自动生成*
