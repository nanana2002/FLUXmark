# OrthoFlow 水印实验系统 - Forth 版本

## 版本特性

基于 third 版本改进，主要变更：
1. **新增 `generate_prompt.py`**: 从 MS-COCO 2017 和 Gustavosta/Stable-Diffusion-Prompts 各随机获取100条 prompt
2. **`prompts_modified.json` 支持**: 语义修改攻击使用用户自己生成的修改后 prompt
3. **全局风格迁移攻击**: SDXL 支持油画、素描、水彩、赛博朋克、动漫等全局风格迁移
4. **模型逐个加载**: SDXL、FluxFill、FLUX 模型逐个加载/卸载，避免显存溢出（OOM）

## 文件结构

```
forth/
├── config.json                      # 配置文件
├── generate_prompt.py               # 生成初始 prompts（只做一次）
├── prompts.json                     # 生成的 prompts（200条）
├── prompts_modified.json            # 语义修改后的 prompts（用户用大模型生成）
├── prompts_modified_example.json    # prompts_modified.json 格式示例
├── no_watermark_img_generate.py     # 生成无水印图像
├── watermarked_img_generate.py      # 生成水印图像
├── attack_img_generate.py           # 生成攻击图像（支持 prompts_modified.json）
├── extract_watermark.py             # 水印提取模块
├── extract_watermarks.py            # 批量提取水印
├── analyze_robustness_result.py     # 鲁棒性分析
├── analyze_invisibility_result.py   # 隐蔽性分析
├── ablation_fft.py                  # 消融实验：无FFT
├── ablation_obj.py                  # 消融实验：无正交
├── ablation_sem.py                  # 消融实验：语义绑定
└── test.sh                          # 完整实验流程脚本
```

## 使用流程

### 第0步：生成 Prompts（只需执行一次）

```bash
python3 generate_prompt.py
```

这会生成 `prompts.json`，包含 200 条 prompts（100条 COCO + 100条 SD-Prompts）。

### 第1步：生成语义修改 Prompts（使用大模型）

用户需要自己使用大模型（ChatGPT/Claude）生成 `prompts_modified.json`。

**System Prompt 示例：**
```
I have a list of image generation prompts. For each prompt, please generate semantic modifications for the following attack types:

1. oil_painting: Transform into an oil painting style
2. sketch: Convert to pencil sketch style
3. watercolor: Transform into watercolor painting
4. cyberpunk: Convert to cyberpunk neon style
5. anime: Transform into anime style
6. inpaint_center: Generate an inpainting prompt for center region
7. inpaint_random: Generate an inpainting prompt for random region

Please output in JSON format.
```

参考格式见 `prompts_modified_example.json`。

### 第2步：运行实验

```bash
bash test.sh
```

或分步执行：

```bash
# 1. 生成无水印图像
python3 no_watermark_img_generate.py

# 2. 生成水印图像
python3 watermarked_img_generate.py

# 3. 生成攻击图像（会自动读取 prompts_modified.json）
python3 attack_img_generate.py

# 4. 提取水印签名
python3 extract_watermarks.py

# 5. 分析结果
python3 analyze_robustness_result.py
python3 analyze_invisibility_result.py
```

## 攻击类型

### 传统攻击
- JPEG压缩（75, 50, 30）
- 高斯模糊（0.5, 1.0, 2.0）
- 中心裁剪（0.75, 0.50）
- 加噪声（0.03, 0.05, 0.10）
- 缩放（0.75, 0.50）
- 亮度/对比度调整
- 黑色方块（中心/随机）

### SDXL 全局风格迁移（支持 prompts_modified.json）
- `sdxl_style_oil_painting`: 油画风格
- `sdxl_style_sketch`: 素描风格
- `sdxl_style_watercolor`: 水彩风格
- `sdxl_style_cyberpunk`: 赛博朋克风格
- `sdxl_style_anime`: 动漫风格

### FluxFill 局部重绘（支持 prompts_modified.json）
- `fluxfill_center`: 中心区域重绘
- `fluxfill_random`: 随机区域重绘

### SDEdit 扩散再生
- `sdedit_0.3`: 真正的扩散再生攻击（使用 FLUX 重新去噪生成）

## prompts_modified.json 格式

```json
{
  "wm_000": {
    "sdxl_style_oil_painting": {
      "original_prompt": "A cat sitting on a sofa",
      "modified_prompt": "[oil_painting] A majestic cat sitting on a velvet sofa...",
      "instruction": "transform into an oil painting style",
      "modification_type": "oil_painting"
    },
    "fluxfill_center": {
      "original_prompt": "A cat sitting on a sofa",
      "modified_prompt": "[inpaint_center] A cat sitting on a sofa (inpainted: ...)",
      "inpaint_prompt": "a beautiful scenic view through the window",
      "modification_type": "inpaint_center"
    }
  }
}
```

## 配置说明（config.json）

```json
{
  "experiment_name": "orthoflow_v1",
  "output_base_dir": "/data/daiyina/project_flux/output",
  "model_path": "/data/daiyina/project_flux/model/flux-schnell",
  "sdxl_path": "/data/daiyina/project_flux/model/sdxl-instructpix2pix",
  "flux_fill_path": "/data/daiyina/project_flux/model/flux-fill",
  "hf_cache": "/data/daiyina/project_flux/hf_cache",
  "num_inference_steps": 4,
  "guidance_scale": 0.0,
  "height": 512,
  "width": 512,
  "batch_size": 4,
  "num_workers": 4,
  "alpha": 0.6,
  "fft_radius_inner": 2,
  "fft_radius_outer": 14,
  "mask_region": [6, 26, 6, 26],
  "secret_key": 42,
  "prompt_seed": 42,
  "gpu_id": 0
}
```

**新增字段：**
- `gpu_id`: 指定使用的 GPU ID（0=第一卡，1=第二卡，以此类推）

## 注意事项

1. **prompts.json 生成后请勿修改**：这是实验的基准数据，修改后会导致结果不可复现
2. **prompts_modified.json 是可选的**：如果没有该文件，攻击会使用默认的风格迁移指令
3. **50GB 显存优化**：所有模型常驻 GPU，无需重复加载
