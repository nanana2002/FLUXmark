#!/bin/bash
# OrthoFlow 完整实验流程脚本（50GB显存极致性能版）

set -e

echo "=========================================="
echo "  OrthoFlow 水印实验系统"
echo "  50GB显存极致性能版本"
echo "=========================================="
echo ""

if [ ! -f "config.json" ]; then
    echo "❌ 错误: 未找到 config.json 文件"
    exit 1
fi

EXP_NAME=$(python3 -c "import json; print(json.load(open('config.json'))['experiment_name'])")
BATCH_SIZE=$(python3 -c "import json; print(json.load(open('config.json')).get('batch_size', 4))")
echo "📝 实验名称: $EXP_NAME"
echo "🚀 批量大小: $BATCH_SIZE"
echo ""

run_python() {
    local script=$1
    local description=$2
    echo "=========================================="
    echo "  $description"
    echo "=========================================="
    python3 "$script"
    if [ $? -ne 0 ]; then
        echo "❌ 错误: $description 失败"
        exit 1
    fi
    echo "✅ $description 完成"
    echo ""
}

ask_to_run() {
    local description=$1
    read -p "是否执行: $description? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# ==========================================
# Phase 0: 生成 Prompts（只需执行一次）
# ==========================================
echo "##########################################"
echo "# Phase 0: Prompt生成（仅需执行一次）"
echo "##########################################"
echo ""

if ask_to_run "0. 生成 prompts.json (generate_prompt.py)"; then
    run_python "generate_prompt.py" "生成 Prompts"
fi

# ==========================================
# Phase 1: 图像生成
# ==========================================
echo "##########################################"
echo "# Phase 1: 图像生成阶段（高性能批量模式）"
echo "##########################################"
echo ""

if ask_to_run "1.1 生成无水印图像 (no_watermark_img_generate.py)"; then
    run_python "no_watermark_img_generate.py" "生成无水印图像"
fi

if ask_to_run "1.2 生成水印图像 (watermarked_img_generate.py)"; then
    run_python "watermarked_img_generate.py" "生成水印图像"
fi

# ==========================================
# Phase 2: 攻击生成
# ==========================================
echo "##########################################"
echo "# Phase 2: 攻击生成阶段（高性能并行模式）"
echo "##########################################"
echo ""

if ask_to_run "2. 生成攻击图像 (attack_img_generate.py)"; then
    run_python "attack_img_generate.py" "生成攻击图像"
fi

# ==========================================
# Phase 3: 提取签名
# ==========================================
echo "##########################################"
echo "# Phase 3: 提取水印签名（高性能批量模式）"
echo "##########################################"
echo ""

if ask_to_run "3.1 提取水印签名 (extract_watermarks.py)"; then
    run_python "extract_watermarks.py" "提取水印签名"
fi

# ==========================================
# Phase 4: 分析评估
# ==========================================
echo "##########################################"
echo "# Phase 4: 分析评估阶段（GPU加速）"
echo "##########################################"
echo ""

if ask_to_run "4.1 鲁棒性分析 (analyze_robustness_result.py)"; then
    run_python "analyze_robustness_result.py" "鲁棒性分析"
fi

if ask_to_run "4.2 隐蔽性分析 (analyze_invisibility_result.py)"; then
    run_python "analyze_invisibility_result.py" "隐蔽性分析"
fi

# ==========================================
# Phase 5: 消融实验（可选）
# ==========================================
echo "##########################################"
echo "# Phase 5: 消融实验阶段（可选）"
echo "##########################################"
echo ""

read -p "是否运行消融实验? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if ask_to_run "5.1 消融实验 - 无FFT约束 (ablation_fft.py)"; then
        run_python "ablation_fft.py" "消融实验 - 无FFT约束"
    fi
    if ask_to_run "5.2 消融实验 - 无正交投影 (ablation_obj.py)"; then
        run_python "ablation_obj.py" "消融实验 - 无正交投影"
    fi
    if ask_to_run "5.3 消融实验 - 语义绑定 (ablation_sem.py)"; then
        run_python "ablation_sem.py" "消融实验 - 语义绑定"
    fi
fi

# ==========================================
# 完成总结
# ==========================================
echo ""
echo "##########################################"
echo "# 实验完成!"
echo "##########################################"
echo ""

echo "📁 实验结果目录结构:"
echo ""
echo "${EXP_NAME}/"
echo "├── prompts.json                     # 初始prompts（MS-COCO + SD-Prompts）"
echo "├── prompts_modified.json            # 语义修改后的prompts（用户用大模型生成）"
echo "├── pic/"
echo "│   ├── no_watermarked_img/          # 无水印图像"
echo "│   ├── watermarked_img/             # 水印图像 + S_orig"
echo "│   ├── attack_watermarked_img/      # 攻击后图像"
echo "│   │   ├── sdxl_style_oil_painting/ # SDXL全局风格迁移：油画"
echo "│   │   ├── sdxl_style_sketch/       # SDXL全局风格迁移：素描"
echo "│   │   ├── sdxl_style_watercolor/   # SDXL全局风格迁移：水彩"
echo "│   │   ├── sdxl_style_cyberpunk/    # SDXL全局风格迁移：赛博朋克"
echo "│   │   ├── sdxl_style_anime/        # SDXL全局风格迁移：动漫"
echo "│   │   ├── fluxfill_center/         # FluxFill局部重绘：中心"
echo "│   │   ├── fluxfill_random/         # FluxFill局部重绘：随机"
echo "│   │   └── sdedit_0.3/              # SDEdit扩散再生攻击"
echo "│   ├── ablation_fft_watermarked_img/    # 消融实验：无FFT"
echo "│   ├── ablation_obj_watermarked_img/    # 消融实验：无正交"
echo "│   └── ablation_sem_watermarked_img/    # 消融实验：语义绑定"
echo "│"
echo "├── watermark_extra/                 # 预计算签名"
echo "└── result/                          # 分析结果"
echo ""

RESULT_BASE=$(python3 -c "import json; print(json.load(open('config.json'))['output_base_dir'])")

echo "📊 关键结果文件:"
[ -f "${RESULT_BASE}/result/merge.json" ] && echo "   ✅ 鲁棒性分析: ${RESULT_BASE}/result/merge.json"
[ -f "${RESULT_BASE}/result/invisibility_analysis.json" ] && echo "   ✅ 隐蔽性分析: ${RESULT_BASE}/result/invisibility_analysis.json"
echo ""

echo "🎉 所有实验步骤已完成！"
echo ""
echo "💡 50GB显存高性能特性:"
echo "   ✓ 模型常驻GPU，无需重复加载"
echo "   ✓ 批量推理 (batch_size=$BATCH_SIZE)"
echo "   ✓ SDXL和FluxFill可同时加载"
echo "   ✓ 预编码全部缓存于GPU"
echo "   ✓ 并行传统攻击处理"
echo "   ✓ GPU加速IoU/F1计算"
echo ""
echo "📝 使用 prompts_modified.json 的方法:"
echo "   1. 先生成 prompts.json (generate_prompt.py)"
echo "   2. 使用大模型（ChatGPT/Claude）为每个prompt生成语义修改版本"
echo "   3. 保存为 prompts_modified.json（参考 prompts_modified_example.json）"
echo "   4. 运行攻击生成 (attack_img_generate.py) 时会自动读取"
echo ""
