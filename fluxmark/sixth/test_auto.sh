#!/bin/bash
# OrthoFlow 完整实验流程脚本（全自动版）
# - 无需交互
# - 跳过 generate_prompt.py
# - no_watermark + watermarked 使用 fluxenv
# - 后续所有步骤使用 attackenv

set -e

echo "=========================================="
echo "  OrthoFlow 水印实验系统"
echo "  全自动版本"
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

# ==========================================
# 运行辅助函数（支持指定 conda 环境）
# ==========================================
run_python() {
    local script=$1
    local description=$2
    local env_name=$3
    echo "=========================================="
    echo "  $description"
    echo "  环境: $env_name"
    echo "=========================================="
    conda run -n "$env_name" --no-capture-output python3 "$script"
    if [ $? -ne 0 ]; then
        echo "❌ 错误: $description 失败"
        exit 1
    fi
    echo "✅ $description 完成"
    echo ""
}

# ==========================================
# Phase 1: 图像生成（fluxenv）
# ==========================================
echo "##########################################"
echo "# Phase 1: 图像生成阶段（fluxenv）"
echo "##########################################"
echo ""

run_python "no_watermark_img_generate.py" "生成无水印图像" "fluxenv"
run_python "watermarked_img_generate.py" "生成水印图像" "fluxenv"

# ==========================================
# Phase 2: 攻击生成（attackenv）
# ==========================================
echo "##########################################"
echo "# Phase 2: 攻击生成阶段（attackenv）"
echo "##########################################"
echo ""

run_python "attack_img_generate.py" "生成攻击图像" "attackenv"

# ==========================================
# Phase 3: 提取签名（attackenv）
# ==========================================
echo "##########################################"
echo "# Phase 3: 提取水印签名（attackenv）"
echo "##########################################"
echo ""

run_python "extract_watermarks.py" "提取水印签名" "attackenv"

# ==========================================
# Phase 4: 分析评估（attackenv）
# ==========================================
echo "##########################################"
echo "# Phase 4: 分析评估阶段（attackenv）"
echo "##########################################"
echo ""

run_python "analyze_robustness_result.py" "鲁棒性分析" "attackenv"
run_python "analyze_invisibility_result.py" "隐蔽性分析" "attackenv"

# ==========================================
# Phase 5: 消融实验（attackenv，可选但这里默认全跑）
# ==========================================
echo "##########################################"
echo "# Phase 5: 消融实验阶段（attackenv）"
echo "##########################################"
echo ""

run_python "ablation_fft.py" "消融实验 - 无FFT约束" "attackenv"
run_python "ablation_obj.py" "消融实验 - 无正交投影" "attackenv"
run_python "ablation_obj_stress.py" "消融实验 - 无正交投影压力测试" "attackenv"
run_python "ablation_sem.py" "消融实验 - 语义绑定" "attackenv"
run_python "ablation_grid.py" "消融实验 - 网格分辨率" "attackenv"
run_python "analyze_ablation.py" "消融实验 - 定量分析" "attackenv"
run_python "visualize_ablation.py" "消融实验 - 可视化对比" "attackenv"

# 汇总报告必须在所有分析（含消融实验）完成后运行
run_python "summarize_all_results.py" "结果汇总" "attackenv"

# ==========================================
# 完成总结
# ==========================================
echo ""
echo "##########################################"
echo "# 实验完成!"
echo "##########################################"
echo ""

RESULT_BASE=$(python3 -c "import json; print(json.load(open('config.json'))['output_base_dir'])")

echo "📊 关键结果文件:"
[ -f "${RESULT_BASE}/result/SUMMARY.md" ] && echo "   ✅ 汇总报告: ${RESULT_BASE}/result/SUMMARY.md"
[ -f "${RESULT_BASE}/result/merge.json" ] && echo "   ✅ 鲁棒性分析: ${RESULT_BASE}/result/merge.json"
[ -f "${RESULT_BASE}/result/invisibility_analysis.json" ] && echo "   ✅ 隐蔽性分析: ${RESULT_BASE}/result/invisibility_analysis.json"
[ -f "${RESULT_BASE}/result/ablation_analysis.json" ] && echo "   ✅ 消融定量分析: ${RESULT_BASE}/result/ablation_analysis.json"
[ -d "${RESULT_BASE}/result/ablation_viz" ] && echo "   ✅ 消融可视化: ${RESULT_BASE}/result/ablation_viz/"
[ -f "${RESULT_BASE}/result/ablation_grid/grid_ablation_summary.json" ] && echo "   ✅ 网格分辨率消融: ${RESULT_BASE}/result/ablation_grid/grid_ablation_summary.json"
echo ""

echo "🎉 所有实验步骤已完成！"
