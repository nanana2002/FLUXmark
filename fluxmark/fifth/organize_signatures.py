#!/usr/bin/env python3
"""
整理 watermark_extra 中的 S_tamp 签名文件：
1. 按攻击类型分文件夹存放
2. 支持精准删除指定攻击类型的所有签名

用法：
  python3 organize_signatures.py
    # 默认：只整理，不分文件夹（会提示）

  python3 organize_signatures.py --organize
    # 按攻击类型分文件夹整理

  python3 organize_signatures.py --delete crop_0.50
    # 删除所有 crop_0.50 的 S_tamp 签名（整理后从子文件夹删，或从根目录删）

  python3 organize_signatures.py --delete crop_0.50 --organize
    # 先整理到子文件夹，再删除指定子文件夹里的内容

  python3 organize_signatures.py --dry-run --delete crop_0.50
    # 只预览要删除哪些文件，不真正执行
"""

import os
import sys
import shutil
import glob
import re

def get_watermark_extra_dir():
    config_path = 'config.json'
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        return os.path.join(config['output_base_dir'], 'watermark_extra')
    return 'watermark_extra'

def parse_attack_name(filename):
    """从文件名解析攻击类型，例如 wm_001_crop_0.50_S_tamp.npy -> crop_0.50"""
    # 去掉前缀 wm_xxx_，后缀 _S_tamp.npy
    core = filename.replace('_S_tamp.npy', '')
    # 找到第一个下划线后的攻击名
    parts = core.split('_', 2)
    if len(parts) >= 3:
        return parts[2]
    return None

def organize_signatures(base_dir, dry_run=False):
    """把根目录的 S_tamp 签名按攻击类型分文件夹"""
    files = [f for f in os.listdir(base_dir) if f.endswith('_S_tamp.npy')]
    if not files:
        print(f"   没有找到需要整理的 S_tamp 文件")
        return

    moved = 0
    for f in files:
        attack_name = parse_attack_name(f)
        if not attack_name:
            continue
        target_dir = os.path.join(base_dir, attack_name)
        src = os.path.join(base_dir, f)
        dst = os.path.join(target_dir, f)
        if dry_run:
            print(f"   [DRY-RUN] 移动: {f} -> {attack_name}/")
        else:
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(src, dst)
        moved += 1

    print(f"   {'[DRY-RUN] ' if dry_run else ''}整理了 {moved} 个 S_tamp 签名文件")

def delete_attack_signatures(base_dir, attack_name, dry_run=False):
    """删除指定攻击类型的所有 S_tamp 签名"""
    # 先尝试从子文件夹找
    subdir = os.path.join(base_dir, attack_name)
    patterns = []
    if os.path.isdir(subdir):
        patterns.append(os.path.join(subdir, f"*_{attack_name}_S_tamp.npy"))
    # 同时也扫描根目录（兼容未整理的情况）
    patterns.append(os.path.join(base_dir, f"*_{attack_name}_S_tamp.npy"))

    files_to_delete = []
    for p in patterns:
        files_to_delete.extend(glob.glob(p))

    files_to_delete = sorted(set(files_to_delete))

    if not files_to_delete:
        print(f"   未找到任何 {attack_name} 的 S_tamp 签名文件")
        return

    for f in files_to_delete:
        if dry_run:
            print(f"   [DRY-RUN] 删除: {f}")
        else:
            os.remove(f)
            print(f"   已删除: {os.path.basename(f)}")

    print(f"   {'[DRY-RUN] ' if dry_run else ''}共处理 {len(files_to_delete)} 个 {attack_name} 签名文件")

def main():
    base_dir = get_watermark_extra_dir()
    print(f"水印签名目录: {base_dir}")

    args = sys.argv[1:]
    dry_run = '--dry-run' in args
    do_organize = '--organize' in args

    # 解析 --delete 参数
    delete_attack = None
    if '--delete' in args:
        idx = args.index('--delete')
        if idx + 1 < len(args):
            delete_attack = args[idx + 1]
        else:
            print("❌ --delete 后面需要跟攻击类型名称，例如: --delete crop_0.50")
            sys.exit(1)

    if dry_run:
        print("\n⚠️  当前是 DRY-RUN 模式，不会真正修改文件\n")

    if do_organize:
        print("\n📁 开始整理 S_tamp 签名到子文件夹...")
        organize_signatures(base_dir, dry_run=dry_run)

    if delete_attack:
        print(f"\n🗑️  开始删除 {delete_attack} 的签名...")
        delete_attack_signatures(base_dir, delete_attack, dry_run=dry_run)

    if not do_organize and not delete_attack:
        print("\n当前状态统计:")
        all_files = glob.glob(os.path.join(base_dir, '*_S_tamp.npy'))
        print(f"   根目录 S_tamp 文件: {len(all_files)} 个")
        subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        for d in sorted(subdirs):
            count = len(glob.glob(os.path.join(base_dir, d, '*_S_tamp.npy')))
            if count > 0:
                print(f"   子目录 {d}/: {count} 个")
        print("\n💡 提示:")
        print("   整理分文件夹: python3 organize_signatures.py --organize")
        print("   删除指定攻击: python3 organize_signatures.py --delete crop_0.50")
        print("   整理+删除:    python3 organize_signatures.py --organize --delete crop_0.50")
        print("   预览不执行:   python3 organize_signatures.py --dry-run --delete crop_0.50")

if __name__ == '__main__':
    main()
