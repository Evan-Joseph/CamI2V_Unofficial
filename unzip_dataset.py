#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CamI2V 数据集终极解压脚本 (V2.0 - 智能路径版)
- 自动将 videos/ 中的 .zip 解压到 video_clips/
- 智能处理压缩包内多余的父目录，确保路径正确
- 支持进度条、错误处理和重复运行
"""

import os
import zipfile
import argparse
from pathlib import Path
from tqdm import tqdm

class Colors:
    G, Y, R, B, C, E = '\033[92m', '\033[93m', '\033[91m', '\033[94m', '\033[96m', '\033[0m'

def print_status(msg: str, status: str = "info"):
    colors = {"info": Colors.B, "success": Colors.G, "warning": Colors.Y, "error": Colors.R}
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    print(f"{colors.get(status, '')}{icons.get(status, '')} {msg}{Colors.E}")

def smart_unzip(zip_path: Path, dest_dir: Path):
    """
    智能解压单个zip文件，移除多余的父目录。
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            if member.is_dir():
                continue

            parts = Path(member.filename).parts
            
            # 找到'test'在路径中的位置，并从那里开始重构路径
            try:
                test_index = parts.index('test')
                correct_subpath = Path(*parts[test_index:])
                target_path = dest_dir / correct_subpath
                
                # 创建目标文件的父目录
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 写入文件
                with open(target_path, 'wb') as f:
                    f.write(zip_ref.read(member.filename))

            except ValueError:
                # 如果路径中没有'test'，则直接解压到根目录
                # (作为一种备用安全策略)
                zip_ref.extract(member, dest_dir)
                print_status(f"在 {member.filename} 中未找到 'test' 目录，已按常规方式解压", "warning")


def unzip_dataset(source_dir: Path, dest_dir: Path):
    print_status(f"开始扫描源目录: {source_dir}", "info")
    
    if not source_dir.exists():
        print_status(f"源目录不存在！请先运行下载脚本。", "error")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_files = list(source_dir.rglob('*.zip'))
    
    if not zip_files:
        print_status(f"未找到 .zip 文件。", "warning")
        return
        
    print_status(f"找到 {len(zip_files)} 个 .zip 文件，准备智能解压到: {dest_dir}", "info")

    success_count, skipped_count, error_count = 0, 0, 0

    with tqdm(total=len(zip_files), desc="解压进度", unit="file") as pbar:
        for zip_path in zip_files:
            video_code = zip_path.stem
            expected_output_dir = dest_dir / 'test' / video_code
            
            if expected_output_dir.exists():
                pbar.set_description(f"跳过 {video_code}")
                skipped_count += 1
                pbar.update(1)
                continue

            try:
                pbar.set_description(f"解压 {video_code}")
                smart_unzip(zip_path, dest_dir)
                success_count += 1
            except Exception as e:
                print_status(f"\n解压 {zip_path.name} 时出错: {e}", "error")
                error_count += 1
            
            pbar.update(1)
            
    print_status(f"\n{'='*15} 解压完成 {'='*15}", "info")
    print_status(f"成功解压: {success_count}", "success")
    print_status(f"跳过 (已存在): {skipped_count}", "warning")
    if error_count > 0:
        print_status(f"失败: {error_count}", "error")

def main():
    default_source = Path.cwd() / "datasets/RealEstate10K/videos"
    default_dest = Path.cwd() / "datasets/RealEstate10K/video_clips"

    parser = argparse.ArgumentParser(description="CamI2V 数据集终极解压脚本 V2.0")
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--dest", type=Path, default=default_dest)
    args = parser.parse_args()
    
    unzip_dataset(args.source, args.dest)

if __name__ == "__main__":
    main()