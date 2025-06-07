#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CamI2V 数据集终极解压脚本 (V1.0)
- 自动将 videos/ 中的 .zip 解压到 video_clips/
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

def unzip_dataset(source_dir: Path, dest_dir: Path):
    """
    主解压函数
    """
    print_status(f"开始扫描源目录: {source_dir}", "info")
    
    if not source_dir.exists():
        print_status(f"源目录不存在！请先运行下载脚本下载数据集。", "error")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    
    zip_files = list(source_dir.rglob('*.zip'))
    
    if not zip_files:
        print_status(f"在源目录中未找到任何 .zip 文件。", "warning")
        return
        
    print_status(f"共找到 {len(zip_files)} 个 .zip 文件，准备解压到: {dest_dir}", "info")

    success_count = 0
    skipped_count = 0
    error_count = 0

    with tqdm(total=len(zip_files), desc="解压进度", unit="file") as pbar:
        for zip_path in zip_files:
            video_code = zip_path.stem  # 获取不带扩展名的文件名，即视频代码
            
            # 智能跳过逻辑：检查解压后的标志性目录是否存在
            # 假设每个zip解压后都会在目标目录创建 test/{video_code} 这样的结构
            expected_output_dir = dest_dir / 'test' / video_code
            if expected_output_dir.exists():
                pbar.set_description(f"跳过 {video_code}")
                skipped_count += 1
                pbar.update(1)
                continue

            try:
                pbar.set_description(f"正在解压 {video_code}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dest_dir)
                success_count += 1
            except zipfile.BadZipFile:
                print_status(f"\n文件损坏，无法解压: {zip_path.name}", "error")
                error_count += 1
            except Exception as e:
                print_status(f"\n解压 {zip_path.name} 时发生未知错误: {e}", "error")
                error_count += 1
            
            pbar.update(1)
            
    print_status(f"\n{'='*15} 解压完成 {'='*15}", "info")
    print_status(f"成功解压: {success_count} 个文件", "success")
    print_status(f"跳过 (已存在): {skipped_count} 个文件", "warning")
    if error_count > 0:
        print_status(f"失败: {error_count} 个文件", "error")

def main():
    default_source = Path.cwd() / "datasets/RealEstate10K/videos"
    default_dest = Path.cwd() / "datasets/RealEstate10K/video_clips"

    parser = argparse.ArgumentParser(description="CamI2V 数据集终极解压脚本 V1.0")
    parser.add_argument("--source", type=Path, default=default_source, help=f"包含.zip文件的源目录 (默认: {default_source})")
    parser.add_argument("--dest", type=Path, default=default_dest, help=f"解压文件的目标目录 (默认: {default_dest})")
    args = parser.parse_args()
    
    unzip_dataset(args.source, args.dest)

if __name__ == "__main__":
    main()