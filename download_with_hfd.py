#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量下载工具 - 使用hfd.sh下载RealEstate10K数据集
此脚本从CSV文件中读取zip文件名，并使用hfd.sh批量下载
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple

def read_zip_filenames(csv_file: str) -> List[str]:
    """
    从CSV文件中读取zip文件名列表
    
    参数:
        csv_file: CSV文件路径
        
    返回:
        包含zip文件名的列表
    """
    zip_filenames = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 跳过表头
            header = next(reader, None)
            
            # 确保CSV格式正确
            if not header or len(header) == 0:
                print(f"错误: CSV文件 '{csv_file}' 格式不正确。")
                return []
                
            for row in reader:
                if row and row[0]:  # 确保行不为空且有第一列
                    zip_filenames.append(row[0])
    
    except FileNotFoundError:
        print(f"错误: 未找到CSV文件 '{csv_file}'")
    except Exception as e:
        print(f"读取CSV文件时发生错误: {e}")
    
    return zip_filenames

def create_batches(items: List[str], batch_size: int) -> List[List[str]]:
    """
    将列表分割成固定大小的批次
    
    参数:
        items: 要分割的项目列表
        batch_size: 每批的最大项目数
        
    返回:
        包含多个批次的列表
    """
    return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

def execute_hfd_command(
    repo_path: str,
    zip_filenames: List[str],
    local_dir: str,
    hf_username: str,
    hf_token: str,
    is_dataset: bool = True,
    dry_run: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    执行hfd.sh命令下载指定的zip文件
    
    参数:
        repo_path: Hugging Face仓库路径，如 'MuteApo/RealCam-Vid'
        zip_filenames: 要下载的zip文件名列表
        local_dir: 本地保存目录
        hf_username: Hugging Face用户名
        hf_token: Hugging Face令牌
        is_dataset: 是否为数据集（默认为True）
        dry_run: 如果为True，只打印命令而不执行
        
    返回:
        (成功标志, 错误消息)
    """
    # 构建基础命令
    cmd = ["./hfd.sh", repo_path]
    
    # 添加数据集标志
    if is_dataset:
        cmd.append("--dataset")
    
    # 添加每个zip文件的--include参数
    for zip_name in zip_filenames:
        cmd.extend(["--include", f"{zip_name}"])
    
    # 添加本地目录
    cmd.extend(["--local-dir", local_dir])
    
    # 添加认证信息
    cmd.extend(["--hf_username", hf_username])
    cmd.extend(["--hf_token", hf_token])
    
    # 执行命令
    command_str = " ".join(cmd)
    if dry_run:
        print(f"将要执行的命令: {command_str}")
        return True, None
    
    print(f"正在执行命令: {command_str}")
    try:
        # 使用subprocess.run执行命令，并捕获输出
        result = subprocess.run(
            cmd, 
            check=True,
            text=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return False, f"命令执行失败，返回码: {result.returncode}, 错误: {result.stderr}"
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"命令执行出错: {e}, 错误输出: {e.stderr}"
    except Exception as e:
        return False, f"执行过程中发生未知错误: {e}"

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用hfd.sh批量下载数据集文件")
    parser.add_argument("--csv", default="realestate10k_test_zips_for_hfd.csv",
                      help="包含zip文件名的CSV文件 (默认: realestate10k_test_zips_for_hfd.csv)")
    parser.add_argument("--repo-path", default="MuteApo/RealCam-Vid",
                      help="Hugging Face仓库路径 (默认: MuteApo/RealCam-Vid)")
    parser.add_argument("--local-dir", default="./RealCam-Vid-Test",
                      help="本地保存目录 (默认: ./RealCam-Vid-Test)")
    parser.add_argument("--username", default="EvanSirius",
                      help="Hugging Face用户名 (默认: EvanSirius)")
    parser.add_argument("--token", default="hf_TOKEN",
                      help="Hugging Face令牌 (默认: hf_TOKEN)")
    parser.add_argument("--batch-size", type=int, default=10,
                      help="每批处理的文件数量 (默认: 10)")
    parser.add_argument("--no-dataset", action="store_true",
                      help="如果目标不是数据集，添加此参数")
    parser.add_argument("--dry-run", action="store_true",
                      help="只打印命令不执行")
    parser.add_argument("--retry-failed", action="store_true",
                      help="自动重试失败的批次")
    parser.add_argument("--max-retries", type=int, default=3,
                      help="最大重试次数 (默认: 3)")
    parser.add_argument("--retry-delay", type=int, default=5,
                      help="重试间隔（秒）(默认: 5)")
    parser.add_argument("--start-batch", type=int, default=0,
                      help="起始批次索引 (默认: 0)")
    parser.add_argument("--end-batch", type=int, default=None,
                      help="结束批次索引 (默认: 处理所有批次)")
    
    args = parser.parse_args()
    
    # 检查hfd.sh是否存在且可执行
    if not os.path.isfile("./hfd.sh"):
        print("错误: 未找到 './hfd.sh' 文件。请确保脚本在当前目录中且有执行权限。")
        return 1
    
    if not os.access("./hfd.sh", os.X_OK):
        print("警告: 'hfd.sh' 可能没有执行权限。尝试添加执行权限...")
        try:
            os.chmod("./hfd.sh", 0o755)  # 添加执行权限
            print("已成功添加执行权限。")
        except Exception as e:
            print(f"无法添加执行权限: {e}")
            print("请手动运行: chmod +x ./hfd.sh")
            return 1
    
    # 读取CSV文件中的zip文件名
    zip_filenames = read_zip_filenames(args.csv)
    if not zip_filenames:
        print(f"错误: 未能从 '{args.csv}' 读取到任何zip文件名或文件为空。")
        return 1
    
    print(f"从 '{args.csv}' 成功读取了 {len(zip_filenames)} 个zip文件名。")
    
    # 将zip文件名分批
    batches = create_batches(zip_filenames, args.batch_size)
    total_batches = len(batches)
    
    print(f"已将 {len(zip_filenames)} 个zip文件分成 {total_batches} 批（每批最多 {args.batch_size} 个文件）")
    
    # 确定处理的批次范围
    start_batch_idx = max(0, min(args.start_batch, total_batches - 1))
    end_batch_idx = total_batches if args.end_batch is None else min(args.end_batch, total_batches)
    
    if start_batch_idx >= end_batch_idx:
        print(f"错误: 起始批次索引 ({start_batch_idx}) 大于或等于结束批次索引 ({end_batch_idx})")
        return 1
    
    print(f"将处理第 {start_batch_idx} 到 {end_batch_idx-1} 批，共 {end_batch_idx-start_batch_idx} 批")
    
    # 下载状态追踪
    successful_batches = []
    failed_batches = []
    
    # 处理每一批
    for batch_idx in range(start_batch_idx, end_batch_idx):
        batch = batches[batch_idx]
        print(f"\n========== 处理第 {batch_idx+1}/{total_batches} 批 ({len(batch)} 个文件) ==========")
        print(f"当前批次文件: {', '.join(batch)}")
        
        retries = 0
        success = False
        error_msg = None
        
        # 如果设置了重试，则尝试多次下载
        while not success and retries <= args.max_retries:
            if retries > 0:
                print(f"重试 #{retries}/{args.max_retries}... (等待 {args.retry_delay} 秒)")
                time.sleep(args.retry_delay)
                
            success, error_msg = execute_hfd_command(
                repo_path=args.repo_path,
                zip_filenames=batch,
                local_dir=args.local_dir,
                hf_username=args.username,
                hf_token=args.token,
                is_dataset=not args.no_dataset,
                dry_run=args.dry_run
            )
            
            if success:
                print(f"批次 #{batch_idx+1} 下载成功！")
                successful_batches.append(batch_idx)
                break
                
            retries += 1
            if not args.retry_failed:
                break
                
        if not success:
            print(f"批次 #{batch_idx+1} 下载失败。错误: {error_msg}")
            failed_batches.append(batch_idx)
    
    # 打印总结报告
    print("\n========== 下载任务总结 ==========")
    print(f"总批次数: {end_batch_idx - start_batch_idx}")
    print(f"成功批次数: {len(successful_batches)}")
    print(f"失败批次数: {len(failed_batches)}")
    
    if failed_batches:
        print("\n失败的批次:")
        for batch_idx in failed_batches:
            print(f"批次 #{batch_idx+1} - 文件: {', '.join(batches[batch_idx])}")
        print("\n要仅重试失败的批次，请使用命令:")
        retry_batch_ranges = []
        current_start = None
        current_end = None
        
        # 转换失败批次列表为连续范围
        for idx in sorted(failed_batches):
            if current_start is None:
                current_start = idx
                current_end = idx
            elif idx == current_end + 1:
                current_end = idx
            else:
                retry_batch_ranges.append((current_start, current_end))
                current_start = idx
                current_end = idx
        
        if current_start is not None:
            retry_batch_ranges.append((current_start, current_end))
        
        for start, end in retry_batch_ranges:
            if start == end:
                print(f"python batch_download_with_hfd.py --start-batch {start} --end-batch {start+1}")
            else:
                print(f"python batch_download_with_hfd.py --start-batch {start} --end-batch {end+1}")
    
    return 0 if not failed_batches else 1

if __name__ == "__main__":
    sys.exit(main())