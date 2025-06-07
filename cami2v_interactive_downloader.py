#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CamI2V 终极下载脚本 (V8.0 - The Pragmatist's Edition)
- 恢复高效的批处理下载，平衡速度与可靠性
- 采用通配符路径匹配，终极解决特殊文件名问题
- 失败重试和日志记录在批次级别工作
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import time

# --- 配置信息 ---
REPOS = {
    "models": {
        "repo": "MuteApo/CamI2V",
        "desc": "CamI2V模型检查点",
        "files": ["256_cami2v.pt", "512_cami2v_50k.pt", "512_cami2v_100k.pt", 
                 "256_cameractrl.pt", "256_motionctrl.pt"],
        "path": "ckpts"
    },
    "base_256": {
        "repo": "Doubiiu/DynamiCrafter", 
        "desc": "DynamiCrafter基础模型(256)",
        "files": ["model.ckpt"],
        "path": "pretrained_models/DynamiCrafter"
    },
    "base_512": {
        "repo": "Doubiiu/DynamiCrafter_512",
        "desc": "DynamiCrafter基础模型(512)", 
        "files": ["model.ckpt"],
        "path": "pretrained_models/DynamiCrafter_512"
    },
    "dataset": {
        "repo": "MuteApo/RealCam-Vid",
        "desc": "RealEstate10K视频数据集",
        "files": [],  # 从CSV动态加载
        "path": "datasets/RealEstate10K/videos",
        "is_dataset": True
    }
}

# 下载参数
BATCH_SIZE = 10
MAX_RETRIES = 3
RETRY_DELAY = 5 # 秒
FAILED_LOG_FILE = "failed_downloads.log"

class Colors:
    G, Y, R, B, C, E = '\033[92m', '\033[93m', '\033[91m', '\033[94m', '\033[96m', '\033[0m'

def print_status(msg: str, status: str = "info"):
    colors = {"info": Colors.B, "success": Colors.G, "warning": Colors.Y, "error": Colors.R}
    icons = {"info": "ℹ️", "success": "✅", "warning": "⚠️", "error": "❌"}
    print(f"{colors.get(status, '')}{icons.get(status, '')} {msg}{Colors.E}")

def setup_directories():
    base = Path.cwd()
    for key, info in REPOS.items():
        Path(info['path']).mkdir(parents=True, exist_ok=True)
    print_status("项目目录结构已就绪", "success")

def find_hfd() -> Optional[Path]:
    locations = [Path("./hfd.sh"), Path.home() / ".local/bin/hfd.sh"]
    try:
        result = subprocess.run(["which", "hfd.sh"], capture_output=True, text=True, check=False)
        if result.returncode == 0: locations.append(Path(result.stdout.strip()))
    except FileNotFoundError: pass

    for path in locations:
        if path.exists() and os.access(path, os.X_OK): return path.resolve()
    
    local_hfd = Path("./hfd.sh")
    if local_hfd.exists() and not os.access(local_hfd, os.X_OK):
        try:
            os.chmod(local_hfd, 0o755)
            print_status("已为 ./hfd.sh 添加执行权限", "success")
            return local_hfd.resolve()
        except OSError as e:
            print_status(f"无法为 ./hfd.sh 添加权限: {e}", "error")
    return None

def _execute_hfd_batch(repo: str, files: List[str], local_dir: Path, is_dataset: bool, token: str) -> bool:
    """执行单批hfd下载命令"""
    hfd_path = find_hfd()
    if not hfd_path:
        print_status("未找到hfd.sh脚本，任务中止", "error")
        return False

    cmd = [str(hfd_path), repo]
    if is_dataset: cmd.append("--dataset")
    
    # 恢复批处理，但对每个文件使用“黄金格式”
    cmd.append("--include")
    for file in files:
        filename_with_zip = file if file.endswith('.zip') else f"{file}.zip"
        glob_pattern = f"*/{filename_with_zip}"
        cmd.append(glob_pattern)
    
    cmd.extend(["--local-dir", str(local_dir)])
    if token: cmd.extend(["--token", token])

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        for line in iter(process.stdout.readline, ''):
            print(f"  {Colors.C}|> {line.strip()}{Colors.E}")
        
        process.wait()
        
        if process.returncode == 0:
            return True
        else:
            print_status(f"hfd.sh 报告失败，退出码: {process.returncode}", "error")
            return False
            
    except Exception as e:
        print_status(f"执行hfd.sh时发生严重错误: {e}", "error")
        return False

def download_manager(repo_info: Dict, files_to_download: List[str], token: str):
    """下载任务管理器，恢复批处理和重试"""
    repo = repo_info["repo"]
    local_dir = Path(repo_info["path"])
    is_dataset = repo_info.get("is_dataset", False)
    
    batches = [files_to_download[i:i + BATCH_SIZE] for i in range(0, len(files_to_download), BATCH_SIZE)] if len(files_to_download) > BATCH_SIZE else [files_to_download]
    
    total_batches = len(batches)
    failed_batches_info = []

    print_status(f"开始下载任务: {repo_info['desc']}", "info")
    print_status(f"总文件数: {len(files_to_download)}, 分为 {total_batches} 个批次", "info")

    for i, batch_files in enumerate(batches):
        # 检查是否该批次所有文件都已存在
        if all((local_dir / (f if f.endswith('.zip') else f"{f}.zip")).exists() for f in batch_files):
            print(f"\n{Colors.Y}--- 跳过批次 {i+1}/{total_batches} (所有文件已存在) ---{Colors.E}")
            continue

        print(f"\n{Colors.Y}--- 处理批次 {i+1}/{total_batches} ({len(batch_files)}个文件) ---{Colors.E}")
        
        retries = 0
        success = False
        while retries <= MAX_RETRIES:
            if retries > 0:
                print_status(f"批次失败。第 {retries} 次重试... ({RETRY_DELAY}秒后)", "warning")
                time.sleep(RETRY_DELAY)
            
            success = _execute_hfd_batch(repo, batch_files, local_dir, is_dataset, token)
            if success:
                print_status(f"批次 {i+1} 下载成功", "success")
                break
            
            retries += 1

        if not success:
            print_status(f"批次 {i+1} 最终下载失败。", "error")
            failed_batches_info.append(batch_files)

    print(f"\n{Colors.B}{'='*20} 任务总结 {'='*20}{Colors.E}")
    if failed_batches_info:
        print_status(f"有 {len(failed_batches_info)} 个批次下载失败，已记录到日志中。", "warning")
        with open(FAILED_LOG_FILE, "a") as f:
            for batch in failed_batches_info:
                for filename in batch:
                    f.write(f"{repo_info['repo']},{filename},{repo_info['path']}\n")
    else:
        print_status("所有任务均已成功完成！", "success")

def load_dataset_files() -> List[str]:
    files = []
    for csv_file in ["realestate10k_test_zips_for_hfd.csv", "realestate10k_train_zips_for_hfd.csv"]:
        if Path(csv_file).exists():
            try:
                with open(csv_file, 'r') as f:
                    next(f, None)
                    files.extend(line.strip().split(',')[0] for line in f if line.strip())
            except Exception as e:
                print_status(f"读取 {csv_file} 失败: {e}", "warning")
    return files

def retry_from_log(token: str):
    log_path = Path(FAILED_LOG_FILE)
    if not log_path.exists():
        print_status("未找到失败日志，无需重试。", "success")
        return

    print_status(f"从 {FAILED_LOG_FILE} 读取失败记录...", "info")
    
    tasks = {}
    with open(log_path, 'r') as f:
        unique_lines = set(f.readlines())
        for line in unique_lines:
            try:
                repo, filename, path = line.strip().split(',')
                if repo not in tasks: tasks[repo] = {"files": []}
                tasks[repo]["files"].append(filename)
            except ValueError:
                print_status(f"忽略格式错误的日志行: {line.strip()}", "warning")
    
    if not tasks:
        print_status("日志文件为空或格式错误。", "warning")
        os.remove(log_path)
        return
        
    os.remove(log_path)

    for repo, task_info in tasks.items():
        repo_key_found = next((key for key, info in REPOS.items() if info["repo"] == repo), None)
        if repo_key_found:
            download_manager(REPOS[repo_key_found], task_info["files"], token)
        else:
            print_status(f"未在配置中找到仓库 {repo}，无法重试", "error")

def show_menu() -> str:
    print(f"\n{Colors.C}{'='*50}\n  CamI2V 终极下载工具 V8.0\n{'='*50}{Colors.E}")
    menu_items = [
        ("1", "CamI2V模型检查点", "models"),
        ("2", "基础模型(256分辨率)", "base_256"),
        ("3", "基础模型(512分辨率)", "base_512"),
        ("4", "视频数据集 (平衡版)", "dataset"),
        ("5", "一键下载所有模型", "all_models"),
        ("6", "从日志重试失败的下载", "retry"),
        ("0", "退出", "exit")
    ]
    for key, desc, _ in menu_items: print(f"  {key}. {desc}")
    while True:
        choice = input(f"\n{Colors.C}请选择操作: {Colors.E}").strip()
        for key, _, action in menu_items:
            if choice == key: return action
        print_status("无效选择", "warning")

def main():
    parser = argparse.ArgumentParser(description="CamI2V 终极下载脚本 V8.0 - The Pragmatist's Edition")
    parser.add_argument("--token", default="", help="HuggingFace token")
    parser.add_argument("--auto-models", action="store_true", help="自动下载所有模型")
    args = parser.parse_args()
    
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    setup_directories()
    
    if args.auto_models:
        print_status("自动下载所有模型...", "info")
        for key in ["models", "base_256", "base_512"]:
            download_manager(REPOS[key], REPOS[key]["files"], args.token)
        return

    try:
        while True:
            action = show_menu()
            if action == "exit": break
            if action == "retry":
                retry_from_log(args.token)
            elif action == "all_models":
                if input(f"{Colors.Y}确认下载所有模型? (y/N): {Colors.E}").lower() == 'y':
                    for key in ["models", "base_256", "base_512"]:
                        download_manager(REPOS[key], REPOS[key]["files"], args.token)
            elif action in REPOS:
                info = REPOS[action]
                files = info["files"] if action != "dataset" else load_dataset_files()
                if not files:
                    print_status("未找到可下载的文件。", "warning")
                else:
                    if input(f"{Colors.Y}找到 {len(files)} 个文件。确认下载? (y/N): {Colors.E}").lower() == 'y':
                        download_manager(info, files, args.token)
            input(f"\n{Colors.C}按回车返回主菜单...{Colors.E}")
            
    except KeyboardInterrupt:
        print_status("\n程序被用户中断", "warning")
    except Exception as e:
        print_status(f"程序出现未知错误: {e}", "error")

if __name__ == "__main__":
    main()