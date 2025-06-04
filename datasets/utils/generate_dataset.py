# 优化版本 - 基于原始脚本进行多项改进
# Original from: https://github.com/cashiwamochi/RealEstate10K_Downloader/blob/master/generate_dataset.py

import concurrent.futures
import glob
import json
import logging
import os
import pickle
import shutil
import signal
import threading
import time
from argparse import ArgumentParser
from pathlib import Path
from time import sleep
from typing import List, Optional, Dict, Any
from uuid import uuid4

from pytubefix import YouTube
from pytubefix.cli import on_progress
from pytubefix.streams import Stream
from tqdm import tqdm


class Data:
    """保持原始数据结构不变"""
    def __init__(self, url: str, seqname: str, list_timestamps: List[str]):
        self.url: str = url
        self.list_seqnames: List[str] = []
        self.list_list_timestamps: List[List[str]] = []

        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def add(self, seqname: str, list_timestamps: List[str]):
        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def __len__(self):
        return len(self.list_seqnames)


class DownloadTimeoutError(Exception):
    """下载超时异常"""
    pass


class ProgressTracker:
    """下载进度跟踪器，用于检测停滞状态"""
    
    def __init__(self, timeout_seconds: int = 60):
        self.timeout_seconds = timeout_seconds
        self.last_update_time = time.time()
        self.last_bytes_downloaded = 0
        self.lock = threading.Lock()
        self.is_stalled = False
        
    def update(self, bytes_downloaded: int):
        """更新下载进度"""
        with self.lock:
            current_time = time.time()
            
            # 如果有新的字节下载，重置状态
            if bytes_downloaded > self.last_bytes_downloaded:
                self.last_update_time = current_time
                self.last_bytes_downloaded = bytes_downloaded
                self.is_stalled = False
            elif current_time - self.last_update_time > self.timeout_seconds:
                # 超时无进度更新，标记为停滞
                self.is_stalled = True
                
    def check_stalled(self) -> bool:
        """检查是否停滞"""
        with self.lock:
            return self.is_stalled
            
    def reset(self):
        """重置跟踪器"""
        with self.lock:
            self.last_update_time = time.time()
            self.last_bytes_downloaded = 0
            self.is_stalled = False


class OptimizedDataDownloader:
    """优化版本的数据下载器，增强的重试和超时检测机制"""
    
    def __init__(self, dataroot: str, split: str, max_workers: int = 3, retry_attempts: int = 3, 
                 download_timeout: int = 300, stall_timeout: int = 60):
        self.dataroot = Path(dataroot)
        self.split = split
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.download_timeout = download_timeout  # 单个视频最大下载时间（秒）
        self.stall_timeout = stall_timeout  # 停滞检测超时时间（秒）
        self.output_root = self.dataroot / "videos" / split
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化数据结构
        self.list_data_pkl = self.dataroot / f"{split}_list_data.pkl"
        self.list_seqnames = sorted(glob.glob(str(self.dataroot / "pose_files" / split / "*.txt")))
        
        # 线程安全的计数器和锁
        self.download_lock = threading.Lock()
        self.success_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.timeout_count = 0  # 超时计数
        
        print(f"[INFO] 正在加载数据列表... ", end="", flush=True)
        self.list_data = self.prepare_list_data()
        self.list_data.reverse()  # 保持原始行为
        
        print(" 完成!")
        print(f"[INFO] {split} 模式下将处理 {len(self.list_data)} 个视频")
        print(f"[INFO] 下载超时设置: {download_timeout}秒, 停滞检测: {stall_timeout}秒")
        
        # 创建状态文件
        self.progress_file = self.dataroot / f"download_progress_{split}.json"
        self.failed_videos_file = self.dataroot / f"failed_videos_{split}.txt"
        
    def _setup_logging(self):
        """设置日志配置"""
        log_file = self.dataroot / f"download_{self.split}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def prepare_list_data(self) -> List[Data]:
        """保持原始逻辑的数据准备方法，添加错误处理"""
        if self.list_data_pkl.exists():
            try:
                with open(self.list_data_pkl, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"加载缓存文件失败: {e}，重新生成...")

        list_data: List[Data] = []
        
        for txt_file in tqdm(self.list_seqnames, desc="加载元数据"):
            try:
                dir_name = Path(txt_file).name
                seq_name = dir_name.split(".")[0]

                # 从txt文件提取信息
                with open(txt_file, "r", encoding='utf-8') as seq_file:
                    lines = seq_file.readlines()
                
                if not lines:
                    self.logger.warning(f"空文件: {txt_file}")
                    continue
                    
                youtube_url = lines[0].strip()
                list_timestamps = []
                
                for line in lines[1:]:
                    try:
                        timestamp = int(line.split(" ")[0])
                        list_timestamps.append(timestamp)
                    except (ValueError, IndexError):
                        continue

                # 检查URL是否已注册
                is_registered = False
                for data in list_data:
                    if youtube_url == data.url:
                        is_registered = True
                        data.add(seq_name, list_timestamps)
                        break

                if not is_registered:
                    list_data.append(Data(youtube_url, seq_name, list_timestamps))
                    
            except Exception as e:
                self.logger.error(f"处理文件 {txt_file} 时出错: {e}")
                continue

        # 保存缓存
        try:
            with open(self.list_data_pkl, "wb") as f:
                pickle.dump(list_data, f)
        except Exception as e:
            self.logger.warning(f"保存缓存文件失败: {e}")
        
        return list_data

    def _create_enhanced_youtube(self, url: str, progress_tracker: ProgressTracker, video_id: str):
        """创建带有停滞检测的YouTube对象"""
        def enhanced_progress_callback(stream, chunk, bytes_remaining):
            try:
                # 首先调用原始进度回调，保持正常的下载机制
                on_progress(stream, chunk, bytes_remaining)
                
                # 更新我们的进度跟踪器
                bytes_downloaded = stream.filesize - bytes_remaining
                progress_tracker.update(bytes_downloaded)
                
                # 检查是否停滞（非阻塞检查）
                if progress_tracker.check_stalled():
                    self.logger.warning(f"[{video_id}] 检测到下载停滞")
                    # 注意：这里不直接抛出异常，因为会中断下载流程
                    # 而是让外部超时机制处理
                
            except Exception as e:
                # 只记录日志，不中断下载流程
                self.logger.debug(f"进度回调处理异常: {e}")
                
        # 创建YouTube对象，使用增强的进度回调
        return YouTube(url, use_oauth=True, on_progress_callback=enhanced_progress_callback)

    def _download_with_timeout(self, data: Data, tmp_path: Path, video_id: str) -> bool:
        """带超时和停滞检测的下载函数"""
        progress_tracker = ProgressTracker(timeout_seconds=self.stall_timeout)
        download_completed = threading.Event()
        exception_holder = [None]
        
        def download_worker():
            try:
                # 创建带有停滞检测的YouTube对象
                yt = self._create_enhanced_youtube(data.url, progress_tracker, video_id)
                
                stream: Stream = yt.streams.filter().order_by("resolution").last()
                if stream is None:
                    raise Exception("未找到可用的视频流")
                
                # 执行下载
                stream.download(output_path=str(tmp_path.parent), filename=tmp_path.name)
                download_completed.set()
                    
            except Exception as e:
                exception_holder[0] = e
                download_completed.set()
        
        # 启动下载线程
        download_thread = threading.Thread(target=download_worker)
        download_thread.daemon = True
        download_thread.start()
        
        # 监控下载进度，检测停滞和超时
        start_time = time.time()
        while download_thread.is_alive():
            # 检查总超时
            if time.time() - start_time > self.download_timeout:
                self.logger.warning(f"[{video_id}] 下载总超时 ({self.download_timeout}秒)")
                raise DownloadTimeoutError(f"下载总超时 ({self.download_timeout}秒)")
            
            # 检查停滞
            if progress_tracker.check_stalled():
                self.logger.warning(f"[{video_id}] 检测到下载停滞超过 {self.stall_timeout} 秒")
                raise DownloadTimeoutError(f"下载停滞超过 {self.stall_timeout} 秒")
            
            # 短暂等待，避免CPU占用过高
            time.sleep(1)
        
        # 等待下载线程完全结束
        download_thread.join(timeout=5)
        
        # 检查下载过程中的异常
        if exception_holder[0]:
            raise exception_holder[0]
            
        return tmp_path.exists()

    def _download_single_video(self, data: Data, global_count: int) -> bool:
        """单个视频下载函数，修复后的版本"""
        video_id = data.url.split('=')[-1]
        filepath = self.output_root / f"{video_id}.mp4"
        
        # 检查文件是否已存在
        if filepath.exists():
            with self.download_lock:
                self.skipped_count += 1
            return True

        last_exception = None
        
        for attempt in range(self.retry_attempts):
            tmpname = f"re10k_{uuid4().fields[0]:x}.mp4"
            tmp_path = Path("/tmp") / tmpname
            
            try:
                self.logger.info(f"[{global_count:04d}] 正在下载 (尝试 {attempt + 1}/{self.retry_attempts}): {data.url}")
                
                # 使用带超时和停滞检测的下载
                if self._download_with_timeout(data, tmp_path, video_id):
                    # 原子性移动文件
                    shutil.move(str(tmp_path), str(filepath))
                    
                    with self.download_lock:
                        self.success_count += 1
                    
                    self.logger.info(f"[{global_count:04d}] 下载完成: {video_id}")
                    return True
                else:
                    raise Exception("下载的临时文件不存在")
                    
            except DownloadTimeoutError as e:
                last_exception = e
                with self.download_lock:
                    self.timeout_count += 1
                self.logger.warning(f"[{global_count:04d}] 尝试 {attempt + 1} 超时: {e}")
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"[{global_count:04d}] 尝试 {attempt + 1} 失败: {e}")
            
            finally:
                # 清理临时文件
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception as cleanup_error:
                        self.logger.debug(f"清理临时文件失败: {cleanup_error}")
            
            # 重试前的等待策略
            if attempt < self.retry_attempts - 1:
                # 根据失败类型调整等待时间
                if isinstance(last_exception, DownloadTimeoutError):
                    sleep_time = min((attempt + 1) * 5, 30)  # 超时错误等待更长时间
                else:
                    sleep_time = (attempt + 1) * 2  # 其他错误使用指数退避
                
                self.logger.info(f"[{global_count:04d}] 等待 {sleep_time} 秒后重试...")
                sleep(sleep_time)

        # 所有重试均失败
        with open(self.failed_videos_file, "a", encoding='utf-8') as f:
            f.write(f"{filepath.name}\t{str(last_exception)}\n")
        
        with self.download_lock:
            self.failed_count += 1
        
        self.logger.error(f"[{global_count:04d}] 所有重试均失败: {data.url}, 最后错误: {last_exception}")
        return False

    def _download_single_video_simple(self, data: Data, global_count: int) -> bool:
        """简化版本的下载函数，回退到接近原始逻辑"""
        video_id = data.url.split('=')[-1]
        filepath = self.output_root / f"{video_id}.mp4"
        
        # 检查文件是否已存在
        if filepath.exists():
            with self.download_lock:
                self.skipped_count += 1
            return True

        last_exception = None
        
        for attempt in range(self.retry_attempts):
            tmpname = f"re10k_{uuid4().fields[0]:x}.mp4"
            tmp_path = Path("/tmp") / tmpname
            
            try:
                self.logger.info(f"[{global_count:04d}] 正在下载 (尝试 {attempt + 1}/{self.retry_attempts}): {data.url}")
                
                # 使用原始的YouTube下载逻辑，但添加基本的错误处理
                yt = YouTube(data.url, use_oauth=True, on_progress_callback=on_progress)
                stream: Stream = yt.streams.filter().order_by("resolution").last()
                
                if stream is None:
                    raise Exception("未找到可用的视频流")
                
                # 下载到临时位置
                stream.download(output_path="/tmp", filename=tmpname)
                
                # 验证文件是否下载成功
                if tmp_path.exists() and tmp_path.stat().st_size > 0:
                    # 原子性移动文件
                    shutil.move(str(tmp_path), str(filepath))
                    
                    with self.download_lock:
                        self.success_count += 1
                    
                    self.logger.info(f"[{global_count:04d}] 下载完成: {video_id}")
                    return True
                else:
                    raise Exception("下载的临时文件不存在或为空")
                    
            except Exception as e:
                last_exception = e
                self.logger.warning(f"[{global_count:04d}] 尝试 {attempt + 1} 失败: {e}")
            
            finally:
                # 清理临时文件
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception as cleanup_error:
                        self.logger.debug(f"清理临时文件失败: {cleanup_error}")
            
            # 重试前的等待
            if attempt < self.retry_attempts - 1:
                sleep_time = (attempt + 1) * 2  # 指数退避
                self.logger.info(f"[{global_count:04d}] 等待 {sleep_time} 秒后重试...")
                sleep(sleep_time)

        # 所有重试均失败
        with open(self.failed_videos_file, "a", encoding='utf-8') as f:
            f.write(f"{filepath.name}\t{str(last_exception)}\n")
        
        with self.download_lock:
            self.failed_count += 1
        
        self.logger.error(f"[{global_count:04d}] 所有重试均失败: {data.url}, 最后错误: {last_exception}")
        return False

    def run(self, use_threading: bool = True, simple_mode: bool = False):
        """执行下载，支持多线程和单线程模式"""
        total_videos = len(self.list_data)
        print(f"[INFO] 开始下载 {total_videos} 个视频")
        print(f"[INFO] 使用 {'多线程' if use_threading else '单线程'} 模式 (工作线程: {self.max_workers if use_threading else 1})")
        
        # 选择下载函数
        download_func = self._download_single_video_simple if simple_mode else self._download_single_video
        
        if use_threading and self.max_workers > 1:
            # 多线程下载
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_data = {
                    executor.submit(download_func, data, i): data 
                    for i, data in enumerate(self.list_data)
                }
                
                # 使用tqdm显示进度
                with tqdm(total=total_videos, desc="下载进度") as pbar:
                    for future in concurrent.futures.as_completed(future_to_data):
                        try:
                            result = future.result()
                            pbar.set_postfix({
                                '成功': self.success_count,
                                '跳过': self.skipped_count, 
                                '失败': self.failed_count
                            })
                        except Exception as e:
                            self.logger.error(f"任务执行异常: {e}")
                        finally:
                            pbar.update(1)
        else:
            # 单线程下载
            for global_count, data in enumerate(tqdm(self.list_data, desc="下载进度")):
                download_func(data, global_count)
                sleep(1)  # 保持原始的延迟

        # 打印最终统计
        self._print_final_stats()

    def _print_final_stats(self):
        """打印最终统计信息"""
        total = len(self.list_data)
        print(f"\n[INFO] 下载完成!")
        print(f"[INFO] 总计: {total} 个视频")
        print(f"[INFO] 成功: {self.success_count} 个")
        print(f"[INFO] 跳过: {self.skipped_count} 个 (已存在)")
        print(f"[INFO] 失败: {self.failed_count} 个")
        print(f"[INFO] 超时: {self.timeout_count} 次")
        
        if self.failed_count > 0:
            print(f"[INFO] 失败的视频列表保存在: {self.failed_videos_file}")
            
        # 计算成功率
        attempted = total - self.skipped_count
        if attempted > 0:
            success_rate = (self.success_count / attempted) * 100
            print(f"[INFO] 成功率: {success_rate:.1f}% ({self.success_count}/{attempted})")

    def run_with_recovery(self, use_threading: bool = True, recovery_mode: bool = False):
        """支持故障恢复的运行模式"""
        if recovery_mode:
            print("[INFO] 启用故障恢复模式，将重试之前失败的视频...")
            self._retry_failed_videos()
        
        self.run(use_threading)
        
    def _retry_failed_videos(self):
        """重试之前失败的视频"""
        if not self.failed_videos_file.exists():
            print("[INFO] 没有找到失败的视频列表")
            return
            
        try:
            with open(self.failed_videos_file, 'r', encoding='utf-8') as f:
                failed_lines = f.readlines()
            
            failed_video_ids = []
            for line in failed_lines:
                video_filename = line.strip().split('\t')[0]  # 去除错误信息部分
                if video_filename.endswith('.mp4'):
                    video_id = video_filename[:-4]  # 移除.mp4扩展名
                    failed_video_ids.append(video_id)
            
            if not failed_video_ids:
                print("[INFO] 失败列表为空")
                return
                
            print(f"[INFO] 找到 {len(failed_video_ids)} 个失败的视频，准备重试...")
            
            # 过滤出需要重试的视频
            retry_data = []
            for data in self.list_data:
                video_id = data.url.split('=')[-1]
                if video_id in failed_video_ids:
                    retry_data.append(data)
            
            print(f"[INFO] 准备重试 {len(retry_data)} 个视频")
            
            # 备份原始失败文件并清空
            backup_file = self.failed_videos_file.with_suffix('.bak')
            shutil.copy2(self.failed_videos_file, backup_file)
            self.failed_videos_file.unlink()
            
            # 临时替换数据列表
            original_data = self.list_data
            self.list_data = retry_data
            
            try:
                self.run(use_threading=False)  # 重试时使用单线程避免过载
            finally:
                # 恢复原始数据列表
                self.list_data = original_data
                
        except Exception as e:
            self.logger.error(f"重试失败视频时出错: {e}")

    def show(self):
        """保持原始的show方法"""
        print("########################################")
        global_count = 0
        for data in self.list_data:
            print(f" URL : {data.url}")
            for idx in range(len(data)):
                print(f" SEQ_{idx} : {data.list_seqnames[idx]}")
                print(f" LEN_{idx} : {len(data.list_list_timestamps[idx])}")
                global_count += 1
            print("----------------------------------------")

        print(f"TOTAL : {global_count} sequences")

    def resume_download(self):
        """支持断点续传"""
        print("[INFO] 检查断点续传...")
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                print(f"[INFO] 发现之前的下载进度，已完成 {progress.get('completed', 0)} 个")
            except:
                pass


# 保持原始的DataDownloader类以确保向后兼容
class DataDownloader(OptimizedDataDownloader):
    """向后兼容的类名"""
    def __init__(self, dataroot: str, split: str):
        super().__init__(dataroot, split, max_workers=1, retry_attempts=1)
    
    def run(self):
        """保持原始的单线程行为"""
        super().run(use_threading=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="优化版本的RealEstate10K数据集下载器")
    parser.add_argument("--dataroot", type=str, default="datasets/RealEstate10K", 
                       help="数据集根目录")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"],
                       help="数据集分割 (train/test)")
    parser.add_argument("--max-workers", type=int, default=3,
                       help="最大并发下载数 (默认: 3)")
    parser.add_argument("--retry-attempts", type=int, default=3,
                       help="每个视频的重试次数 (默认: 3)")
    parser.add_argument("--download-timeout", type=int, default=300,
                       help="单个视频最大下载时间（秒，默认: 300）")
    parser.add_argument("--stall-timeout", type=int, default=60,
                       help="停滞检测超时时间（秒，默认: 60）")
    parser.add_argument("--simple-mode", action="store_true",
                       help="使用简化下载模式（接近原始逻辑，更稳定）")
    parser.add_argument("--single-thread", action="store_true",
                       help="使用单线程模式 (与原始脚本行为一致)")
    parser.add_argument("--show-only", action="store_true",
                       help="仅显示数据信息，不下载")
    parser.add_argument("--recovery-mode", action="store_true",
                       help="故障恢复模式，重试之前失败的视频")
    
    args = parser.parse_args()

    # 根据参数选择下载器
    if args.single_thread:
        downloader = DataDownloader(args.dataroot, args.split)
    else:
        downloader = OptimizedDataDownloader(
            args.dataroot, 
            args.split,
            max_workers=args.max_workers,
            retry_attempts=args.retry_attempts,
            download_timeout=args.download_timeout,
            stall_timeout=args.stall_timeout
        )

    downloader.show()
    
    if not args.show_only:
        if hasattr(downloader, 'resume_download'):
            downloader.resume_download()
        
        if args.single_thread:
            downloader.run()
        elif hasattr(downloader, 'run_with_recovery'):
            if args.simple_mode:
                # 使用简化模式
                downloader.run(use_threading=True, simple_mode=True)
            else:
                downloader.run_with_recovery(use_threading=True, recovery_mode=args.recovery_mode)
        else:
            downloader.run(use_threading=True)