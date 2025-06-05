import glob
from argparse import ArgumentParser

import torch
from fvdcal import FVDCalculation
from fvdcal.video_preprocess import load_video
from torch import Tensor
from tqdm import tqdm


class MyFVDCalculation(FVDCalculation):
    def calculate_fvd_by_video_list(self, real_videos: Tensor, generated_videos: Tensor, model_path="FVD/model"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._load_model(model_path, device)

        fvd = self._compute_fvd_between_video(model, real_videos, generated_videos, device)

        return fvd.detach().cpu().numpy()


def load_videos(paths, desc):
    videos = []
    for path in tqdm(paths, desc=desc):
        video = load_video(path, num_frames=None)
        videos.append(video)
        
    # 添加调试信息
    if videos:
        print(f"Video shape: {videos[0].shape}")
        print(f"Video dtype: {videos[0].dtype}")
        print(f"Video range: [{videos[0].min():.3f}, {videos[0].max():.3f}]")
    
    return torch.stack(videos)


def metric(gt_folder, sample_folder):
    gt_video_paths = sorted(glob.glob(f"{gt_folder}/*.mp4"))
    sample_video_paths = sorted(glob.glob(f"{sample_folder}/*.mp4"))
    
    print(f"Found {len(gt_video_paths)} GT videos")
    print(f"Found {len(sample_video_paths)} sample videos")
    
    # 确保视频数量匹配
    min_count = min(len(gt_video_paths), len(sample_video_paths))
    gt_video_paths = gt_video_paths[:min_count]
    sample_video_paths = sample_video_paths[:min_count]
    print(f"Using {min_count} video pairs for evaluation")

    gt_videos = load_videos(gt_video_paths, "loading real videos")
    sample_videos = load_videos(sample_video_paths, "loading generated videos")
    
    print(f"GT videos tensor shape: {gt_videos.shape}")
    print(f"Sample videos tensor shape: {sample_videos.shape}")

    score_videogpt = fvd_videogpt.calculate_fvd_by_video_list(gt_videos, sample_videos)
    print(f"FVD (VideoGPT): {score_videogpt}")

    score_stylegan = fvd_stylegan.calculate_fvd_by_video_list(gt_videos, sample_videos)
    print(f"FVD (StyleGAN): {score_stylegan}")

    return score_videogpt, score_stylegan


fvd_videogpt = MyFVDCalculation(method="videogpt")
fvd_stylegan = MyFVDCalculation(method="stylegan")

parser = ArgumentParser()
parser.add_argument("--gt_folder", type=str)
parser.add_argument("--sample_folder", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    metric(args.gt_folder, args.sample_folder)