import glob
import os
import re

import numpy as np
import pandas as pd

regs = {int: r"(0|[1-9]\d*)", float: r"(\d+(\.\d+)?)", bool: r"(True|False)", str: r"([a-z]+)"}
configs = {
    "iters": ("", "k", int, 0),
    "ImageTextcfg": ("ImageTextcfg", "", float, 7.5),
    "CameraCondition": ("CameraCondition", "", bool, False),
    "CameraCfg": ("CameraCfg", "", float, 1.0),
    "eta": ("eta", "", float, 1.0),
    "guidanceRescale": ("guidanceRescale", "", float, 0.7),
    "cfgScheduler": ("cfgScheduler=", "", str, "constant"),
    "frameStride": ("frameStride", "", int, 8),
}


def order(file):
    def capture(before, after, kind, default):
        cap = re.search(f"{before}{regs[kind]}{after}", setting)
        if cap is None:
            return default
        item = cap.group(1)
        return eval(item) if kind in (int, float, bool) else item

    _, method, setting, _ = file.split("/")
    return method, [capture(*v) for v in configs.values()]


def safe_extract(pattern, text, default="0"):
    """安全地提取正则表达式匹配，如果没找到则返回默认值"""
    match = re.search(pattern, text)
    return match.group(1) if match else default


metrics = ["RotErr", "TransErr", "CamMC"]
with open("summary.csv", "w") as f:
    f.write("Method,ImageTextcfg,CameraCfg,Time," + ",".join(metrics) + "\n")

summary = []
for file in sorted(glob.glob("results/*/*/merge.csv"), key=order):
    print(f"Processing: {file}")  # 添加调试信息
    
    _, method, setting, _ = file.split("/")
    print(f"Setting: {setting}")  # 查看setting的内容
    
    method = re.sub(r"_\d+(_\dgpu)?$", "", method.removeprefix("test_256_"))
    
    # 使用安全提取函数，提供默认值
    ticfg = safe_extract(rf"ImageTextcfg({regs[float]})", setting, "7.5")
    ccfg = safe_extract(rf"CameraCfg({regs[float]})", setting, "1.0")

    try:
        data = pd.read_csv(file, skiprows=[0]).iloc[:, 1:].mean(axis=0).values.tolist()
        summary.append([method, ticfg, ccfg] + data)
        print(list(map(lambda x: round(x, 4), data)))
    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

pd.DataFrame(summary).to_csv("summary.csv", mode="a", header=False, index=False)