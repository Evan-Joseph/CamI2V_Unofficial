# CamI2V评估环境完整配置指南 (版本 3.0 - 优化版)

## 概述

本指南基于官方README和实际部署经验，为CamI2V评估环境提供一套完整、可靠的配置流程。采用"Conda优先"策略，针对AutoDL等云平台进行优化，确保环境稳定、空间充裕。

## 1. 环境准备与依赖检查

### 1.1 系统要求
- **操作系统**: Ubuntu 20.04/22.04 LTS
- **Python版本**: 3.10 (必须，官方指定)
- **CUDA版本**: 12.1+ (GPU加速推荐)
- **内存要求**: 最低16GB，推荐32GB
- **存储空间**: 至少50GB (模型、数据集、编译文件)

### 1.2 网络环境配置 (国内用户必须)

```bash
# 配置GitHub镜像源解决网络连接问题
git config --global url."https://gh-proxy.com/https://github.com/".insteadof "https://github.com/"

# 设置Git超时和缓冲区
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 300
git config --global http.postBuffer 1048576000

# 验证配置
git config --global --list | grep insteadof
```

### 1.3 配置软件源镜像 (国内用户推荐)

```bash
# 配置Conda镜像源 (清华源)
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes

# 配置pip镜像源 (清华源)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 2. 空间优化配置 (AutoDL等云平台)

### 2.1 配置Conda使用数据盘

**背景**: AutoDL等平台系统盘空间有限(~30GB)，数据盘空间充裕(50GB+)。

```bash
# 创建Conda专用目录 (以AutoDL为例)
mkdir -p /root/autodl-tmp/conda_envs
mkdir -p /root/autodl-tmp/conda_pkgs

# 配置Conda使用数据盘
conda config --add envs_dirs /root/autodl-tmp/conda_envs
conda config --add pkgs_dirs /root/autodl-tmp/conda_pkgs

# 验证配置
conda info
```

### 2.2 项目部署到数据盘

```bash
# 切换到数据盘
cd /root/autodl-tmp

# 克隆项目 (使用镜像源)
git clone https://github.com/ZGCTroy/CamI2V.git
cd CamI2V

# 更新子模块
git submodule update --init
```

## 3. Conda环境配置 (核心步骤)

### 3.1 创建基础环境

```bash
# 创建Python 3.10环境 (官方要求)
conda create -n cami2v python=3.10 -y
conda activate cami2v
```

### **3.2 核心依赖安装 (V4.0 “屠龙者”版)**

**警告**: 本节是整个安装过程中最关键、也最容易出错的环节。请严格按照以下**顺序**执行，不要随意更改或颠倒！本方案整合了 `conda` 和 `pip` 的优点，并解决了官方 `requirements.txt` 中存在的依赖冲突。

#### **第1步：Conda 安装核心基石 (Foundation)**

我们首先使用 Conda 安装最底层、最核心的 PyTorch 和 xFormers。这能最大程度保证与 CUDA 驱动的兼容性和稳定性。

```bash
# 确保你已在 cami2v 环境中
conda activate cami2v

# 使用 conda 安装 pytorch 和 xformers
conda install -y pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y xformers -c xformers
```
**说明**: 完成此步后，你的环境已经拥有了稳定可靠的 GPU 计算基础。

#### **第2步：Pip 安装大部分常规依赖**

接下来，我们使用 pip 安装 `requirements.txt` 中的绝大部分包。但为了避免冲突，我们**必须先处理掉有问题的 `autoawq`**。

1.  **编辑 `requirements.txt` 文件**：
    打开项目根目录下的 `requirements.txt` 文件，找到最后一行 `autoawq`，在它前面加一个 `#` 将其注释掉。
    ```diff
    ...
    accelerate
    qwen_vl_utils[decord]
    optimum
    # autoawq
    ```

2.  **安装依赖**:
    现在，可以安全地安装其余的依赖了。
    ```bash
    pip install -r requirements.txt
    ```

#### **第3步：“外科手术式”安装 `autoawq`**

由于 `autoawq` 的依赖问题和国内镜像源的同步延迟问题，我们必须手动、精准地从官方 PyPI 源安装它的最新兼容版本。

```bash
# 使用 --index-url 直接从官方 PyPI 安装，绕过国内镜像
# 使用 --no-cache-dir 避免任何本地缓存问题
pip install --no-cache-dir "autoawq>=0.2.8" --index-url https://pypi.org/simple
```
**说明**: 这条命令会强制 `pip` 从全球官方仓库下载与 `torch 2.4.1` 兼容的最新版 `autoawq`，完美解决冲突。

---

### **3.3 验证核心环境 (增强版)**

完成上述所有步骤后，我们进行一次增强版的验证。

```bash
# 验证 PyTorch 和 CUDA
python -c "import torch; print(f'✅ PyTorch 版本: {torch.__version__}'); print(f'✅ CUDA 可用: {torch.cuda.is_available()}')"

# 验证其他关键依赖
python -c "import transformers; print('✅ Transformers OK')"
python -c "import diffusers; print('✅ Diffusers OK')"
python -c "import lightning; print('✅ Lightning OK')"

# 验证“天坑”级依赖 autoawq (注意导入名是 awq !)
python -c "import awq; print('✅ AutoAWQ (导入名为 awq) OK')"
```

**核心提示**: `autoawq` 这个包，在 `pip install` 时用的是名字 `autoawq`，但在代码中 `import` 时，必须使用它的“真名” `awq`！

## 4. 系统依赖安装

### 4.1 基础编译环境

```bash
# 更新软件包列表
sudo apt update

# 安装编译工具
sudo apt install -y \
    git cmake ninja-build build-essential ccache \
    gcc-10 g++-10

# 安装数学和图像处理库
sudo apt install -y \
    libgoogle-glog-dev libgflags-dev libatlas-base-dev \
    libeigen3-dev libsuitesparse-dev libboost-program-options-dev \
    libboost-graph-dev libboost-system-dev libboost-filesystem-dev \
    libflann-dev libfreeimage-dev libmetis-dev libsqlite3-dev

# 安装GUI和测试框架
sudo apt install -y \
    libglew-dev qtbase5-dev libqt5opengl5-dev \
    libcgal-dev libceres-dev libgtest-dev libgmock-dev
```

### 4.2 编译器配置

```bash
# 设置GCC 10为默认编译器 (推荐)
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10

# 永久配置
echo 'export CC=/usr/bin/gcc-10' >> ~/.bashrc
echo 'export CXX=/usr/bin/g++-10' >> ~/.bashrc
echo 'export CUDAHOSTCXX=/usr/bin/g++-10' >> ~/.bashrc
source ~/.bashrc
```

## 5. CMake版本升级 (关键步骤)

### 5.1 检查版本需求

```bash
# 检查当前版本
cmake --version

# GLOMAP需要CMake >= 3.28
```

### 5.2 升级CMake (V4.0 高速版)

#### **方法1: 使用 Pip 高速安装 (强烈推荐)** ✅

鉴于 `conda-forge` 频道在国内可能存在解析超时的问题，我们采用 `pip` 进行安装，它会利用已配置的镜像源，速度更快、更直接。

```bash
# 在 cami2v 环境中执行
pip install "cmake==3.28"
```

#### **方法2: 使用 Conda (备选方案，可能较慢)**

如果你希望所有工具都由 Conda 管理，可以尝试此方法。如果长时间卡住或超时，请果断 `Ctrl+C` 并使用方法1。

```bash
# 此命令可能因网络原因解析缓慢
conda install -c conda-forge cmake>=3.28 -y
```

#### **验证安装**

```bash
# 验证版本号
cmake --version

# 确认 cmake 命令的路径
which cmake
```

## 6. 核心组件编译

### 6.1 处理GTest兼容性

```bash
# 手动编译系统GTest确保兼容性
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make -j$(nproc)
sudo cp lib/*.a /usr/lib/ 2>/dev/null || sudo cp *.a /usr/lib/
```

### 6.2 编译Ceres-solver 2.3.0

```bash
cd evaluation/ceres-solver

# 清理之前的构建
rm -rf build
git clean -fdx
git submodule update --init --recursive

# 配置编译 (按照官方指南)
cmake -S . -B build -G Ninja \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCERES_THREADING_MODEL=OPENMP

# 编译安装
sudo cmake --build build --target install
```

### 6.3 编译COLMAP 3.11.0

```bash
cd ../colmap

# 清理构建目录
rm -rf build

# 配置编译 (自动检测CUDA)
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "配置GPU版本COLMAP..."
    cmake -S . -B build -G Ninja \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ENABLED=ON \
        -DGUI_ENABLED=OFF \
        -DTESTS_ENABLED=OFF
else
    echo "配置CPU版本COLMAP..."
    cmake -S . -B build -G Ninja \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ENABLED=OFF \
        -DGUI_ENABLED=OFF \
        -DTESTS_ENABLED=OFF
fi

# 编译安装
sudo cmake --build build --target install
```

### 6.4 编译GLOMAP 1.0.0

```bash
cd ../glomap

# 清理构建目录
rm -rf build

# 配置编译 (按照官方指南)
cmake -S . -B build -G Ninja \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build build

# 创建系统链接
sudo ln -sf $(pwd)/build/glomap/glomap /usr/local/bin/glomap
```

### 6.5 安装FVD评估工具

```bash
cd ..  # 回到evaluation目录
pip install FVD/fvdcal-1.0-py3-none-any.whl
```

## 7. 模型与数据下载 (V4.0 “一键终极版”)

**重大升级**：我们不再使用繁琐的 `wget` 命令，而是采用项目根目录下的终极下载脚本 `cami2v_interactive_downloader.py`。该脚本由社区贡献，支持菜单选择、断点续传、失败重试，并能一键下载所有必需资源，极大提升下载效率和成功率。

### 7.1 创建目录结构 (可选)

脚本会自动创建所需目录，但你也可以手动预先创建以确认结构。

```bash
cd /root/autodl-tmp/CamI2V  # 项目根目录

# 创建必要目录
mkdir -p ckpts
mkdir -p pretrained_models/{DynamiCrafter,DynamiCrafter_512,Qwen2-VL-7B-Instruct-AWQ}
mkdir -p datasets/RealEstate10K
```

### 7.2 运行终极下载脚本

1.  **确保脚本可执行** (如果需要)
    ```bash
    chmod +x cami2v_interactive_downloader.py
    ```

2.  **启动交互式下载器**
    ```bash
    python cami2v_interactive_downloader.py
    ```

3.  **根据菜单提示操作**
    脚本会显示一个清晰的菜单，你可以按需选择：
    *   `1. CamI2V模型检查点`: 下载 CamI2V 官方发布的 `.pt` 模型。
    *   `2. 基础模型(256分辨率)`: 下载 DynamiCrafter 256px 版本。
    *   `3. 基础模型(512分辨率)`: 下载 DynamiCrafter 512px 版本。
    *   `4. 视频数据集`: (高级) 下载 RealEstate10K 数据集。
    *   `5. 一键下载所有模型`: **强烈推荐初次配置时使用此选项！**
    *   `6. 从日志重试失败的下载`: 如果有下载失败，脚本会自动记录，此选项可让你只重试失败的部分。

    **推荐步骤**:
    - 首次配置，请直接选择 **选项5**，等待所有模型下载完成。
    - 如果需要训练或使用完整数据集，再单独运行并选择 **选项4**。

### 7.3 配置模型路径 (必须)

下载完成后，项目的配置文件并不知道模型具体在哪。你需要编辑 `configs/models.json` 文件，确保路径正确。**以下是标准配置，通常无需修改**。

```json
{
    "cami2v_256": "ckpts/256_cami2v.pt",
    "cami2v_512_50k": "ckpts/512_cami2v_50k.pt",
    "cami2v_512_100k": "ckpts/512_cami2v_100k.pt",
    "cameractrl_256": "ckpts/256_cameractrl.pt",
    "motionctrl_256": "ckpts/256_motionctrl.pt"
}
```

### 7.4 下载基础模型

```bash
# 下载DynamiCrafter基础模型 (根据需要选择)
# 注意: 这些是大文件，确保网络稳定和存储空间充足

# 256x256版本
cd pretrained_models/DynamiCrafter
# 下载model.ckpt文件

# 512x320版本
cd ../DynamiCrafter_512
# 下载model.ckpt文件
```

### 7.5 下载测试数据

```bash
cd datasets/RealEstate10K
wget https://huggingface.co/MuteApo/CamI2V/resolve/main/CamI2V_test_metadata_1k.pth
```

## 8. 环境验证

### 8.1 完整性检查

```bash
# 检查工具版本
echo "=== 工具版本检查 ==="
cmake --version | head -n1
python --version
conda --version

# 检查编译工具
echo "=== 编译工具检查 ==="
colmap --help > /dev/null && echo "COLMAP: OK" || echo "COLMAP: FAILED"
glomap --help > /dev/null && echo "GLOMAP: OK" || echo "GLOMAP: FAILED"

# 检查Python依赖
echo "=== Python依赖检查 ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from fvdcal import FVDCalculation; print('FVD: OK')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 8.2 模型文件检查

```bash
echo "=== 模型文件检查 ==="
ls -lh ckpts/
ls -lh pretrained_models/
ls -lh datasets/RealEstate10K/
```

## 9. 快速测试

### 9.1 运行Gradio演示

```bash
# 启动Gradio演示 (可选使用Qwen2-VL)
python cami2v_gradio_app.py --use_qwenvl_captioner

# 如果网络连接问题
python cami2v_gradio_app.py --use_host_ip
```

### 9.2 生成测试视频

```bash
# 设置参数
config_file=configs/inference/003_cami2v_256x256.yaml
save_root=../test_results
suffix_name=256_CamI2V

# 运行测试 (根据GPU数量调整nproc_per_node)
torchrun --standalone --nproc_per_node 1 main/trainer.py --test \
    --base $config_file --logdir $save_root --name $suffix_name
```

### 9.3 运行评估

```bash
# 设置实验目录
EXP_DIR=${save_root}/${suffix_name}/images/test/$(basename $config_file .yaml)

# 相机控制指标评估
python evaluation/glomap_evaluation.py --exp_dir $EXP_DIR
python evaluation/utils/merge.py
python evaluation/utils/summary.py

# FVD视觉质量评估
python evaluation/fvd_test.py --gt_folder $EXP_DIR/gt_video --sample_folder $EXP_DIR/samples
```

## 10. 常见问题解决方案

### 10.1 网络连接问题

**症状**: GitHub克隆失败，模型下载中断

**解决方案**:
```bash
# 1. 使用GitHub镜像源 (见1.2节)
# 2. 使用多线程下载工具
sudo apt install axel
axel -n 8 [下载链接]
```

### 10.2 空间不足问题

**症状**: 编译时提示"No space left on device"

**解决方案**:
```bash
# 1. 清理Conda缓存
conda clean --all -y

# 2. 清理pip缓存
pip cache purge

# 3. 清理编译缓存
rm -rf evaluation/*/build

# 4. 添加交换空间
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 10.3 编译错误问题

**症状**: CMake配置失败，找不到依赖

**解决方案**:
```bash
# 1. 确保编译器版本正确
gcc-10 --version
g++-10 --version

# 2. 检查环境变量
echo $CC $CXX $CUDAHOSTCXX

# 3. 重新安装系统依赖
sudo apt update
sudo apt install --reinstall [相关包名]
```

### 10.4 CUDA兼容性问题

**症状**: torch.cuda.is_available() 返回False

**解决方案**:

```bash
# 1. 检查NVIDIA驱动
nvidia-smi

# 2. 检查CUDA版本兼容性
nvcc --version

# 3. 重新安装PyTorch
conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### **10.5  pip install 成功但 import 失败（新增）**

**症状**:

1.  运行 `pip install package-name` 提示 `Successfully installed`。
2.  运行 `pip list` 能看到 `package-name`。
3.  但在 Python 中 `import package_name` 却报 `ModuleNotFoundError`。

**可能原因与解决方案**:

*   **原因1：包的“艺名”与“真名”不一致 (最常见)**
    *   **诊断**: 就像我们遇到的 `autoawq`，它的安装名叫 `autoawq`，但导入名是 `awq`。
    *   **解决**: 去这个包的 PyPI 官方页面或 GitHub 仓库，查看它的使用示例，确认正确的导入名称。

*   **原因2：安装不完整，只留下了元数据**
    *   **诊断**: 在 `site-packages` 目录里 `ls`，只能看到 `package-name.dist-info` 文件夹，却看不到 `package_name` 文件夹。
    *   **解决**: 这是罕见的安装错误。使用终极命令强制重装：`pip install --no-cache-dir --force-reinstall package-name`。

## 11. 性能优化建议

### 11.1 编译优化

```bash
# 启用ccache加速编译
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

# 使用所有CPU核心
export OMP_NUM_THREADS=$(nproc)
```

### 11.2 运行时优化

```bash
# GPU设备选择
export CUDA_VISIBLE_DEVICES=0,1

# 内存优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## 12. 一键安装脚本

```bash
#!/bin/bash
# CamI2V一键安装脚本 (v3.0)

set -e

echo "开始安装CamI2V评估环境..."

# 前置检查
if ! command -v conda &> /dev/null; then
    echo "错误: 请先安装Anaconda或Miniconda"
    exit 1
fi

# 配置镜像源
echo "配置镜像源..."
git config --global url."https://gh-proxy.com/https://github.com/".insteadof "https://github.com/"
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 创建环境
echo "创建Conda环境..."
conda create -n cami2v python=3.10 -y
source activate cami2v

# 安装依赖
echo "安装Python依赖..."
conda install -y pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y xformers -c xformers
pip install -r requirements.txt

# 安装系统依赖
echo "安装系统依赖..."
sudo apt update
sudo apt install -y git cmake ninja-build build-essential ccache gcc-10 g++-10
# ... (其他依赖)

# 编译工具
echo "编译评估工具..."
# ... (编译步骤)

echo "安装完成！"

# (新增) 下载模型文件
echo "下载模型文件..."
python cami2v_interactive_downloader.py --auto-models

echo "安装完成！模型已下载，请运行验证脚本检查环境。"
```

## 13. 环境验证清单

完成安装后，请确保以下检查全部通过：

- [ ] Conda环境 `cami2v` 已创建并激活
- [ ] PyTorch + CUDA 正常工作
- [ ] CMake版本 >= 3.28
- [ ] COLMAP 和 GLOMAP 命令可用
- [ ] FVD库可正常导入
- [ ] 模型文件下载完成
- [ ] 配置文件路径正确
- [ ] 测试数据准备就绪

---

**总结**: 本指南整合了官方文档和实践经验，提供了完整的安装流程。特别针对国内用户优化了镜像源配置，针对云平台优化了存储空间管理。按照此指南操作，应该能够成功搭建CamI2V评估环境。