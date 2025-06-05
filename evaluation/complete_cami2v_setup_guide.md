# CamI2V评估环境完整配置指南

## 概述

本指南基于实际部署经验，整合了所有已知问题的解决方案，为CamI2V评估环境提供一套完整、可靠的配置流程。

## 1. 环境准备与依赖检查

### 1.1 系统要求
- Ubuntu 20.04/22.04 LTS
- Python 3.10
- CUDA 12.1+ (可选，用于GPU加速)
- 至少16GB内存，32GB推荐
- 足够的存储空间（至少50GB用于模型和数据）

### 1.2 网络环境配置（重要）
如果在国内服务器或网络受限环境中，首先配置GitHub镜像源：

```bash
# 配置GitHub镜像源（解决网络连接问题）
git config --global url."https://gh-proxy.com/https://github.com/".insteadof "https://github.com/"

# 验证配置
git config --global --list | grep insteadof

# 增加Git超时设置
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 300
git config --global http.postBuffer 1048576000
```

## 2. 基础环境配置

### 2.1 创建Conda环境
```bash
# 创建Python环境
conda create -n cami2v python=3.10
conda activate cami2v

# 安装PyTorch和相关依赖
conda install -y pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y xformers -c xformers

# 验证CUDA可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"
```

### 2.2 安装项目依赖
```bash
# 在CamI2V项目根目录执行
cd /path/to/CamI2V
pip install -r requirements.txt

# 更新子模块
git submodule update --init
```

## 3. 系统依赖安装

### 3.1 基础编译工具
```bash
# 基础编译工具
sudo apt update
sudo apt install -y git cmake ninja-build build-essential ccache

# 数学和图像处理库
sudo apt install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev

# Boost库和其他依赖
sudo apt install -y libboost-program-options-dev libboost-graph-dev libboost-system-dev libboost-filesystem-dev
sudo apt install -y libflann-dev libfreeimage-dev libmetis-dev libsqlite3-dev
sudo apt install -y libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev

# 测试框架
sudo apt install -y libgtest-dev libgmock-dev
```

### 3.2 编译器配置
```bash
# 安装并配置GCC 10
sudo apt install -y gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10

# 添加到bashrc以持久化
echo 'export CC=/usr/bin/gcc-10' >> ~/.bashrc
echo 'export CXX=/usr/bin/g++-10' >> ~/.bashrc
echo 'export CUDAHOSTCXX=/usr/bin/g++-10' >> ~/.bashrc
```

## 4. CMake版本升级（关键步骤）

GLOMAP需要CMake >= 3.28，而Ubuntu默认版本通常较旧。

### 4.1 检查当前版本
```bash
cmake --version
```

### 4.2 升级CMake（推荐使用Conda）
```bash
# 方法1：使用Conda安装（推荐）
conda install -c conda-forge cmake=3.29

# 方法2：使用APT从Kitware仓库安装
sudo apt remove --purge cmake
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
sudo apt update
sudo apt install -y cmake

# 验证安装
cmake --version  # 应显示 >= 3.28
```

### 4.3 处理可能的SSL问题
如果遇到libssl1.1缺失问题：
```bash
# 添加源并安装libssl1.1
echo "deb http://security.ubuntu.com/ubuntu focal-security main" | sudo tee /etc/apt/sources.list.d/focal-security.list
sudo apt update
sudo apt install -y libssl1.1
```

## 5. 核心组件编译安装

### 5.1 GTest手动编译（确保兼容性）
```bash
# 先用APT安装
sudo apt install -y libgtest-dev libgmock-dev

# 然后手动编译以确保兼容性
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make -j$(nproc)
sudo cp lib/*.a /usr/lib/ 2>/dev/null || sudo cp *.a /usr/lib/
```

### 5.2 Ceres-solver 2.3.0
```bash
cd /path/to/CamI2V/evaluation/ceres-solver

# 清理之前的构建（如果有）
rm -rf build
git clean -fdx
git submodule update --init --recursive

# 配置编译选项
cmake -S . -B build -G Ninja \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DCERES_THREADING_MODEL=OPENMP \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=FALSE

# 编译安装
sudo cmake --build build --target install
```

### 5.3 COLMAP 3.11.0
```bash
cd /path/to/CamI2V/evaluation/colmap

# 清理构建目录
rm -rf build

# 根据GPU可用性选择配置
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    # GPU版本
    cmake -S . -B build -G Ninja \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ENABLED=ON \
        -DGUI_ENABLED=OFF \
        -DTESTS_ENABLED=OFF \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=FALSE
else
    # CPU版本
    cmake -S . -B build -G Ninja \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCUDA_ENABLED=OFF \
        -DGUI_ENABLED=OFF \
        -DTESTS_ENABLED=OFF \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=FALSE
fi

# 编译安装
sudo cmake --build build --target install
```

### 5.4 GLOMAP 1.0.0（关键修复）
```bash
cd /path/to/CamI2V/evaluation/glomap

# 清理构建目录
rm -rf build

# 配置和编译
cmake -S . -B build -G Ninja \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=FALSE

cmake --build build

# 创建符号链接
sudo ln -sf $(pwd)/build/glomap/glomap /usr/local/bin/glomap
```

**注意**：如果GLOMAP编译时遇到网络问题，确保已配置GitHub镜像源（步骤1.2）。

### 5.5 FVD评估工具
```bash
cd /path/to/CamI2V/evaluation
pip install FVD/fvdcal-1.0-py3-none-any.whl
```

## 6. 验证安装

### 6.1 检查所有工具
```bash
# 检查版本和可用性
cmake --version          # >= 3.28
colmap --help           # 应显示帮助信息
glomap --help           # 应显示帮助信息
python -c "from fvdcal import FVDCalculation; print('FVD OK')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 6.2 路径检查
```bash
which cmake
which colmap
which glomap
ls -la /usr/local/bin/glomap
```

## 7. 模型和数据准备

### 7.1 创建目录结构
```bash
cd /path/to/CamI2V
mkdir -p ckpts
mkdir -p pretrained_models
mkdir -p datasets/RealEstate10K
```

### 7.2 下载CamI2V检查点
```bash
cd ckpts

# 下载预训练模型（选择需要的版本）
wget https://huggingface.co/MuteApo/CamI2V/resolve/main/512_cami2v_50k.pt
wget https://huggingface.co/MuteApo/CamI2V/resolve/main/512_cami2v_100k.pt
wget https://huggingface.co/MuteApo/CamI2V/resolve/main/256_cami2v.pt
```

### 7.3 配置模型路径
```bash
# 编辑configs/models.json，确保路径正确
# 示例配置：
{
    "cami2v_256": "ckpts/256_cami2v.pt",
    "cami2v_512_50k": "ckpts/512_cami2v_50k.pt",
    "cami2v_512_100k": "ckpts/512_cami2v_100k.pt"
}
```

### 7.4 下载DynamiCrafter基础模型
```bash
cd pretrained_models

# 创建目录结构
mkdir -p DynamiCrafter DynamiCrafter_512

# 下载基础模型（根据需要）
# 注意：这些是大文件，确保有足够的存储空间和网络带宽
```

### 7.5 下载测试元数据
```bash
cd datasets/RealEstate10K
wget https://huggingface.co/MuteApo/CamI2V/resolve/main/CamI2V_test_metadata_1k.pth
```

## 8. 运行评估测试

### 8.1 生成测试视频
```bash
cd /path/to/CamI2V

# 设置配置参数
config_file=configs/inference/003_cami2v_256x256.yaml
save_root=../test_results
suffix_name=256_CamI2V

# 运行评估（根据GPU数量调整nproc_per_node）
torchrun --standalone --nproc_per_node 1 main/trainer.py --test \
    --base $config_file --logdir $save_root --name $suffix_name
```

### 8.2 相机控制指标评估
```bash
# 设置实验目录路径
EXP_DIR=${save_root}/${suffix_name}/images/test/$(basename $config_file .yaml)

# 运行相机指标评估
python glomap_evaluation.py --exp_dir $EXP_DIR
python utils/merge.py
python utils/summary.py
```

### 8.3 FVD视觉质量评估
```bash
python fvd_test.py --gt_folder $EXP_DIR/gt_video --sample_folder $EXP_DIR/samples
```

## 9. 常见问题与解决方案

### 9.1 网络连接问题
**症状**：GitHub克隆失败，连接超时
```bash
# 解决方案：配置镜像源
git config --global url."https://gh-proxy.com/https://github.com/".insteadof "https://github.com/"
```

### 9.2 CMake版本问题
**症状**：GLOMAP要求CMake >= 3.28
```bash
# 解决方案：使用Conda升级
conda install -c conda-forge cmake=3.29
```

### 9.3 GTest编译错误
**症状**：找不到GTest或版本不兼容
```bash
# 解决方案：手动编译GTest
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make -j$(nproc)
sudo cp lib/*.a /usr/lib/ 2>/dev/null || sudo cp *.a /usr/lib/
```

### 9.4 CUDA兼容性问题
**症状**：CUDA版本不匹配或不可用
```bash
# 检查CUDA状态
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# 如果CUDA不可用，使用CPU版本配置
```

### 9.5 内存不足问题
**症状**：编译或评估时内存不足
```bash
# 解决方案：
# 1. 减少并行编译进程数
make -j2  # 而不是 make -j$(nproc)

# 2. 增加swap空间
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 9.6 权限问题
**症状**：无法安装到系统目录
```bash
# 解决方案：确保使用sudo或配置用户权限
sudo cmake --build build --target install
```

## 10. 性能优化建议

### 10.1 编译优化
```bash
# 使用更多CPU核心加速编译
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)

# 启用ccache加速重复编译
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
```

### 10.2 运行时优化
```bash
# 设置OpenMP线程数
export OMP_NUM_THREADS=$(nproc)

# 优化CUDA内存使用
export CUDA_VISIBLE_DEVICES=0  # 指定使用的GPU
```

## 11. 环境验证清单

在开始评估之前，确保以下检查全部通过：

- [ ] CMake版本 >= 3.28
- [ ] Python 3.10环境激活
- [ ] PyTorch + CUDA可用（如果有GPU）
- [ ] COLMAP命令可用
- [ ] GLOMAP命令可用
- [ ] FVD库可导入
- [ ] 模型文件下载完成
- [ ] 测试数据准备完成
- [ ] 配置文件路径正确

## 12. 故障排除流程

如果遇到问题，请按以下顺序排查：

1. **检查网络连接**：确保GitHub镜像源配置正确
2. **验证依赖版本**：CMake、GCC、Python等版本是否满足要求
3. **清理重建**：删除build目录，重新编译
4. **检查日志**：查看详细的编译或运行日志
5. **逐步验证**：每安装一个组件就测试一次
6. **环境隔离**：确保在正确的Conda环境中操作

## 13. 附录：完整的安装脚本

```bash
#!/bin/bash
# CamI2V评估环境一键安装脚本

set -e  # 遇到错误立即退出

echo "开始安装CamI2V评估环境..."

# 1. 配置GitHub镜像源
git config --global url."https://gh-proxy.com/https://github.com/".insteadof "https://github.com/"

# 2. 安装系统依赖
sudo apt update
sudo apt install -y git cmake ninja-build build-essential ccache gcc-10 g++-10
sudo apt install -y libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev
sudo apt install -y libboost-program-options-dev libboost-graph-dev libboost-system-dev libboost-filesystem-dev
sudo apt install -y libflann-dev libfreeimage-dev libmetis-dev libsqlite3-dev libgtest-dev libgmock-dev
sudo apt install -y libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev

# 3. 设置环境变量
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10

# 4. 升级CMake
conda install -c conda-forge cmake=3.29 -y

# 5. 编译核心组件
cd evaluation

# Ceres-solver
cd ceres-solver
rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
sudo cmake --build build --target install

# COLMAP
cd ../colmap
rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Release -DGUI_ENABLED=OFF -DTESTS_ENABLED=OFF
sudo cmake --build build --target install

# GLOMAP
cd ../glomap
rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_CXX_STANDARD=17 -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo ln -sf $(pwd)/build/glomap/glomap /usr/local/bin/glomap

# 6. 安装Python依赖
cd ..
pip install FVD/fvdcal-1.0-py3-none-any.whl

echo "CamI2V评估环境安装完成！"
echo "请运行以下命令验证安装："
echo "cmake --version"
echo "colmap --help"
echo "glomap --help"
echo "python -c \"from fvdcal import FVDCalculation; print('FVD OK')\""
```

---

**注意事项**：
- 整个安装过程可能需要1-2小时，取决于网络速度和硬件性能
- 建议在开始前确保有足够的磁盘空间（至少20GB用于编译）
- 如果在生产环境中使用，建议先在测试环境中验证整个流程
- 定期备份重要的配置文件和模型文件

通过遵循本指南，您应该能够成功搭建CamI2V评估环境并运行相关实验。