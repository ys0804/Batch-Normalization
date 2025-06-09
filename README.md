# Batch-Normalization
## 项目背景
批归一化（BN）是深度神经网络训练中的核心技术，通过稳定层输入的分布（减少内部协变量偏移），加速模型收敛并提升泛化能力。本项目基于 CIFAR-10 数据集 和 简化版 VGG-A 架构，对比 “带 BN” 与 “不带 BN” 模型的训练表现、优化景观（Loss Landscape），深入探究 BN 对优化过程的影响机制。
## 实验目标
验证 BN 的训练有效性：对比有无 BN 的 VGG-A 模型在损失收敛速度、准确率提升幅度上的差异。
探索 BN 的优化机制：通过分析 损失景观、梯度范数 等指标，揭示 BN 如何让优化空间更平滑、梯度更稳定。
环境依赖

# 核心依赖（Python 3.7+）
torch >= 1.7.0      # 模型训练与张量运算  
torchvision >= 0.8.0 # 数据集加载与数据增强  
matplotlib >= 3.0.0  # 结果可视化  
numpy >= 1.18.0      # 数值计算  
tqdm >= 4.0.0        # 训练进度条  


安装命令：
pip install torch torchvision matplotlib numpy tqdm

## 实验设计与运行步骤
运行命令：
python BN.py


训练过程实时输出 每轮训练损失、测试损失、准确率、梯度范数。
