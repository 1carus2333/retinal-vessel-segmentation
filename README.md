# 第四章：眼底血管分割

本仓库提供第四章实验的 U-Net 代码和已提交的预测结果。课程平台上此版本 Public Score 为 **0.79775**，是目前已知线上成绩最高的版本。之后测试的残差模型线上得分为 0.78662，因此不作为推荐提交。

## 文件

- `vessel_unet.py`：从头训练并预测的代码。
- `results/submission_vessels.csv`：线上得分 0.79775 的提交文件。
- `requirements.txt`：Python 依赖。

压缩包不包含课程数据集、训练权重或个人环境文件。

## 数据准备

从课程提供的压缩包获取数据，解压后按以下结构放置：

```text
data/segmentation/
├── train/
│   ├── image/1.jpg ... 20.jpg
│   └── label/1.jpg ... 20.jpg
└── test/
    └── image/1.jpg ... 20.jpg
```

## 运行

安装依赖后，在仓库根目录执行：

```bash
python vessel_unet.py --stage validate --epochs 25
python vessel_unet.py --stage final --epochs 25
```

代码需要可用的 CUDA GPU。默认训练过程先用训练图像 1–16 号训练、17–20 号验证，然后使用选出的轮次在全部 20 张训练图上训练，并预测测试图。预测 CSV 生成在 `vessel_unet_results/submission_vessels.csv`。

## 方法与验证

模型为小型 U-Net，从随机初始化开始训练。输入由 RGB 和绿色通道局部对比度组成；使用 128×128 随机裁块、旋转翻转及亮度增强，训练损失为 BCE 加 0.5 倍 Dice loss。推理时平均四种翻转方向的预测。

同一 17–20 号留出组上，初始 U-Net 的本地平均 Dice 为 0.79662；平台 Public Score 为 0.79775。验证集参与了轮次和阈值选择，因此本地分数不是独立测试成绩。线上成绩来自课程平台截图。提交编码血管区域，采用从 1 开始的行优先 RLE。
