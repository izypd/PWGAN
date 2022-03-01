<p align="center">
  <a href="https://izypd.com/pwgan">
    <img src="https://cdn.jsdelivr.net/gh/izypd/Gauss@main/PWGAN/PWGAN结果.png">
  </a>
</p>

<h1 align="center">Icon-to-Icon Translation using Pixel-Level Weight Adversarial Networks</h1>

<p align="center">
  <a href="https://izypd.com/pwgan">
    预览
  </a>
  <a href="http://bit.kuas.edu.tw/~jni/2022/vol7/s1/11.JNI0275.pdf">
    论文
  </a>
</p>

<div align="center">

PWGAN 是一种基于生成对抗网络的图标转换方法，与现有方法对比，在定量和定性评估中，都达到了目前最好的水准。
本项目是 PWGAN 的 PyTorch 实现。
助力智能图标设计，只需提供转换目标图标集，与输入图标集共同参与无监督训练。利用训练得到的模型，可自动将输入图标集转换为目标图标集的风格，实现智能图标风格迁移。

[![Python](https://img.shields.io/badge/Python-3.6+-3776AB?style=for-the-badge)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4+-EE4C2C?style=for-the-badge)](https://pytorch.org)
[![PR Welcome](https://img.shields.io/badge/PR-welcome-60ca2b?style=for-the-badge)](https://github.com/izypd/blog-react/pulls)
[![License](https://img.shields.io/badge/License-GPL-60ca2b?style=for-the-badge)](https://www.gnu.org/licenses/gpl-3.0.html)

</div>

## Prerequisites
- Linux or macOS
- Python 3.6+
- PyTorch 1.4+ with CUDA/ROCm or CPU

## 快速上手

### 文档

模型训练、测试 [tips](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md)
常见[问题](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md)
自定义数据集的[template](data/template_dataset.py)
自定义模型的[template](models/template_model.py)
代码结构[overview](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/overview.md)
如何使用自己的数据集[datasets](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md)

### 安装

- Clone this repo:
```bash
git clone https://github.com/izypd/PWGAN.git
cd PWGAN
```

- 安装[PyTorch](http://pytorch.org)和其他依赖（如torchvision，[visdom](https://github.com/facebookresearch/visdom)，[dominate](https://github.com/Knio/dominate)）。
```bash
pip install -r requirements.txt
```

### 模型训练、测试
- 自行收集图标数据集(考虑到可能出现的版权问题，不放出论文实验使用的图标数据集，请根据论文中图标集的引用信息，自行收集数据集)
- 想实时查看训练结果和损失图，运行`python -m visdom.server`，然后打开http://localhost:8097。
- 训练模型:
```bash
python train.py --dataroot ./datasets/Cardicons2_MBEStyle --name C2M_square_0.618 --gpu_ids 0 \
--no_dropout --no_flip \
--cycle_use_weighted --identity_use_weighted \
--cycle_center 0.618 --identity_center 0.618 --center_type square
```
要查看更多训练过程中的结果，请打开`./checkpoints/C2M_square_0.618/web/index.html`。
- 测试模型:
```bash
python test.py --dataroot ./datasets/Cardicons2_MBEStyle --name C2M_square_0.618 --gpu_ids 0 \
--no_dropout --no_flip --phase test
```
- 测试结果会保存至`./results/C2M_square_0.618/latest_test/index.html`。

### 计算生成图标集与原图标集间的 Circle weighted quadratic loss

```bash
python weighted_loss.py SOURCE_DIR GENERATED_DIR
```

## 引用

如果你的研究使用了本代码，请引用：
```
@article{PWGAN,
  author    = {Yao-Ming Pan and
               Kai-Biao Lin and
               Chin-Ling Chen},
  title     = {Icon-to-Icon Translation using Pixel-Level Weight Adversarial Networks},
  journal   = {Journal of Network Intelligence},
  volume    = {7},
  number    = {1},
  pages     = {147--160},
  year      = {2022}
}
```

## 致谢
我们的代码，很大程度上启发于[CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)。
