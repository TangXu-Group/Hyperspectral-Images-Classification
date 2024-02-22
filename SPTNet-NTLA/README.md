# WNet

This repository provides the code for the methods and experiments presented in our paper '**WNet: W-shaped Hierarchical Network for Remote Sensing Image Change Detection**'. (TGRS2023)

**If you have any questions, you can send me an email. My mail address is 21181214261@stu.xidian.edu.cn.**

## Datasets

Download the building change detection dataset. 

- [SVCD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit)
- [WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/)
- [LEVIR-CD](https://justchenhao.github.io/LEVIR/)
- [SYSU-CD](https://github.com/liumency/SYSU-CD)

Prepare datasets into following structure,

```
├─Train
    ├─A
    ├─B
    ├─label
├─Val
    ├─A
    ├─B
    ├─label
├─Test
    ├─A
    ├─B
    ├─label
```

In the following experiments, each image in the dataset is pre-cropped into multiple image patches of size 256 × 256.

## Requirements

>Python 3.7<br>
>PyTorch 1.7.1

## Preparation

* Install DCNv2

```shell
cd DCNv2
python setup.py build develop
cd ..
```

**Attention:** Other versions of Python and PyTorch may cause compilation errors in DCNv2.


* Install other dependencies

All other dependencies can be installed via 'pip'.

* Pretrained weights

Place [ResNet-18](https://download.pytorch.org/models/resnet18-5c106cde.pth) and [DAT-tiny](https://drive.google.com/file/d/1I08oJlXNtDe8jJPxHkroxUi7lYX2lhVc/view?usp=sharing) pretrained weights in `./pretrained`.

## Train

```python
python train.py
```

All the hyperparameters can be adjusted in `./option`.

## Test

```python
python test.py --load_pretrain True --which_epoch 249
```

All the hyperparameters can be adjusted in `./option`.

### Acknowlogdement

This repository is built under the help of the projects [ISNet](https://github.com/xingronaldo/ISNet) and [DAT](https://github.com/LeapLabTHU/DAT) for academic use only.

