# SPTNet-NTLA

This repository provides the code for the method in our paper '**Spatial Pooling Transformer Network and Noise-Tolerant Learning for Noisy Hyperspectral Image Classification**'. (TGRS2024)

**If you have any questions, you can send me an email. My mail address is 22171214790@stu.xidian.edu.cn.**

## Datasets

    We conduct the experiments on the Indian Pines, University of Pavia, and Houston datasets. To train and test our model, you should 
    download the data set and modify image's path according to your needs.

## Requirements

>Python 3.7<br>
>PyTorch 1.12.1
>scikit-learn 1.0.2
numpy

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

