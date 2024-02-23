# SPTNet-NTLA

This repository provides the code for the method in our paper '**Spatial Pooling Transformer Network and Noise-Tolerant Learning for Noisy Hyperspectral Image Classification**'. (TGRS2024)
![本地路径](network.png )

**If you have any questions, you can send me an email. My mail address is 22171214790@stu.xidian.edu.cn.**

## Datasets

We conduct experiments on the Indian Pines, University of Pavia, Houston and Houston2018 datasets. To train and test our model, you should download the required data set and modify the corresponding parameters in *main.py* to meet your needs.

## Requirements

>python 3.7<br>
>torch 1.12.1<br>
>scikit-learn 1.0.2<br>
>numpy<br>
>einops

## Train
Before training, the superpixel segmentation map of the corresponding data set should be placed in the *segment* directory. Then by executing the following command, the experimental results can be obtained.
```python
python main.py
```
