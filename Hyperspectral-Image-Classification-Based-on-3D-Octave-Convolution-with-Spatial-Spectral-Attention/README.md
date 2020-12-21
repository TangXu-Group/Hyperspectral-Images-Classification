# Hyperspectral-Image-Classification-Based-on-3D-Octave-Convolution-with-Spatial-Spectral-Attention
The official codes for paper "Hyperspectral Image Classification Based on 3D Octave Convolution With Spatial-spectral Attention Network"
## Install dependencies
    numpy
    python==3.6
    sklearn
    tensorflow==1.5
    pycharm
## dataset
    We conduct the experiments on the University of Pavia data set. To train and test our model, you should 
    download the data set and modify image's path according to your needs.
## data process 
    Since the input of our network is patch, it is necessary to preprocess the original hyperspectral image. 
    The image preprocessing includes data normalization and patch cutting. 
    In our experiments, the patch size is set to 13*13, and the training set and testing set are divided.
    If you want to divide the data set, run the program `data_prosess.py'      
## train
    All the configurations are diaplayed in `model.py', and you can modify them by your needs. Please download the
    `model.py' for training and testing data set first.    
#### train the model
    Please run the program 'train.py' and save the parameters. In the `train.py', the path to read the data
    should be changed according to your own situation.
## test
    Please read the saved parameters and run the program 'test.py'. In the `test.py', the path to read the data
    should be changed according to your own situation.
    
