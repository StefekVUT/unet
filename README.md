# Implementation of deep learning framework -- Unet, using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Data

TEM images, 200 kV, imaging modes: bright/dark field, STEM, magnification range 2k-5000kx.
 The initial non-augmented training dataset consists of images and appropriate binary masks of different elements
 and different samples(Au, C, Zn, Bronz, KCl, Spinel, Az, Grid).

### Pre-processing


The data for training contains 100 512*512 images, which are far not enough to feed a deep learning neural network.
Prediction data contains 6 512*512 images.
To do data augumentation, an image augmentation method was used, which was implemented with the [Augmentor tool](https://github.com/StefekVUT/Augmentor)
All images were firstly rotated by 90, 180 and 270 degrees. Then Rotation, Zoom, Distort functions from Augmentor tool were applied.
Current best results were achieved when training dataset contained 10 000 images.

### Model

![img/u-net-architecture.png](img/u-net-architecture.png)

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

### Training

The model is trained for 10 epochs.

After 10 epochs, calculated accuracy is about FILL LATER

Loss function for the training is basically just a binary crossentropy

---

## How to use

FILL LATER

### Dependencies

This tutorial depends on the following libraries:

* Tensorflow
* Keras >= 1.0

Also, this code should be compatible with Python versions 2.7-3.5.

### Prepare the data

First transfer 3D volume tiff to 30 512*512 images.

To feed the unet, data augmentation is necessary.






### Define the model

* Check out ```get_unet()``` in ```unet.py``` to modify the model, optimizer and loss function.

### Train the model and generate masks for test images

* Run ```python unet.py``` to train the model.


After this script finishes, in ```imgs_mask_test.npy``` masks for corresponding images in ```imgs_test.npy```
should be generated. I suggest you examine these masks for getting further insight of your model's performance.

### Results

Use the trained model to do segmentation on test images, the result is statisfactory.

![img/0test.png](img/0test.png)

![img/0label.png](img/0label.png)


## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.
supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.
Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.
