
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from google.colab.patches import cv2_imshow
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from math import floor
from keras.datasets import cifar10
import keras
import tensorflow_datasets as tfds
import os
import random
from IPython import display
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
import sys
from keras.layers import Dense, Reshape, Input, BatchNormalization,Concatenate
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Convolution2D,MaxPooling2D,Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizer_v1 import Adam
from keras import initializers
import cv2
from numpy import load
from numpy import zeros
from numpy.random import randint
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.datasets.mnist import load_data
from skimage.transform import resize
from IPython.display import clear_output 



def convert_image(img):
    """ This function converts the image to pencil drawn image """
    def dodgeV2(x, y):                                  # dodging and merging
        return cv2.divide(x, 255 - y, scale=256)
    # convert to grey
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # bitwising
    img_invert = cv2.bitwise_not(img_gray)
    # smoothing and scaling
    img_smoothing = cv2.GaussianBlur(img_invert, (27, 27),sigmaX=-1.0, sigmaY=-1.0)  # blurring by applying Gaussian filter to the inverted image
    final_img = dodgeV2(img_gray, img_smoothing)
    # adjust the shape and return
    pp_image= np.stack([final_img,final_img,final_img],axis=-1)
    return pp_image

def resize_128(img):
    """ resize the image to 128 to fit the UNET """ 
    return cv2.resize(img,(128,128))

def preprocess_the_dataset(x_train):
    """ This function preprocesses, resizes and produce the pencil images the data """
    # initialize empty lists
    x_real,x_pencil=[],[]
    for image in x_train[:8000]:
        real_image = resize_128(image)
        # make it pencil drawn
        pencil_image = convert_image(real_image)
        x_real.append(real_image/127.5-1)
        x_pencil.append(pencil_image/127.5-1)
    return x_real,x_pencil

