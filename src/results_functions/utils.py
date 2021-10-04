
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
import matplotlib.gridspec as gridspec


def show_batch_image(images,status,epoch=0):
	# convert to numpy array
    img = np.array(images)
    # make sure it is normalized
    if np.min(img)<-0.9:
        img = (img+1)*0.495
    # create resized batch
    img = np.array([cv2.resize(i,(32,32)) for i in img])
    num = len(img)
    # set the coordinates for the figure
    ax = np.ceil(np.sqrt(num)) 
    ay = np.rint(np.sqrt(num)) 
    fig = plt.figure(figsize=(64, 64))
    gs = gridspec.GridSpec(6, 6)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(img):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample * 0.5 + 0.5)
    plt.savefig(f'outputs/{epoch}_{status}.png', bbox_inches='tight')
    plt.close(fig)
