from tqdm.notebook import tqdm

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from google.colab.patches import cv2_imshow
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from math import floor
import tensorflow_addons as tfa
from tensorflow_addons.layers import SpectralNormalization
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
from IPython.display import clear_output 

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
import argparse

from src.generator.utils import *
from src.gan.utils import *
from src.helper_functions.utils import *
from src.train.utils import *
from src.data_manipulation.utils import *
from src.results_functions.utils import *
from src.discriminator.utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--epochs',type=int, required = True)
parser.add_argument('--batchSize',type=int,required = True)

opt = parser.parse_args()
epochs = opt.epochs
batchSize = opt.batchSize

x_train = get_xtrain()
x_real,x_course=preprocess_the_dataset(x_train)
generator = Generator(latent_size=10)
discriminator = Discriminator(input_dim=32)
gan = GAN(discriminator,generator)
train_object = GAN_Train(epochs,batchSize,gan,x_real[:8000-36],x_course[:8000-36],x_real[8000-36:],x_course[8000-36:])
train_object.train_model()



