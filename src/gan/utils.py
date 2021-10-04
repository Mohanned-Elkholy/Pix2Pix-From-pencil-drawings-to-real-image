from IPython.display import clear_output 

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


class GAN(object):
    def __init__(self,discriminator,generator,lr=0.0002):
        self.optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5,decay=1e-5)
        self.gan_generator = generator
        self.gan_discriminator = discriminator
        self.gan_discriminator.model.trainable = False
        self.model = None

        self.loss = discriminator.loss
        self.loss = 'hinge'

        self.build_model()

    def build_model(self):
        """ This function builds the model """
        input_pencil = tf.keras.Input((128,128,3))
        # generator's output
        gen_image = self.gan_generator.model(input_pencil)
        # generator's output
        x = self.gan_discriminator.model([input_pencil,gen_image])
        model = tf.keras.Model(input_pencil,[x,gen_image])
        # compiling the model
        model.compile(loss=['hinge', 'mae'], optimizer = self.optimizer,loss_weights=[1,100], metrics=['accuracy'])
        self.model = model

    def get_model_summary(self):
        if self.model == None:
            print('Initialize the model first')
        else:
            print(self.model.summary())

    def save_model(self,path):
        if self.model == None:
            print('Initialize the model first')
        else:
            tf.keras.models.save_model(self.model,path)


