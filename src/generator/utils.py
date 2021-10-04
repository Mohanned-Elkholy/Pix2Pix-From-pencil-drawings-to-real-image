
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


class Generator(object):
    """ This class is responsible for the generator """
    def __init__(self,latent_size=8,lr=0.0002):
        self.model = self.define_generator()
        self.lr = lr
        self.optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5,decay=1e-5)
        self.latent_size = latent_size
        self.input_dim = (latent_size,latent_size,1)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def get_random_input(self,batch=32):
        return np.random.normal(0.0,1.0,[batch,self.input_dim[0],self.input_dim[1],self.input_dim[2]])
        # return np.random.choice(self.random_latent,[batch,self.input_dim[0],self.input_dim[1],self.input_dim[2]])

    def residual(self,x,original_channels,BN=True):
        """ This is the residual layer """
        x2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu',padding='same')(x)
        if BN:
            x = tf.keras.layers.BatchNormalization(momentum=0.8)(x2)
        x2 = tf.keras.layers.Conv2D(original_channels, (3, 3), activation='relu',padding='same')(x)
        if BN:
            x = tf.keras.layers.BatchNormalization(momentum=0.8)(x2)
        return x+x2

    
    # define the standalone generator model
    def define_generator(self,image_shape=(128,128,3)):

        def define_encoder_block(layer_in, n_filters, batchnorm=True):
            # weight initialization
            init = RandomNormal(stddev=0.02)
            # add downsampling layer
            g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
            # conditionally add batch normalization
            if batchnorm:
                g = BatchNormalization()(g, training=True)
            # leaky relu activation
            g = LeakyReLU(alpha=0.2)(g)
            return g
        
        # define a decoder block
        def decoder_block(layer_in, skip_in, n_filters, dropout=True):
            # weight initialization
            init = RandomNormal(stddev=0.02)
            # add upsampling layer
            g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
            # add batch normalization
            g = BatchNormalization()(g, training=True)
            # conditionally add dropout
            if dropout:
                g = Dropout(0.5)(g, training=True)
            # merge with skip connection
            g = Concatenate()([g, skip_in])
            # relu activation
            g = Activation('relu')(g)
            return g


        # weight initialization
        init = RandomNormal(stddev=0.02)
        # image input
        in_image = Input(shape=image_shape)
        # encoder model
        e1 = define_encoder_block(in_image, 64, batchnorm=False)
        e2 = define_encoder_block(e1, 128)
        e3 = define_encoder_block(e2, 256)
        e4 = define_encoder_block(e3, 512)
        e5 = define_encoder_block(e4, 512)
        e6 = define_encoder_block(e5, 512)
        # e7 = define_encoder_block(e6, 512)
        # bottleneck, no batch norm and relu
        b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e6)
        b = Activation('relu')(b)
        # decoder model
        # d1 = decoder_block(b, e7, 512)
        d2 = decoder_block(b, e6, 512)
        d3 = decoder_block(d2, e5, 512)
        d4 = decoder_block(d3, e4, 512, dropout=False)
        d5 = decoder_block(d4, e3, 256, dropout=False)
        d6 = decoder_block(d5, e2, 128, dropout=False)
        d7 = decoder_block(d6, e1, 64, dropout=False)
        # output
        g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
        # gr = get_grey_scale()
        # reversed_output = gr(g)
        out_image = Activation('tanh')(g)
        # define model
        model = tf.keras.models.Model(in_image, out_image)
        return model

    def get_model_summary(self):
        """ This function prints the model summary """
        if self.model == None:
            print('Initialize the model first')
        else:
            print(self.model.summary())

    def save_model(self,path):
        if self.model == None:
            print('Initialize the model first')
        else:
            tf.keras.models.save_model(self.model,path)