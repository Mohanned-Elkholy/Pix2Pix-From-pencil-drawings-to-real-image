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


class Discriminator(object):
    """ This class is responsible for the discriminator """
    def __init__(self,input_dim,lr=0.0002):
        
        self.lr = lr
        self.input_dim = (input_dim,input_dim,3)
        self.optimizer = tf.keras.optimizers.Adam(lr=2e-4, beta_1=0.5,decay=1e-5)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss = 'hinge'
        self.model = self.define_discriminator()

    def residual(self,x,original_channels,BN=True):
        """ This is the residual layer """
        x2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu',padding='same')(x)
        if BN:
            x2 = tf.keras.layers.BatchNormalization()(x2)
        x2 = tf.keras.layers.Conv2D(original_channels, (3, 3), activation='relu',padding='same')(x)
        if BN:
            x2 = tf.keras.layers.BatchNormalization()(x2)
        return x+x2




    def build_model(self,list_of_conv_channels, BN=True, residual = True, num_residual = 5, activation = tf.keras.layers.LeakyReLU()):
        """ This build the model: return None, modify the self.model """
        input_A = tf.keras.Input(shape=(128,128,3))
        input_B = tf.keras.Input(shape=(128,128,3))
        input_layer = tf.keras.layers.Concatenate(axis=-1)([input_A, input_B])
        FS=64
        up_layer_1 = tf.keras.layers.Conv2D(FS, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(input_layer)
        up_layer_2 = tf.keras.layers.Conv2D(FS*2, kernel_size=4, strides=2, padding='same',activation=LeakyReLU(alpha=0.2))(up_layer_1)
        leaky_layer_2 =  tf.keras.layers.BatchNormalization(momentum=0.8)(up_layer_2)
        up_layer_3 = tf.keras.layers.Conv2D(FS*4, kernel_size=4, strides=2,padding='same',activation=LeakyReLU(alpha=0.2))(leaky_layer_2)
        leaky_layer_3 =  tf.keras.layers.BatchNormalization(momentum=0.8)(up_layer_3)
        up_layer_4 = tf.keras.layers.Conv2D(FS*8, kernel_size=4, strides=2,padding='same',activation=LeakyReLU(alpha=0.2))(leaky_layer_3)
        leaky_layer_4 = tf.keras.layers.BatchNormalization(momentum=0.8)(up_layer_4)

        output_layer = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same',activation='sigmoid')(leaky_layer_4)
        self.model =  tf.keras.models.Model([input_A, input_B],output_layer)
        # self.model = self.create_discriminator()
        self.model.compile(loss=self.loss,optimizer = self.optimizer, metrics=['accuracy'])

    def define_discriminator(self,image_shape=(128,128,3)):
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # source image input
        in_src_image = Input(shape=image_shape)
        # target image input
        in_target_image = Input(shape=image_shape)
        # concatenate images channel-wise
        merged = Concatenate()([in_src_image, in_target_image])
        # C64
        d = SpectralNormalization(tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init))(merged)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = SpectralNormalization(tf.keras.layers.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = SpectralNormalization(tf.keras.layers.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init))(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = SpectralNormalization(tf.keras.layers.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init))(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = SpectralNormalization(tf.keras.layers.Conv2D(512, (4,4), padding='same', kernel_initializer=init))(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # patch output
        d = SpectralNormalization(tf.keras.layers.Conv2D(1, (4,4), padding='same', kernel_initializer=init))(d)
        patch_out = (d)
        # define model
        model = tf.keras.models.Model([in_src_image, in_target_image], patch_out)
        # compile model
        model.compile(loss='hinge', optimizer=self.optimizer, loss_weights=[0.5])
        return model

        

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
