
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
from tqdm.notebook import tqdm

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
from src.results_functions.utils import *



class GAN_Train(object):
    def __init__(self,epochs,batch_size,gan_model,x_train,x_pencil,real_images_test,pencil_images_test,ratio_d_g=1):
        self.epochs = epochs
        self.pencil_images_test = pencil_images_test
        self.real_images_test = real_images_test
        self.batch_size = batch_size
        self.gan_model = gan_model
        self.x_train = x_train
        self.x_pencil = x_pencil
        self.ratio_d_g = ratio_d_g
        self.inception_list = []


    def plot_output(self,i):
        """ This function plots real pencil and model's output """
        print('pencil')
        show_batch_image(self.pencil_images_test[:36],'pencil',i)
        print('real')
        show_batch_image(self.real_images_test[:36],'real',i)
        print('fake')
        fake_images_test = self.gan_model.gan_generator.model(np.reshape(self.pencil_images_test[:36],(-1,128,128,3)))
        print(np.min(fake_images_test),np.max(fake_images_test))
        show_batch_image(fake_images_test,'fake',i)    	


    def train_one_epoch(self,print_every_period,generator_loss,discrimantor_loss,test=False):
        """ This function trains one epoch """
        for i in tqdm(range(int(len(self.x_train)/self.batch_size)-1)):
        	# adjust the shapes of real and pencil images
            pencil_images = np.reshape(self.x_pencil[i*self.batch_size:(i+1)*self.batch_size],(-1,128,128,3))
            real_images = np.reshape(self.x_train[i*self.batch_size:(i+1)*self.batch_size],(-1,128,128,3))
            
            # initialize the labels
            the_generator_label = np.ones((self.batch_size,8,8,1))*-1
            the_discriminator_label = the_generator_label*-1

            # the generator
            x_batch_generator = self.gan_model.gan_generator.model(pencil_images)

            # the discriminator
            x_batch_discriminator_images = np.concatenate([x_batch_generator,real_images])
            x_batch_discriminator_pencil = np.concatenate([pencil_images,pencil_images])

            # train generator
            generator_loss = self.gan_model.model.train_on_batch(pencil_images,[the_generator_label,real_images])

            # train discriminator
            # on fake images
            discrimantor_loss_1 = self.gan_model.gan_discriminator.model.train_on_batch([pencil_images,x_batch_generator],the_discriminator_label)
            # on real images
            discrimantor_loss_2 = self.gan_model.gan_discriminator.model.train_on_batch([pencil_images,real_images],the_generator_label)
            # comnbine both
            discrimantor_loss = 0.5*(discrimantor_loss_1+discrimantor_loss_2)

        print(f"generator loss is {generator_loss}")
        print(f"discriminator loss is {discrimantor_loss}")
        return generator_loss,discrimantor_loss


    def train_model(self):
        generator_loss,discrimantor_loss = 0,0
        for i in range(self.epochs):
            if i%50==0:
                clear_output()   

            print(f"epoch number: {i}")             
            generator_loss,discrimantor_loss=self.train_one_epoch(i%200==0,generator_loss,discrimantor_loss)

            self.plot_output(i)


    def generate_image(self):
        the_random_input = self.gan_model.gan_generator.get_random_input(self.batch_size)
        # the generator
        x_batch_generator = self.gan_model.gan_generator.model([the_random_input,np.random.randint(0,9,self.batch_size)])
        return x_batch_generator
    


