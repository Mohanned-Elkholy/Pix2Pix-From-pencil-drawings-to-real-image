# Pix2Pix-From-pencil-drawings-to-real-image
This is keras implementation of Pix2Pix model on pencil drawn images. You can learn more about Pix2Pix model here (https://arxiv.org/pdf/1611.07004.pdf).


![image](https://user-images.githubusercontent.com/47930821/130732882-ad893c4b-7015-4d5b-81c8-59e06488ceae.png)

# Pix2Pix model
This is one of the family of the style transfer neural network. It basically changes the style of an image to a style of another image. It is considered cGAN (conditional generative network) because it depends on an input to change the style. In this repo, the original image is a pencil drawn image, and the generated image is a fully colored image. Cifar10 is used as the dataset
#add an image here

---
# Pix2Pix archeticture
Pix2Pix model is divided into two models

# Generative model
This model implements a UNET network for image generation. The reason behind the choice of UNET network is its uncanny ability to change the style of the image without affecting the geometric features of the image due to its residual connection between the inner hidden convolutional layers. You can learn more about UNET neural network here (https://arxiv.org/abs/1505.04597)

![image](https://user-images.githubusercontent.com/47930821/130734339-82c0b330-2f9a-40ea-80d8-5b60bcc21cae.png)
# discriminative network
The discriminitive model implements patch GAN loss which basically judge the realisticity of each pixel on its own rather than the whole image. Patch GAN is used here for the sake of making the colors accross the image consistent because squeezing the information in the image into one output (either real or fake) will not be as useful in this problem. 

---
# Loss functions
Pix2Pix implements two loss functions:

# L1 Loss (Pixel-wise loss function)
There are multiple pixel-wise loss function that can be chosen for this task, but after multiple trials, L1 loss worked best. The reason behind this is because the gradient in the L1 loss doesn't depend on the value of the loss itself. Thus, the training becomes more consistent and avoids asymptotic behaviours.

![image](https://user-images.githubusercontent.com/47930821/130596676-1cc4bbc7-0afe-4357-99ec-eb26596d2404.png)

# Adverserial Loss
This is why the discriminator network is here. Training just pixel-wise loss usually produces a blurry image. This is why adding a discriminator to learn the distribution of the real image and forcing the generator to produce within this distribution is crucial.

---

# Converting a colored image to a pencil drawn image
In order to acheive this task, the image is converted to a grey-scale image. Later a bitwise filter is applied on it. Layer a gaussian blur is applied on the image.
You can learn more about the bitwise filter here (https://docs.opencv.org/4.5.2/d0/d86/tutorial_py_image_arithmetics.html).
#add an image here

---
# Prerequisites
1- python3 

2- CPU or NVIDIA GPU (GPU is recommended for faster inversion)

---
# Install dependencies
In this repo, a pretrained biggan in a specified library
```python
pip install torch torchvision matplotlib lpips numpy nltk cv2 pytorch-pretrained-biggan
```
---

# Training
Run this script on a colab notebook to start the inversion. (GPU is require).
```python
!pip install tensorflow_addons
!pip install argparse
!git clone https://github.com/Mohanned-Elkholy/Pix2Pix-From-pencil-drawings-to-real-image
%cd /content/Pix2Pix-From-pencil-drawings-to-real-image
!python main.py --epochs 200 --batchSize 128
```
You can also run the colab notebook provided in the repo, or you can open this link: https://colab.research.google.com/drive/1wJrPeNv7sx8s0s5jCF9XNxW9TmAG6QOO?usp=sharing.

---

# Results

![image](https://user-images.githubusercontent.com/47930821/135886863-9b66f3a8-b2ee-4074-95cc-a09b3e7610f1.png)

