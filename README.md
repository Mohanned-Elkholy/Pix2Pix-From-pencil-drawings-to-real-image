# Pix2Pix-From-pencil-drawings-to-real-image
This is keras implementation of Pix2Pix model on pencil drawn images. You can learn more about Pix2Pix model here (https://arxiv.org/pdf/1611.07004.pdf).


![image](https://user-images.githubusercontent.com/47930821/130732882-ad893c4b-7015-4d5b-81c8-59e06488ceae.png)

# Pix2Pix model
This is one of the family of the style transfer neural network. It basically changes the style of an image to a style of another image. It is considered cGAN (conditional generative network) because it depends on an input to change the style. In this repo, the original image is a pencil drawn image, and the generated image is a fully colored image
#add an image here

---
# Pix2Pix archeticture
Pix2Pix model is divided into two models

# Generative model
This model implements a UNET network for image generation. The reason behind the choice of UNET network is its uncanny ability to change the style of the image without affecting the geometric features of the image due to its residual connection between the inner hidden convolutional layers. You can learn more about UNET neural network here (https://arxiv.org/abs/1505.04597)

![image](https://user-images.githubusercontent.com/47930821/130733364-eaa7dd39-56b6-4f73-85f5-1251f3905841.png)

# discriminative network
The discriminitive model implements patch GAN loss which basically judge the realisticity of each pixel on its own rather than the whole image. Patch GAN is used here for the sake of making the colors accross the image consistent because squeezing the information in the image into one output (either real or fake) will not be as useful in this problem. 
