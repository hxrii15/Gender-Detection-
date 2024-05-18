# Gender-Detection-
Gender detection using Machine Learning || CNN 

**ABSTRACT**
 Gender detection is a method of recognizing if there is a face in an image. The goal of face analysis is to detect and identify the different gender. A gender categorizing model uses face from a given image to predict the gender (male or female) based on their appearance like baldness, long hair, dimple, beard and moustache and geometric facial structure by using Machine Learning Technology.

**HOW THE MACHINE IS TRAINED.?**

MACHINE LEARNING
Machine learning is when computers learns from data to make decision without explicit programming. 

DEEP LEARNING
Deep learning which is a subset of machine learning that uses several layers neural network to repeatedly gain higher level of features from the given input images.

NEURAL NETWORKS
A neural network is a sequence of process that is capable to identify hidden relationships in a set of data and the process is similar to the operation of human brain. 

DATA SET USED FOR THE TRAINING

**Kaggle Dataset:**

Data: 1747 male and 1747 female training images, 100 test and validation images for each gender.
Preprocessing: Face cropping using (Multi-task) MTCNN to isolate facial images. 

**Nottingham Scan Database:**

Data: 100 human faces, 50 male and 50 female.
Image Format: .gif, converted to .jpg.
These datasets were used for gender estimation in the study.

HOW CNN DO CLASSIFY THE DATA.?
CNNs work through multiple layers:
1. Convolution Layer: It uses filters to perform convolutions, detecting features.
2. ReLU Layer: This applies rectified linear unit operations for feature mapping. ( Rectified Linear Unit )
3. Pooling Layer: It down-samples feature maps, reducing their dimensions.
4. Fully Connected Layer: It classifies and identifies images after flattening the pooled feature map.

**CONCLUSON**
In this project we have used both image processing technique and machine learning algorithm for implementation and achieved a promising result for both Kaggle dataset and Nottingham Scan Database. As part of image processing, a pre-processing technique has been applied first. After pre-processing, feature extraction and classification are implemented in this system.  A sigmoid function has been used as classifier in our model. Different optimizers have been used to determine which optimizer gives a better result. For assessing the effectiveness of our model, we have applied 5-fold cross-validation which has helped to evaluate our model. After analysing the result, a comparison of two previous work with our paper has also been shown where our system gives better result than them


