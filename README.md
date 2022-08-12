# Custom Object Detection using SSD MobileNet v2

# Summary
 - [Introduction](#Introduction)
 - [Goal](#Goal)
 - [Requirements](#Requirements)
   - [Installed/Updated dependencies and repositories](#Installed/Updated-dependencies-and-repositories)
     - [Google Cloud and Google API Python Client](#Google-Cloud-and-Google-API-Python-Client)
     - [CUDA and CUDNN](#CUDA-and-CUDNN) 
     - [Tensorflow and other Tensorflow dependencies](#Tensorflow-and-other-Tensorflow-dependencies)
     - [LVIS](#LVIS)
   - [Libraries](#Libraries)
     - [Used Libraries](#Used-Libraries)
 - [Code](#Code)
   - [Gathering of the images to use in the training of the neural network](#Gathering-of-the-images-to-use-in-the-training-of-the-neural-network)
   - [Development of the neural network](#Development-of-the-neural-network)
 - [Results](#Results)
 - [What can be improved](#What-can-be-improved)
 - [Conclusion](#Conclusion)

<a name="Introduction"></a>
# Introduction

- This project was developed by Bruno Murça and Rui Borreicho.
- This project was developed for as the intership work for the conclusion of our Bachelor's Degree in Computer Science at [ESGTS](https://siesgt.ipsantarem.pt/esgt/si_main)
- This project was developed during our professional internship at [Capgemini Engineering](https://capgemini-engineering.com/pt/pt-pt/), where we had the opportunity to participate in the V2X initiative at the Embedded and Software Critical Systems unit.

<a name="Goal"></a>
# Goal 

With this project we want to be able to train a neural network model for custom object detection. Specifically, we want to be able to identify vehicles, street signs, traffic lights, people and other objects that can be found when a car is on the being driven. In order to provide us with the images to be used to train our model and also the live feed from the car which will then be used to detect the objects, we will use a software named [CARLA](https://carla.org/).

UPDATE: Because of insufficient requirements to our machines we were not able to install CARLA, so, in order to continue the project the goal has somewhat changed. Instead of using CARLA to get the images used to train our neural network, we used a already existing dataset of images [nuImages](https://www.nuscenes.org/nuimages), and also instead of using the live feed from CARLA to detect the objects we used some images of the same dataset as a form of testing the results of the machine learning. In order to develop the project in the time we also had to limit the number of objects to be identified to include only the cars. 

<a name="Requirements"></a>
# Requirements

This project was developed using Google Colaboratory, so all requirements were already installed and the additional packages required are shown in the cells of the Colab Notebook provided.

<a name="Installed/Updated-dependencies-and-repositories"></a>
## Installed/Updated dependencies and repositories

<a name="Google-Cloud-and-Google-API-Python-Client"></a>
### Google Cloud and Google API Python Client

These were used in order to export the provided images into a Google Drive

```
!pip install --upgrade google-api-python-client
!pip install google-cloud
!pip install google-cloud-vision
```

<a name="CUDA-and-CUDNN"></a>
### CUDA and CUDNN

Installation of the latest version of [CUDA](https://developer.nvidia.com/cuda-zone) and [CUDNN](https://developer.nvidia.com/cudnn)

```
# Check libcudnn8 version
!apt-cache policy libcudnn8

# Install latest version
!apt install --allow-change-held-packages libcudnn8=8.4.1.50-1+cuda11.6

# Export env variables
!export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
!export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
!export LD_LIBRARY_PATH=/usr/local/cuda-11.4/include:$LD_LIBRARY_PATH
!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
```

<a name="Tensorflow-and-other-Tensorflow-dependencies"></a>
### Tensorflow and other Tensorflow dependencies

Installation of [Tensorflow](https://www.tensorflow.org/)

```
# Install tensorflow
!pip install tflite-model-maker==0.4.0
!pip uninstall -y tensorflow && pip install -q tensorflow==2.9.1
!pip install pycocotools==2.0.4
!pip install opencv-python-headless==4.6.0.66

# Check tensorflow version
%tensorflow_version 2.x
!pip show tensorflow

#Install other Tensorflow dependencies
!pip install tf_slim
!pip install tensorflow_io
!pip install -U tf-models-official
!pip install tensorflow-io
```

Installation of the Tensorflow Object Detection API
```
!pip install tensorflow-object-detection-api
```

<a name="LVIS"></a>
### LVIS

Installation of [LVIS](https://www.lvisdataset.org/)

```
!pip install lvis
```

<a name="Libraries"></a>
## Libraries

The installation of the libraries used in this project were not required since they come already installed in Google Colaboratory.

<a name="Used-Libraries"></a>
### Used Libraries

- NumPy
- Matplotlib
- CV2

<a name="Code"></a>
# Code

Our code was divided into two big sections. We will be explaining what each of these sections achieve in the following lines. It is also worth noting that there are two versions of the Colab Notebook we used. The redeNeuralv1.ipynb is the notebook that was created, developed and ran by us, and the second version (name) is a simplified version, without the executable parts of the code, in order to save lines and make it more readable for others.

<a name="Gathering-of-the-images-to-use-in-the-training-of-the-neural-network"></a>
## Gathering of the images to use in the training of the neural network

 1. Creation of the directory where the dataset will be stored and download and decompression of the dataset
 2. Creation of the directory where the images of the dataset will be stored
 3. Import of used libraries
 4. Definition the the source and destiny directories
 5. Code that gets all .jpg files from the dataset and stores them in the created directory
 6. Import of the Google Drive library
 7. Copy of the images directory into a Google Drive

<a name="Development-of-the-neural-network"></a>
## Development of the neural network

 1. Cloning of the Tensorflow Models repository into our project
 2. Instalation of Tenserboard
 3. Configuration of the environment
 4. Gathering of the images and creation of labels/annotations
 5. Cloning of the repository of the project with the images annotated and other files from GitHub
 6. Creation of the label map
 7. Creation of the TFRecords
 8. Setup of our model
 9. Import of libraries
 10. Configuration of the config path
 11. Configuration of the pipeline config file - Definition of the training parameters
 12. Training of the model
 13. Exporting the model
 14. Loading the model
 15. Creation of function to load images into numpy array
 16. Creation of function to run inference for a single image
 17. Doing inference

<a name="Results"></a>
# Results

![result1](https://user-images.githubusercontent.com/65675428/184346600-e3363a03-0431-4e5b-a9a7-16fdede69819.png)
![result2](https://user-images.githubusercontent.com/65675428/184346693-8ba919f0-96c6-4c04-b042-59d0f56b0549.png)

Unfortunately the results were not the best, the network was able to identify a couple of cars but also made some mistakes.

<a name="What-can-be-improved"></a>
# What can be improved

 - Utilzition of a larger and more detailed dataset
 - Increment in the number of training steps in the model
 - Utilization of the CARLA Software
 - Implement a larger number of objects to be identified (people, safety/traffic cones, traffic lights and others)

<a name="Conclusion"></a>
# Conclusion

This software emerged as part of Capgemini's V2X initiative project, with the ultimate goal of providing a neural network that can be used in order to identify cars, people and other objects.

In this project I had the help and collaboration of [Frederico Martins](https://github.com/fredpedroso), [Daniel Sader](https://github.com/danielpontello) and [Rédi Vildo]() from Capgemini.
