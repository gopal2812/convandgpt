# [Session 5 Assignment](https://canvas.instructure.com/courses/6743641/quizzes/14478657?module_item_id=86577025)

- [Session 5 Assignment](#session-5-assignment)
  - [Assignment Objectives](#assignment-objectives)
  - [Steps followed for solving assignment](#steps-followed-for-solving-assignment)
  - [Module Contents](#module-contents)
  - [Model Summary](#model-summary)
  - [Train and Test Metrics](#train-and-test-metrics)

<br>

## Assignment Objectives

- **Primary Objective**
  - Modularization of the code to improve:
    - Readability of Notebook
    - Improve development speed using reusable functions and classes
- **Secondary Objective**
  - Increase familiarity with Python and PyTorch code
  - Gain understanding of concepts like Convolutional Neural Networks, Fully Connected Networks, Max Pooling etc. using MNIST dataset
  - Debug issues with code and network architecture

<br>

## Steps followed for solving assignment

1. A local Conda environment was created to gain familiarity with local environments for Deep Learning
2. Code from **Session 4** which adds loss function as criterion parameter to train and test functions was debugged and fixed in line with class solution
3. Code was modularized into [_S5.ipynb_](S5.ipynb), [_models.py_](model.py) and [_utils.py_](utils.py). More details present in **Module Contents** section.
4. Assignment and general documentation was added in a [_README.md_](README.md) file

<br>

## Module Contents

- [**models.py**](model.py)
  - Contains a class called Net which defines our network and the forward function
  - Contains lists to hold metrics to track accuracy and loss for both training and testing. These are used by training, testing and plotting related functions.
  - Contains the following model train and test related functions:
    - _train()_: Train the model on the train dataset.
    - _test()_: Test the model on the test dataset.
    - _plot_train_test_metrics()_: Plot the training and test metrics
- [**utils.py**](utils.py)
  - Contains the following utility functions:
    - _get_device()_: Get the device to be used for training and testing
    - _get_correct_prediction_count()_: Get the count of correct predictions given both predictions and labels
    - _plot_sample_training_images()_: Plot sample images from the training data to gain a better understanding of the data
- [**S5.ipynb**](S5.ipynb)
  - Contains our end to end workflow to predict labels for MNIST dataset. This imports functions from utils and the network from models file.
  - The **flow** is as follows:
    - Install any dependencies needed for the script in the working environment
    - Import all external and custom modules for the script
    - Initialize device used for training
    - Define Transformations for the dataset using _apply_mnist_image_transformations()_
    - Download MNIST dataset and split into 2 and apply relevant Transformation definitions
    - Load both test and train data using DataLoader with options like batch size
    - Plot sample input data using _plot_sample_training_images()_
    - Instantiate model and send to device and check out summary
    - Train and test the model for every epoch so that gradients, learning rate and weights are updated to improve accuracy
    - Plot the final test and train metrics using _plot_train_test_metrics()_

<br>

## Model Summary

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 32, 26, 26]             320
                Conv2d-2           [-1, 64, 24, 24]          18,496
                Conv2d-3          [-1, 128, 10, 10]          73,856
                Conv2d-4            [-1, 256, 8, 8]         295,168
                Linear-5                   [-1, 50]         204,850
                Linear-6                   [-1, 10]             510
    ================================================================
    Total params: 593,200
    Trainable params: 593,200
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.67
    Params size (MB): 2.26
    Estimated Total Size (MB): 2.94
    ----------------------------------------------------------------

<br>

## Train and Test Metrics
