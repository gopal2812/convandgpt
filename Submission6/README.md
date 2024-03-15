# Assignment 06

**Table of Contents**

- [Assignment 06](#assignment-06)
  - [Objectives](#objectives)
  - [Part 1](#part-1)
    - [Introduction](#introduction)
    - [The Network](#the-network)
    - [Terminologies](#terminologies)
    - [Calculation](#calculation)
    - [Final Calculation in Excel](#final-calculation-in-excel)
    - [Learning Rate vs Errors](#learning-rate-vs-errors)
  - [Part 2](#part-2)
    - [Context](#context)
    - [Architecture Explanation (Chosen Model)](#architecture-explanation-chosen-model)
    - [Model Code](#model-code)
    - [Model summary and parameter count](#model-summary-and-parameter-count)
    - [Accuracy](#accuracy)

## Objectives

- **Part 1**
  - Understanding how backpropagation works by manually implementing it for a simple neural network in Excel
  - Understanding the interplay partial derivatives, gradients, learning rate and weights
  - Assessing impact on loss by changing learning rates
- **Part 2**
  - Building a performant neural network even when there are restrictions on the number of parameters and epochs
  - Practically using a number of concepts introduced in sessions till now and determining how they impact accuracy

<br>
<br>
<br>

## Part 1

### Introduction

We demonstrate how the concept of back propagation is used to improve the neural network. Backpropagation is an algorithm that back-propagates the errors from the output nodes to the input nodes. Therefore, it is simply referred to as the backward propagation of errors.

The entire calculation can be seen in this [Excel file](BackPropagation.xlsx):

<br>

### The Network

The below simple neural network has been chosen in order to make the calculations easier.

![image](https://github.com/gopal2812/convandgpt/assets/39087216/64ec8fe9-fc23-4bde-8777-09ffe7cb92fd)


<br>

### Terminologies

Here are some terminologies needed to understand the neural network.
The network has:

- One input layer (i1 and i2)
- One hidden layer (h1 and h2)
- One output layer (o1 and o2).
- The straight arrows represented by **w** are weights.
- The circular arrows represent the sigmoid activation function
- a_h and a_o are the output of activation function on hidden layer and output layer respectively
- t1 and t2 are the target values
- E is the total loss which is a summation of E1 and E2 calculated by comparing indivdual outputs to target values

<br>

### Calculation

The backpropagation algorithm works by computing the gradient of the loss function with respect to each weight via the chain rule, computing the gradient layer by layer, and iterating backward from the last layer to avoid redundant computation of intermediate terms in the chain rule.


### Learning Rate vs Errors

As Learning rate increases, the value of errors reduces very quickly. Here is the impact of different learning rates on errors in a visual manner:


## Part 2

### Context

Image classification was performed on MNIST data using a deep convolutional neural network. File [Session6.ipynb](Session6.ipynb) contains the entire code and any dependencies needed to execute the code.

The neural networks crafted involved experimenting with concepts such as 2D Convolution, Fully connected and Gap layers, Max pooling, Padding, Dropout, Batch normalization, ReLU as the activation function etc.

<br>

### Architecture Explanation (Chosen Model)

After multiple iterations, the following network was created

- **Block 1**: It consists of a series of operations applied to the input image.
  - Convolutional layer: Extracts features from the input image using 16 filters of size 3x3.
  - ReLU activation: Applies the rectified linear unit function, which introduces non-linearity.
  - Batch normalization: Normalizes the outputs of the previous layer to stabilize the learning process.
  - Another convolutional layer with the same configurations.
  - ReLU activation and batch normalization.
  - Padding: Adds a border of 1 pixel to the feature maps to preserve spatial dimensions.
  - ReLU activation and batch normalization.
  - Dropout: Randomly sets a fraction of inputs to 0 during training to prevent overfitting.
- **Block 2**:
  - Convolutional layer: Uses 16 filters of size 3x3 to extract more features.
  - ReLU activation and batch normalization.
  - Another convolutional layer with the same configurations.
  - ReLU activation and batch normalization.
  - Padding, ReLU activation, and batch normalization.
  - Max pooling: Reduces the spatial dimensions of the feature maps by taking the maximum value in each 2x2 region.
  - Dropout.
- **Block 3**:
  - Convolutional layer: Extracts features using 16 filters of size 3x3.
  - ReLU activation and batch normalization.
  - Another convolutional layer with the same configurations.
  - ReLU activation and batch normalization.
  - Padding, ReLU activation, and dropout.
- **Block 4**: - This block performs the final operations to produce the classification output.
  - Average pooling: Reduces the spatial dimensions to 1x1 by taking the average value in each 5x5 region.
  - Flattening: Reshapes the output into a 1-dimensional vector.
  - Linear layer: Maps the flattened features to the output classes (10 classes in this case).

<br>

### Model Code

```
# Class to define the NN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Using Sequential API to define the model as it seems to be more readable

        # General Notes

        # ReLU used after every Convolution layer
        # Batch Normalization used after every Convolution layer
        # Dropout used after every block/ layer
        # Max Pooling preferably used after every block
        # GAP used at the end of the model

        # Sequence in a block could be as follows
        # Convolution Layer -> ReLU -> Batch Normalization -> Max Pooling -> Dropout

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # nn.Dropout(0.1),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Gap layer
        self.block4 = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.AvgPool2d(kernel_size=5),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        # print(x.shape)

        x = self.block2(x)
        # print(x.shape)

        x = self.block3(x)
        # print(x.shape)

        x = self.block4(x)
        # print(x.shape)

        return F.log_softmax(x)
```

<br>

### Model summary and parameter count

The model summary and parameter count is as follows:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             160
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
            Conv2d-4           [-1, 16, 24, 24]           2,320
              ReLU-5           [-1, 16, 24, 24]               0
       BatchNorm2d-6           [-1, 16, 24, 24]              32
            Conv2d-7           [-1, 16, 24, 24]           2,320
              ReLU-8           [-1, 16, 24, 24]               0
       BatchNorm2d-9           [-1, 16, 24, 24]              32
          Dropout-10           [-1, 16, 24, 24]               0
           Conv2d-11           [-1, 16, 22, 22]           2,320
             ReLU-12           [-1, 16, 22, 22]               0
      BatchNorm2d-13           [-1, 16, 22, 22]              32
           Conv2d-14           [-1, 16, 20, 20]           2,320
             ReLU-15           [-1, 16, 20, 20]               0
      BatchNorm2d-16           [-1, 16, 20, 20]              32
           Conv2d-17           [-1, 16, 20, 20]           2,320
             ReLU-18           [-1, 16, 20, 20]               0
      BatchNorm2d-19           [-1, 16, 20, 20]              32
        MaxPool2d-20           [-1, 16, 10, 10]               0
          Dropout-21           [-1, 16, 10, 10]               0
           Conv2d-22             [-1, 16, 8, 8]           2,320
             ReLU-23             [-1, 16, 8, 8]               0
      BatchNorm2d-24             [-1, 16, 8, 8]              32
           Conv2d-25             [-1, 16, 6, 6]           2,320
             ReLU-26             [-1, 16, 6, 6]               0
      BatchNorm2d-27             [-1, 16, 6, 6]              32
           Conv2d-28             [-1, 16, 6, 6]           2,320
             ReLU-29             [-1, 16, 6, 6]               0
          Dropout-30             [-1, 16, 6, 6]               0
        AvgPool2d-31             [-1, 16, 1, 1]               0
          Flatten-32                   [-1, 16]               0
           Linear-33                   [-1, 10]             170
================================================================
Total params: 19,146
Trainable params: 19,146
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.28
Params size (MB): 0.07
Estimated Total Size (MB): 1.36
----------------------------------------------------------------
```

<br>

### Accuracy

The test accuracy is as follows:

![image](https://github.com/gopal2812/convandgpt/blob/main/Submission6/Highlight.png?raw=true)
