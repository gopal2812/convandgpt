# Session 8

## Introduction

This assignment compares different normalization techniques: **Batch Norm, Layer Norm** and **Group Norm**.

We are presented with a multiclass classification problem on the CIFAR10 dataset.

### Target
1. Accuracy > 70%
2. Number of Parameters < 50k
3. Epochs <= 20

Use of Residual Connection is also advised.

## Implementation
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             448
         GroupNorm-2           [-1, 16, 32, 32]              32
              ReLU-3           [-1, 16, 32, 32]               0
           Dropout-4           [-1, 16, 32, 32]               0
            Conv2d-5           [-1, 32, 32, 32]           4,640
         GroupNorm-6           [-1, 32, 32, 32]              64
              ReLU-7           [-1, 32, 32, 32]               0
           Dropout-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 16, 32, 32]             528
        MaxPool2d-10           [-1, 16, 16, 16]               0
           Conv2d-11           [-1, 16, 16, 16]           2,320
        GroupNorm-12           [-1, 16, 16, 16]              32
             ReLU-13           [-1, 16, 16, 16]               0
          Dropout-14           [-1, 16, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]           4,640
        GroupNorm-16           [-1, 32, 16, 16]              64
             ReLU-17           [-1, 32, 16, 16]               0
          Dropout-18           [-1, 32, 16, 16]               0
           Conv2d-19           [-1, 32, 16, 16]           9,248
           Conv2d-20           [-1, 32, 16, 16]             544
        GroupNorm-21           [-1, 32, 16, 16]              64
             ReLU-22           [-1, 32, 16, 16]               0
          Dropout-23           [-1, 32, 16, 16]               0
           Conv2d-24           [-1, 16, 16, 16]             528
        MaxPool2d-25             [-1, 16, 8, 8]               0
           Conv2d-26             [-1, 16, 8, 8]           2,320
        GroupNorm-27             [-1, 16, 8, 8]              32
             ReLU-28             [-1, 16, 8, 8]               0
          Dropout-29             [-1, 16, 8, 8]               0
           Conv2d-30             [-1, 32, 8, 8]           4,640
        GroupNorm-31             [-1, 32, 8, 8]              64
             ReLU-32             [-1, 32, 8, 8]               0
          Dropout-33             [-1, 32, 8, 8]               0
           Conv2d-34             [-1, 32, 8, 8]           9,248
           Conv2d-35             [-1, 32, 8, 8]             544
        GroupNorm-36             [-1, 32, 8, 8]              64
             ReLU-37             [-1, 32, 8, 8]               0
          Dropout-38             [-1, 32, 8, 8]               0
AdaptiveAvgPool2d-39             [-1, 32, 1, 1]               0
           Conv2d-40             [-1, 10, 1, 1]             330
================================================================
Total params: 40,394
Trainable params: 40,394
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.56
Params size (MB): 0.15
Estimated Total Size (MB): 2.72
----------------------------------------------------------------
The above structure with two residual connections is used.

## Normalization Technique Comparison
_Note: We use GN with num_groups = 4_

### Metrics
|    | Train Acc | Test Acc | Train Loss | Test Loss |
|----|-----------|----------|------------|-----------|
| BN | 80.18     | 79.68    | 0.58       | 0.64      |
| GN | 79.62     | 79.26    | 0.68       | 0.72      |
| LN | 80.26     | 79.76    | 0.34      | 0.58       |

## Performance Curves
We see that the graphs portray BN > GN (4 groups) > LN consistently in all the training continues. We explore the reason for this in the next sections.
![image](https://github.com/gopal2812/convandgpt/assets/39087216/f70c8cb1-af1a-49ef-a175-4d8b75524b9a)

## Misclassified Images
**Batch Norm**

Total Incorrect Preds = 2032

![image](https://github.com/gopal2812/convandgpt/assets/39087216/2965a4fb-e178-4d43-96ed-7eb406133618)



**Group Norm**

Total Incorrect Preds = 2074
![image](https://github.com/gopal2812/convandgpt/assets/39087216/2910b4e6-c9e7-48d6-b090-56f8a573e527)


**Layer Norm**

Total Incorrect Preds = 2024
![image](https://github.com/gopal2812/convandgpt/assets/39087216/32c5f564-cc86-4372-8081-6db1ff9cdcf4)


### BN
- Batch Normalization (BN) is a milestone technique in the development of deep learning, enabling various networks to train. However, normalizing along the batch dimension introduces problems — BN’s error increases rapidly when the batch size becomes smaller, caused by inaccurate batch statistics estimation. This limits BN’s usage for training larger models and transferring features to computer vision tasks including detection, segmentation, and video, which require small batches constrained by memory consumption.
- BN normalizes the features by the mean and variance computed within a (mini)batch. This has been shown by many practices to ease optimization and enable very deep networks to converge. The stochastic uncertainty of the batch statistics also acts as a regularizer that can benefit generalization.
- But the concept of “batch” is not always present, or it may change from time to time. For example, batch-wise normalization is not legitimate at inference time, so the mean and variance are pre-computed from the training set, often by running average; consequently, there is no normalization performed when testing. The pre-computed statistics may also change when the target data distribution changes. These issues lead to in-consistency at training, transferring, and testing time.

### Reasoning for GN
- The channels of visual representations are not entirely independent.
- It is not necessary to think of deep neural network features as unstructured vectors. For example, for conv1 (the first convolutional layer) of a network, it is reasonable to expect a filter and its horizontal flipping to exhibit similar distributions of filter responses on natural images. If conv1 happens to approximately learn this pair of filters, or if the horizontal flipping (or other transformations) is made into the architectures by design, then the corresponding channels of these filters can be normalized together.
- For e.g. if the layer learns Horizontal and Vertical edge detectors, they could be grouped together.
- -Specifically, the pixels in the same group are normalized together by the same μ and σ. GN also learns the per-channel γ and β.

### GN
**Relation to LN** `If we set the number of groups to 1, GN becomes LN. LN assumes all channels in a layer make similar contributions and thus restrict the system. GN tries to mitigate this becaquse only each group shares common mean and variance.`

**Relation to IN** `GN becomes IN if we set the number of groups to C (1 group per channel). But IN only relies on the spatial dimension for computing the mean and variance and thus misses the opportunity of exploiting the channel dependence.`

**Effect of Group Number**
- In the extreme case of G = 1, GN is equivalent to LN, and its error rate is higher than all cases of G > 1 studied.
- In the extreme case of 1 channel per group, GN is equivalent to IN. Even if using as few as 2 channels per group, GN has substantially lower error than IN (25.6% vs. 28.4%). This result shows the effect of grouping channels when performing normalization.


