# Session 10

## Introduction
 ResNet architecture for CIFAR10 that has the following architecture:
- PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
- Layer1 -
  -  X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
  -   R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
  - Add(X, R1)
- Layer 2 -
  - Conv 3x3 [256k]
  -  MaxPooling2D
  -    BN
  - ReLU
- Layer 3 -
  - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
  - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
  - Add(X, R2)
  - MaxPooling with Kernel Size 4
  - FC Layer 
- ​    SoftMax
- Uses One Cycle Policy such that:
  - Total Epochs = 24
  - Max at Epoch = 5
  - LRMIN = FIND
  - LRMAX = FIND
  - NO Annihilation
- Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
- Batch size = 512
- Use ADAM, and CrossEntropyLoss
- Target Accuracy: 90%
- Kakao Brain's Architecture


## Structure
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
           Dropout-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,856
         MaxPool2d-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
              ReLU-8          [-1, 128, 16, 16]               0
           Dropout-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,584
      BatchNorm2d-11          [-1, 128, 16, 16]             256
             ReLU-12          [-1, 128, 16, 16]               0
          Dropout-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 128, 16, 16]         147,584
      BatchNorm2d-15          [-1, 128, 16, 16]             256
             ReLU-16          [-1, 128, 16, 16]               0
          Dropout-17          [-1, 128, 16, 16]               0
           Conv2d-18          [-1, 256, 16, 16]         295,168
        MaxPool2d-19            [-1, 256, 8, 8]               0
      BatchNorm2d-20            [-1, 256, 8, 8]             512
             ReLU-21            [-1, 256, 8, 8]               0
          Dropout-22            [-1, 256, 8, 8]               0
           Conv2d-23            [-1, 512, 8, 8]       1,180,160
        MaxPool2d-24            [-1, 512, 4, 4]               0
      BatchNorm2d-25            [-1, 512, 4, 4]           1,024
             ReLU-26            [-1, 512, 4, 4]               0
          Dropout-27            [-1, 512, 4, 4]               0
           Conv2d-28            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-29            [-1, 512, 4, 4]           1,024
             ReLU-30            [-1, 512, 4, 4]               0
          Dropout-31            [-1, 512, 4, 4]               0
           Conv2d-32            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-33            [-1, 512, 4, 4]           1,024
             ReLU-34            [-1, 512, 4, 4]               0
          Dropout-35            [-1, 512, 4, 4]               0
        MaxPool2d-36            [-1, 512, 1, 1]               0
           Linear-37                   [-1, 10]           5,130
================================================================
Total params: 6,575,370
Trainable params: 6,575,370
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 8.00
Params size (MB): 25.08
Estimated Total Size (MB): 33.10
----------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F


class CustomResNet(nn.Module):
    # This defines the structure of the NN.
    def __init__(self, dropout_percentage=0, norm="bn", num_groups=2, padding=1):
        super(Net, self).__init__()

        if norm == "bn":
            self.norm = nn.BatchNorm2d
        elif norm == "gn":
            self.norm = lambda in_dim: nn.GroupNorm(
                num_groups=num_groups, num_channels=in_dim
            )
        elif norm == "ln":
            self.norm = lambda in_dim: nn.GroupNorm(num_groups=1, num_channels=in_dim)

        # This defines the structure of the NN.

        # Prep Layer

        self.prep_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 32x32x3 | 1 -> 32x32x64 | 3
            self.norm(64),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )

        self.l1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 32x32x128 | 5
            nn.MaxPool2d(2, 2),  # 16x16x128 | 6
            self.norm(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l1res = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 16x16x128 | 10
            self.norm(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 16x16x128 | 14
            self.norm(128),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 16x16x256 | 18
            nn.MaxPool2d(2, 2),  # 8x8x256 | 19
            self.norm(256),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 8x8x512 | 27
            nn.MaxPool2d(2, 2),  # 4x4x512 | 28
            self.norm(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.l3res = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 4x4x512 | 36
            self.norm(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 4x4x512 | 44
            self.norm(512),
            nn.ReLU(),
            nn.Dropout(dropout_percentage),
        )
        self.maxpool = nn.MaxPool2d(4, 4)

        # Classifier
        self.lr = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.l1(x)
        x = x + self.l1res(x)
        x = self.l2(x)
        x = self.l3(x)
        x = x + self.l3res(x)
        x = self.maxpool(x)
        x = x.view(-1, 512)
        x = self.lr(x)
        return F.log_softmax(x, dim=1)
```

### Metrics
| Train Acc | Test Acc | Train Loss | Test Loss |
|-----------|----------|------------|-----------|
| 97.53     | 92.68    | 0.07       | 0.24      |


## Performance Curve
![image](https://github.com/gopal2812/convandgpt/assets/39087216/8f492fd9-9f40-49a2-9f1d-0cfe87cd8f64)



## Confusion Matrix

![image](https://github.com/gopal2812/convandgpt/assets/39087216/6e3ab54b-a883-4768-b4b1-67a80fbcbdeb)


## Data Exploration

![image](https://github.com/gopal2812/convandgpt/assets/39087216/16f9f69f-2a94-4423-9ab6-76e0927f6023)



```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Train data transformations
train_transforms = A.Compose(
    [
        A.PadIfNeeded(min_height=48, min_width=48, always_apply=True, border_mode=0),
        A.RandomCrop(height=32, width=32, always_apply=True),
        A.HorizontalFlip(p=0.5),
        # A.PadIfNeeded(min_height=64, min_width=64, always_apply=True, border_mode=0),
        A.CoarseDropout(
            p=0.2,
            max_holes=1,
            max_height=8,
            max_width=8,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=(0.4914, 0.4822, 0.4465),
            mask_fill_value=None,
        ),
        # A.CenterCrop(height=32, width=32, always_apply=True),
        A.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ToTensorV2(),
    ]
)

# Test data transformations
test_transforms = A.Compose(
    [
        A.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ToTensorV2(),
    ]
)

```

As seen above, three transforms from the Albumentations library RandomCrop, HoriznotalFlip and CourseDropout were used.

## LR Finder

![image](https://github.com/gopal2812/convandgpt/assets/39087216/a23d0afc-907b-4248-8083-12fc809527b9)


`LR suggestion: steepest gradient
Suggested LR: 2.00E-03`

From the above figure we can see that the optimal lr is found using the steepest gradient at the 2.00E-03 point. Please note the setting for the lr_finder was the following:

```python
from torch_lr_finder import LRFinder
model = Net(dropout_percentage=0.02, norm="bn").to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = F.cross_entropy

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state
```

## Misclassified Images

Total Incorrect Preds = 732

![image](https://github.com/gopal2812/convandgpt/assets/39087216/7e8477ac-8135-4a54-82d0-f27d200d7935)




We see that the misclassified images in all three models have classes very close to each other as misclassified. These misclassified images would be hard for a human to classify correctly too!

## Training Log

```
Train: 100% Loss=0.8755 Batch_id=97 Accuracy=60.40
Test set: Average loss: 1.1007, Accuracy: 6248/10000 (62.48%)

Epoch 3
Train: 100% Loss=0.7879 Batch_id=97 Accuracy=70.89
Test set: Average loss: 0.8401, Accuracy: 7208/10000 (72.08%)

Epoch 4
Train: 100% Loss=0.7300 Batch_id=97 Accuracy=75.84
Test set: Average loss: 0.7401, Accuracy: 7570/10000 (75.70%)

Epoch 5
Train: 100% Loss=0.5914 Batch_id=97 Accuracy=78.51
Test set: Average loss: 0.7848, Accuracy: 7499/10000 (74.99%)

Epoch 6
Train: 100% Loss=0.4339 Batch_id=97 Accuracy=80.93
Test set: Average loss: 0.5935, Accuracy: 7999/10000 (79.99%)

Epoch 7
Train: 100% Loss=0.4903 Batch_id=97 Accuracy=83.79
Test set: Average loss: 0.4936, Accuracy: 8330/10000 (83.30%)

Epoch 8
Train: 100% Loss=0.3822 Batch_id=97 Accuracy=85.25
Test set: Average loss: 0.6265, Accuracy: 8061/10000 (80.61%)

Epoch 9
Train: 100% Loss=0.4612 Batch_id=97 Accuracy=86.72
Test set: Average loss: 0.5000, Accuracy: 8376/10000 (83.76%)

Epoch 10
Train: 100% Loss=0.4004 Batch_id=97 Accuracy=88.20
Test set: Average loss: 0.4411, Accuracy: 8620/10000 (86.20%)

Epoch 11
Train: 100% Loss=0.3473 Batch_id=97 Accuracy=89.24
Test set: Average loss: 0.4209, Accuracy: 8660/10000 (86.60%)

Epoch 12
Train: 100% Loss=0.3076 Batch_id=97 Accuracy=90.17
Test set: Average loss: 0.4411, Accuracy: 8650/10000 (86.50%)

Epoch 13
Train: 100% Loss=0.2566 Batch_id=97 Accuracy=90.95
Test set: Average loss: 0.3850, Accuracy: 8756/10000 (87.56%)

Epoch 14
Train: 100% Loss=0.2242 Batch_id=97 Accuracy=91.69
Test set: Average loss: 0.3220, Accuracy: 8951/10000 (89.51%)

Epoch 15
Train: 100% Loss=0.2346 Batch_id=97 Accuracy=92.47
Test set: Average loss: 0.3544, Accuracy: 8877/10000 (88.77%)

Epoch 16
Train: 100% Loss=0.1399 Batch_id=97 Accuracy=92.96
Test set: Average loss: 0.3131, Accuracy: 9017/10000 (90.17%)

Epoch 17
Train: 100% Loss=0.1584 Batch_id=97 Accuracy=93.54
Test set: Average loss: 0.3169, Accuracy: 8983/10000 (89.83%)

Epoch 18
Train: 100% Loss=0.1754 Batch_id=97 Accuracy=94.34
Test set: Average loss: 0.2972, Accuracy: 9071/10000 (90.71%)

Epoch 19
Train: 100% Loss=0.1350 Batch_id=97 Accuracy=94.93
Test set: Average loss: 0.2968, Accuracy: 9117/10000 (91.17%)

Epoch 20
Train: 100% Loss=0.1026 Batch_id=97 Accuracy=95.56
Test set: Average loss: 0.2853, Accuracy: 9143/10000 (91.43%)

Epoch 21
Train: 100% Loss=0.1324 Batch_id=97 Accuracy=96.13
Test set: Average loss: 0.2651, Accuracy: 9198/10000 (91.98%)

Epoch 22
Train: 100% Loss=0.0735 Batch_id=97 Accuracy=96.75
Test set: Average loss: 0.2571, Accuracy: 9233/10000 (92.33%)

Epoch 23
Train: 100% Loss=0.0934 Batch_id=97 Accuracy=97.17
Test set: Average loss: 0.2528, Accuracy: 9227/10000 (92.27%)

Epoch 24
Train: 100% Loss=0.0718 Batch_id=97 Accuracy=97.53
Test set: Average loss: 0.2449, Accuracy: 9268/10000 (92.68%)
```
