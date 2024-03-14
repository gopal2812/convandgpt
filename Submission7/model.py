import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

dropout_value = 0.1

class Model_1(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Model_1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, stride=2),
            # nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.avgpool = nn.AvgPool2d(5)
        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 32)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


class Model_2(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Model_2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 11, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(11),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(11, 10, kernel_size=1),
            nn.ReLU(),
            # nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            #nn.Dropout(0.1),
	    nn.Conv2d(10, 10, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 22, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(22),
            nn.Conv2d(22, 10, kernel_size=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

class Model_3(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Model_3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),  # 28 >> 26
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, kernel_size=3), # 26 >> 24
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 11, kernel_size=3), # 24 >> 22
            nn.ReLU(),
            nn.BatchNorm2d(11),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(11, 10, kernel_size=1), #22 >> 22
            nn.ReLU(),
            # nn.BatchNorm2d(10),
            nn.Conv2d(10, 10, kernel_size=3), #22 >> 20
            nn.ReLU(),
            nn.BatchNorm2d(10),
            #nn.Dropout(0.1),
            nn.Conv2d(10, 10, kernel_size=3), #22 >> 18
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Conv2d(10, 16, kernel_size=3), #22 >> 16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 10, kernel_size=1), #16 >> 16
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)  #1

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)
