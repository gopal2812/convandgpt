import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic block of the ResNet.
    """

    expansion = 1

    def __init__(self, in_planes, planes, norm_type, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(1, planes) if norm_type == "LN" else nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(1, planes) if norm_type == "LN" else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, self.expansion * planes)
                if norm_type == "LN"
                else nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        """
        Forward method.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet Architecture.
    """

    def __init__(self, block, num_blocks, norm_type, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(1, 64) if norm_type == "LN" else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], norm_type, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], norm_type, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], norm_type, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], norm_type, stride=1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, norm_type, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_ in strides:
            layers.append(block(self.in_planes, planes, norm_type, stride_))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward method.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(norm_type="BN"):
    """
    ResNet18 Model.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], norm_type)


def resnet34(norm_type="BN"):
    """
    ResNet34 Model.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], norm_type)
