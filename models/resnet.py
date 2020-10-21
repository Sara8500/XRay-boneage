'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, self.expansion * out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_in_channels=1, num_out_channels=1): #num_out_channels = num of classes, num_in_channels= RGB or gray scale
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(num_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(512 * block.expansion, num_out_channels)
        print("constructor ResNet18: nn.Linear( x, x) ", 512 * block.expansion, num_out_channels)

    def _make_layer(self, block, out_planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_planes, stride))
            self.in_planes = out_planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        #print("conv1 : out.size: ", out.size())
        out = self.bn1(out)
        #print("bn1: out.size: ", out.size())
        out = self.maxpool(out)
        #print("maxpool: out.size: ", out.size())
        out = self.layer1(out)
        #print("layer1: out.size: ", out.size())
        out = self.layer2(out)
        #print("layer2: out.size: ", out.size())
        out = self.layer3(out)
        #print("layer3: out.size: ", out.size())
        out = self.layer4(out)
        #print("layer4: out.size: ", out.size())
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        #print("view : out.size: ", out.size())
        out = self.linear(out)
        #print("end : out.size: ", out.size())
        return out

    def init_params(self, mode):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if mode == "kaiming_uniform":
                    nn.init.kaiming_uniform_(m.weight)
                elif mode == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight)
                elif mode == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif mode == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def resnet18(**kwargs):

    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs) #2 classes


def resnet34(**kwargs):

    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):

    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet101(**kwargs):

    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):

    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
