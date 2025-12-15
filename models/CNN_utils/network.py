import torch
import torch.nn as nn
from torchvision import models

class BatteryAlexNet(nn.Module):
    """
    AlexNet for Battery Lifetime Prediction (Regression)
    基于论文：使用预训练的 AlexNet 并进行微调
    """
    def __init__(self, pretrained=True):
        super(BatteryAlexNet, self).__init__()
        
        # 加载预训练的 AlexNet
        # weights='DEFAULT' 相当于 pretrained=True，加载 ImageNet 权重
        if pretrained:
            self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        else:
            self.model = models.alexnet(weights=None)
        
        # 修改分类器部分
        # AlexNet 的 classifier 部分结构如下:
        # (0): Dropout
        # (1): Linear(256 * 6 * 6 -> 4096)
        # (2): ReLU
        # (3): Dropout
        # (4): Linear(4096 -> 4096)
        # (5): ReLU
        # (6): Linear(4096 -> 1000) (原始输出层)

        for param in self.model.features.parameters():
            param.requires_grad = False
        
        # 获取最后一个全连接层的输入特征数
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 256),  # 9216 -> 256 (大幅减少参数)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),           # 256 -> 64
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)              # 输出层
        )
        
    def forward(self, x):
        # 输入 x: (Batch, 3, 224, 224)
        return self.model(x)

# 如果想尝试论文中提到的简单 TCNN，也可以备选如下：
class TCNN(nn.Module):
    """
    Simple CNN (TCNN) from the paper
    4 layers, 3x3 kernels, ReLU
    """
    def __init__(self):
        super(TCNN, self).__init__()
        # 这是一个简化的复现，具体通道数论文未完全详述，这里参考常见配置
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 224 -> 112 -> 56 -> 28 -> 14
        self.classifier = nn.Sequential(
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x