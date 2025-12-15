import torch
import torch.nn as nn
from torchvision import models

class BatteryAlexNet(nn.Module):
    """
    AlexNet for Battery Lifetime Prediction (Regression)
    基于论文：使用预训练的 AlexNet 并进行微调
    """
    def __init__(self, pretrained=True, freeze_up_to=0): # 冻结最前面的卷积层
        super(BatteryAlexNet, self).__init__()
        
        # 加载预训练的 AlexNet
        # weights='DEFAULT' 相当于 pretrained=True，加载 ImageNet 权重
        if pretrained:
            self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        else:
            self.model = models.alexnet(weights=None)

        # 精细化冻结策略
        # 我们遍历 features 的所有子层
        child_counter = 0
        for child in self.model.features.children():
            # 只有卷积层是有参数的，ReLU/Pool 没有参数
            # 如果索引小于阈值，就冻结；否则解冻
            if child_counter < freeze_up_to:
                for param in child.parameters():
                    param.requires_grad = False
                # 打印日志确认
                # print(f"Layer {child_counter} ({type(child).__name__}): FROZEN")
            else:
                for param in child.parameters():
                    param.requires_grad = True
                # print(f"Layer {child_counter} ({type(child).__name__}): TRAINABLE")
            
            child_counter += 1
        
        # 修改分类器部分
        # AlexNet 的 classifier 部分结构如下:
        # (0): Dropout
        # (1): Linear(256 * 6 * 6 -> 4096)
        # (2): ReLU
        # (3): Dropout
        # (4): Linear(4096 -> 4096)
        # (5): ReLU
        # (6): Linear(4096 -> 1000) (原始输出层)
        # 我们将其替换为适合回归任务的结构，防止多参数过拟合
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),            # 保持高 Dropout
            nn.Linear(256 * 6 * 6, 256),  # 9216 -> 512
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),            # 第二次 Dropout
            nn.Linear(256, 128),           # 进一步降维
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)              # 输出层
        )
        
        
        
    
    def forward(self, x):
        # 输入 x: (Batch, 3, 224, 224)
        return self.model(x)