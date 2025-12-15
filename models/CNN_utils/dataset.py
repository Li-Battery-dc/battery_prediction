import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class BatteryDataset(Dataset):
    """
    PyTorch 数据集封装
    输入数据 X 形状应为 (N, 224, 224, 3)
    """
    def __init__(self, X, y_array):
        """
        Args:
            X: numpy array of feature images, shape (N, H, W, C)
            y_array: numpy array of target values, shape (N,)
        """
        self.X = X
        self.y = torch.tensor(y_array, dtype=torch.float32)

        # 定义预训练模型必须的标准化步骤
        self.transform = transforms.Compose([
            transforms.ToPILImage(), # 转换为 PIL Image 以便处理
            transforms.ToTensor(),   # 转回 Tensor 并将 [0, 255] 缩放到 [0.0, 1.0]
            transforms.Normalize(    # ImageNet 标准化
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # 输入 X 是 (224, 224, 3) 的 float64/32 数组，范围 [0, 1]
        img_data = self.X[idx]
        
        # 为了使用 ToPILImage，我们需要将数据转为 uint8 [0, 255]
        # 因为我们之前的 FeatureExtractor 输出的是 [0, 1] 的 float
        img_uint8 = (img_data * 255).astype(np.uint8)
        
        # 应用变换：uint8 -> PIL -> Tensor -> Normalize
        img_tensor = self.transform(img_uint8)
        
        # 获取标签
        target = self.y[idx]
        
        return img_tensor, target