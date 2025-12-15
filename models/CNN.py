from models.base import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import time
import copy

from models.CNN_utils.network import BatteryAlexNet
from models.CNN_utils.dataset import BatteryDataset

class CNNModel(BaseModel):
    """基于 CNN-BLSTM 的深度学习模型实现"""
    
    def __init__(self, config=None, cnn_config=None):
        super().__init__(config, cnn_config)
        
        # 从 CNNConfig 中获取 ModelConfig
        if cnn_config is not None:
            model_config = cnn_config.get_model_config()
        else:
            # 使用默认配置
            from config import CNNConfig
            model_config = CNNConfig.get_model_config()
        
        # 获取模型配置参数
        self.batch_size = getattr(model_config, 'batch_size', 32)
        self.epochs = getattr(model_config, 'epochs', 100)
        self.lr = getattr(model_config, 'learning_rate', 0.001)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        print(f"   - Model initialized on device: {self.device}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """训练模型"""
        
        # 1. 准备数据
        train_dataset = BatteryDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = BatteryDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
        # 2. 初始化网络

        self.model = BatteryAlexNet(pretrained=True).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # 3. 训练循环
        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(self.model.state_dict())
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"   - Starting training for {self.epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # --- Training ---
            self.model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # outputs 形状是 (Batch, 1), targets 形状是 (Batch,)
                # 需要 targets.unsqueeze(1) 将 targets 形状变为 (Batch, 1)
                targets = targets.unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            history['train_loss'].append(epoch_loss)
            
            # --- Validation ---
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        targets = targets.unsqueeze(1)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                
                epoch_val_loss = val_loss / len(val_loader.dataset)
                history['val_loss'].append(epoch_val_loss)
                scheduler.step(epoch_val_loss)
                
                # Save best model
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                
                if (epoch + 1) % 10 == 0:
                    print(f"     Epoch {epoch+1}/{self.epochs} - Train mse Loss: {epoch_loss:.4f} - Val mse Loss: {epoch_val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"     Epoch {epoch+1}/{self.epochs} - Train mse Loss: {epoch_loss:.4f}")
                    
        time_elapsed = time.time() - start_time
        print(f"   - Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        
        # Load best weights
        if val_loader:
            self.model.load_state_dict(best_model_wts)
            
        self.is_fitted = True
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """模型推理"""
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
            
        # 准备数据 (不需要 y)
        # 创建一个 dummy y 用于 Dataset 初始化
        dummy_y = np.zeros(len(X))
        dataset = BatteryDataset(X, dummy_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                # outputs 形状是 (Batch, 1)，需要展平为 (Batch,)
                predictions.append(outputs.cpu().squeeze().numpy())
                
        return np.concatenate(predictions)

    def get_model_params(self):
        """保存模型参数用于复现"""
        return {
            'optimizer': 'Adam',
            'batch_size': self.batch_size,
            'learning_rate': self.lr
        }