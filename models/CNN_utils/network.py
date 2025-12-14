import torch
import torch.nn as nn

class CNN_BLSTM(nn.Module):
    def __init__(self, input_length=1000, input_channels=1, 
                 cnn_filters=[32, 64], kernel_sizes=[5, 3], 
                 lstm_hidden_size=64, lstm_layers=1, dropout=0.3):
        """
        Args:
            input_length: 输入序列长度 (例如 1000 个电压点)
            input_channels: 输入特征数 (例如 1 代表仅电压)
            cnn_filters: 卷积层滤波器数量列表
            kernel_sizes: 卷积核大小列表
            lstm_hidden_size: LSTM 隐藏层维度
            lstm_layers: LSTM 层数
            dropout: Dropout 比率
        """
        super(CNN_BLSTM, self).__init__()
        
        # --- 1. 1D CNN 特征提取模块 ---
        self.cnn_layers = nn.ModuleList()
        in_c = input_channels
        
        for out_c, k_size in zip(cnn_filters, kernel_sizes):
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, padding=k_size//2),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2) # 降采样，减少序列长度
            ))
            in_c = out_c
            input_length = input_length // 2 # MaxPool 减半长度
            
        self.dropout = nn.Dropout(dropout)
        
        # --- 2. BLSTM 时序建模模块 ---
        # CNN 输出: (Batch, Channels, Length) -> 需要 permute 为 (Batch, Length, Channels) 给 LSTM
        self.lstm = nn.LSTM(
            input_size=in_c,  # CNN 的输出通道数作为 LSTM 的输入特征数
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # --- 3. 回归预测模块 ---
        # 双向 LSTM 输出维度是 hidden_size * 2
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # 输出 log(cycle_life)
        )

    def forward(self, x):
        # x shape: (Batch, Length, Channels) -> 需要转换为 (Batch, Channels, Length) 给 Conv1d
        x = x.permute(0, 2, 1)
        
        # CNN Forward
        for layer in self.cnn_layers:
            x = layer(x)
        
        # 准备输入 LSTM: (Batch, Channels, Length) -> (Batch, Length, Channels)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        
        # LSTM Forward
        # output shape: (Batch, Seq_Len, Hidden*2)
        # hidden/cell shape: (Layers*2, Batch, Hidden)
        output, (hn, cn) = self.lstm(x)
        
        # 使用最后一个时间步的输出，或者使用 Global Average Pooling
        # 这里使用最后一个时间步的输出 (包含双向信息)
        # output[:, -1, :] 取最后一个时间步
        x = output[:, -1, :] 
        
        # Regression
        out = self.regressor(x)
        return out.squeeze(1) # 返回 (Batch,)