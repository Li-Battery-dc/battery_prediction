"""
CNN Feature extraction module for battery cycle life prediction
Extracts raw time-series features for deep learning models
"""
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.preprocessing import MinMaxScaler
from config import Config
import cv2

class CNNFeatureExtractor():
    """CNN feature extractor for raw time-series data"""
    
    def __init__(self, config: Config = None):
        """Initialize CNN feature extractor
        
        Args:
            config: Configuration object
        """
        self.config = config if config else Config()
        self.normalize = self.config.NORMALIZE_FEATURES
        self.log_transform_target = self.config.LOG_TRANSFORM_TARGET
        
        self.max_cycles = self.config.FEATURE_CYCLE_END # 只用前100个循环
        self.voltage_points = self.config.CNN_VOLTAGE_POINTS  # 电压采样分辨率
        self.target_size = (224, 224)  # CNN输入图像大小
        self.v_low = 2.0
        self.v_high = 3.5
        self.ref_voltage = np.linspace(self.v_low, self.v_high, self.voltage_points)

        # 论文提到使用仿射变换将像素归一化 (black=min, white=max)
        self.pixel_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
    
    def extract_features(self, battery_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 3-channel graphical features from battery data.
        
        Args:
            battery_data: Dict where key is cell_id, value is a dict containing 
                          'cycles' (dict of dfs) and 'summary' (df).
                          Assuming cell_data['cycles'][k] is a DataFrame with 'V' and 'Q' columns.
                          
        Returns:
            X: numpy array of shape (N, 100, 100, 3)
            y: numpy array of shape (N,)
        """
        features_list = []
        targets_list = []
        
        print(f"   [FeatureExtractor] Starting graphical feature extraction for {len(battery_data)} batteries...")

        valid_cell_ids = []

        # --- 1. 提取原始特征矩阵 ---
        # 我们先收集所有电池的原始 F1, F2, F3 数据，然后再统一进行归一化
        raw_f1_list = [] # List of (100, 100) matrices
        raw_f2_list = []
        raw_f3_list = []

        for cell_id, cell_data in battery_data.items():
            # 获取真实寿命 (根据你的数据结构调整，这里假设在 summary 中)
            cycle_life = cell_data['cycle_life'].reshape(-1)[0]

            # 处理单个电池，获取 (100, 100, 3) 的原始数据
            f1_matrix, f2_matrix, f3_matrix = self._process_single_cell(cell_data)
            
            raw_f1_list.append(f1_matrix)
            raw_f2_list.append(f2_matrix)
            raw_f3_list.append(f3_matrix)
            targets_list.append(cycle_life)
            valid_cell_ids.append(cell_id)

        if len(raw_f1_list) == 0:
            raise ValueError("No valid battery data found!")

        # 转换为 Numpy 数组: (N, 100, 100)
        X_f1 = np.array(raw_f1_list)
        X_f2 = np.array(raw_f2_list)
        X_f3 = np.array(raw_f3_list)
        y = np.array(targets_list, dtype=np.float32)

        print(f"   [FeatureExtractor] Extracted raw features. Shape per channel: {X_f1.shape}")

        # --- 2. 数据归一化 (Affine Transformation) ---
        # 论文提到：Pixel data are obtained by affine transformation 
        # 这通常意味着 Min-Max Scaling。
        # 我们可以对整个数据集的每个通道进行全局归一化，或者对每个样本单独归一化。
        # 深度学习中通常对整个数据集进行统计归一化。
        
        if not self.is_fitted:
            # 针对每个通道分别计算 Min 和 Max
            self.f1_min, self.f1_max = X_f1.min(), X_f1.max()
            self.f2_min, self.f2_max = X_f2.min(), X_f2.max()
            self.f3_min, self.f3_max = X_f3.min(), X_f3.max()
            self.is_fitted = True
            print(f"   [FeatureExtractor] Scaler fitted. F1 range: [{self.f1_min:.3f}, {self.f1_max:.3f}]")

        # 应用归一化到 [0, 1]
        X_f1_norm = (X_f1 - self.f1_min) / (self.f1_max - self.f1_min + 1e-8)
        X_f2_norm = (X_f2 - self.f2_min) / (self.f2_max - self.f2_min + 1e-8)
        X_f3_norm = (X_f3 - self.f3_min) / (self.f3_max - self.f3_min + 1e-8)

        # --- 3. 堆叠通道 ---
        # 最终形状: (N, 100, 100, 3) [cite: 437, 583]
        # 轴顺序: (Samples, Cycles, Voltage_Steps, Channels)
        # 注意：如果是 PyTorch，通常需要 permute 到 (N, 3, 100, 100)
        X_stacked = np.stack([X_f1_norm, X_f2_norm, X_f3_norm], axis=-1)

        # --- 4. 图像resize ---
        # 适应图像卷积网络输入大小 (e.g., 224x224)
        print(f"   [FeatureExtractor] Resizing images from (100, 100) to {self.target_size}...")
        X_resized = []
        for i in range(X_stacked.shape[0]):
            # cv2.resize 接收 (width, height)
            # 输入 img 是 (100, 100, 3)
            img_resized = cv2.resize(X_stacked[i], self.target_size, interpolation=cv2.INTER_CUBIC)
            X_resized.append(img_resized)
        
        X = np.array(X_resized) # Shape: (N, 224, 224, 3)

        # --- 5. 目标值变换 ---
        if self.log_transform_target:
            y = np.log10(y)

        print(f"   [FeatureExtractor] Final X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def _process_single_cell(self, cell_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        处理单个电池数据，生成 F1, F2, F3 矩阵
        """
        # 初始化矩阵 (Cycles=100, Voltage_Steps=100)
        mat_f1 = np.zeros((self.max_cycles, self.voltage_points))
        mat_f2 = np.zeros((self.max_cycles, self.voltage_points))
        mat_f3 = np.zeros((self.max_cycles, self.voltage_points))
        
        all_cycles_data = cell_data.get('cycles', {})
            
        # 获取第1个循环的基准 Q(V)
        q_ref = self._get_interpolated_q(all_cycles_data[1] if 1 in all_cycles_data else all_cycles_data['1'])

        # 遍历前100个循环
        for i in range(self.max_cycles):
            cycle_idx = str(i + 1)  # 忽略0号循环，循环编号从1开始
            
            if cycle_idx not in all_cycles_data:
                raise ValueError(f"Cycle {cycle_idx} data missing for cell.")
            
            # 获取当前循环的 DataFrame
            cycle_data = all_cycles_data[cycle_idx]
            
            # 1. 计算插值后的 Q(V) -> F1
            q_curr = self._get_interpolated_q(cycle_data)
            mat_f1[i, :] = q_curr
            
            # 2. 计算 dQ/dV -> F2 
            # 使用 numpy 的梯度函数计算微分
            dq_dv = cycle_data['dQdV'][:self.voltage_points]
            mat_f2[i, :] = dq_dv
            
            # 3. 计算 Q_k - Q_1 -> F3 
            # 论文公式: F3 = Q_k(V) - Q_1(V)
            delta_q = q_curr - q_ref
            mat_f3[i, :] = delta_q
            
        return mat_f1, mat_f2, mat_f3

    def _get_interpolated_q(self, data: Dict) -> np.ndarray:
        """
        将任意放电片段的 (V, Q) 插值到固定的 self.ref_voltage 坐标轴上
        """
        # 假设 df 包含 'V' (电压) 和 'Qd' (放电容量/容量) 列
        # 注意 MIT 数据集中，放电过程中电压是下降的，Q 是增加的
        v_raw = data['V']
        q_raw = data['Qd']
        
        # 去除重复电压值并排序 (np.interp 需要 x 轴单调递增)
        # 因为放电电压是下降的，我们需要先按照电压从小到大排序
        sort_idx = np.argsort(v_raw)
        v_sorted = v_raw[sort_idx]
        q_sorted = q_raw[sort_idx]
        
        # 去除重复的 x (电压) 点，防止插值错误
        v_unique, unique_indices = np.unique(v_sorted, return_index=True)
        q_unique = q_sorted[unique_indices]
        
        if len(v_unique) < 10: # 数据点太少，视为无效
            return None
            
        # 线性插值  (Voltage evenly spaced)
        # self.ref_voltage 是从 2.0 到 3.5
        # left=0, right=max_capacity 意味着如果在 2.0V 以下或 3.5V 以上，我们如何填充
        # 通常放电曲线在 3.5V 时容量接近 0，在 2.0V 时容量最大
        q_interp = np.interp(self.ref_voltage, v_unique, q_unique, left=q_unique.max(), right=0.0)
        
        return q_interp

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """还原预测值到真实循环次数"""
        if self.log_transform_target:
            return np.power(10, y)
        return y
    
    def get_feature_names(self):
        """返回特征名称（CNN使用图像特征）"""
        return ['CNN_3D_Features (100x100x3)']