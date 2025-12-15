"""
Extended feature extraction module for battery cycle life prediction
This module extends StandardFeatureExtractor with additional features
"""
import numpy as np
from typing import Dict, Tuple, Any
from feature_extraction.standard import StandardFeatureExtractor
from config import Config
import re


class ExtendedFeatureExtractor(StandardFeatureExtractor):
    """Extended feature extractor that adds DeltaQc_var and ChargingPolicy to standard features"""
    
    def __init__(self, config: Config = None):
        """Initialize extended feature extractor
        
        Args:
            config: Configuration object containing feature extraction parameters
        """
        super().__init__(config)
        # Add new feature names to the list
        self.feature_names = self.feature_names + [
            'DeltaQc_var',   # 充电容量差异的方差 (Charge Capacity Difference Variance)
            'DeltaQc_min',    # 充电容量差异的最小值 (Charge Capacity Minimum)
            'ChargingPolicy'      # 充电策略 (Ordinal Encoded)
        ]
        
        # Create charging policy mapping
        self._policy_mapping = None
    
    def _get_c_rate_category(self, cell: Dict[str, Any]) -> float:
        """
        根据充电策略字符串提取 C-rate 并进行分类 (Ordinal Encoding)。
        
        逻辑依据论文:
        - 提取策略字符串中的最大 C-rate。
        - Group 1 (High): C-rate < 4.0 (即 1, 2, 3)
        - Group 2 (Very High): 4.0 <= C-rate < 7.0 (即 4, 5, 6)
        - Group 3 (Extremely High): C-rate >= 7.0 (即 7, 8)
        """
        try:
            policy_str = cell['charge_policy']
        except KeyError:
            print("Warning: 'charge_policy' not found in cell data.")
            return 0.0
     
        # 使用正则表达式提取所有 C-rate 数值
        # 匹配模式：数字 + 可选小数点 + 数字 + "C"
        # 例如 "3.6C(80%)-3.6C" -> 匹配 ["3.6", "3.6"]
        # 例如 "8C(15%)-3.6C" -> 匹配 ["8", "3.6"]
        matches = re.findall(r"(\d+\.?\d*)C", policy_str)
        
        if not matches:
            return 0.0
            
        # 转换为浮点数并取最大值
        c_rates = [float(x) for x in matches]
        max_c_rate = max(c_rates)
        
        # 3. 根据论文规则分类
        if max_c_rate < 4.0:
            return 1.0  # High (1, 2, 3)
        elif max_c_rate < 7.0:
            return 2.0  # Very High (4, 5, 6)
        else:
            return 3.0  # Extremely High (7, 8)
    
    def extract_features(self, battery_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract extended features from battery data
        
        Args:
            battery_data: Dictionary containing battery measurement data for multiple cells
            
        Returns:
            Tuple of (feature_array, cycle_life_array)
            - feature_array: shape (N, num_features) with 11 features
            - cycle_life_array: shape (N,)
        """
        N = len(battery_data)
        
        # First extract standard features
        X_standard, y = super().extract_features(battery_data)
         
        # Initialize arrays for new features
        DeltaQc_var = np.zeros(N)
        DeltaQc_min = np.zeros(N)
        ChargingPolicy = np.zeros(N)
        
        for i, key in enumerate(battery_data.keys()):
            cell = battery_data[key]
            
            # 1. DeltaQc_var: variance of charge capacity differences Qc_100 - Qc_10
            # 使用插值将原始Qc数据映射到1000个固定点
            cycle_100_data = cell['cycles'][str(self.config.FEATURE_CYCLE_100)]
            cycle_10_data = cell['cycles'][str(self.config.FEATURE_CYCLE_10)]
            
            # 获取原始Qc和V数据
            Vdlin = cell['Vdlin']  # 放电电压数据，用于插值
            V_100_raw = cycle_100_data['V']   # 原始电压
            Qc_100_raw = cycle_100_data['Qc'] # 原始充电容量
                
            # Cycle 10 原始数据
            V_10_raw = cycle_10_data['V']   # 原始电压
            Qc_10_raw = cycle_10_data['Qc'] # 原始充电容量
            
            # 插值到1000个固定点
            Qc_100 = self._interpolate_to_1000_points(V_100_raw, Qc_100_raw, Vdlin)
            Qc_10 = self._interpolate_to_1000_points(V_10_raw, Qc_10_raw, Vdlin)
            
            # 计算差异
            delta_Qc = Qc_100 - Qc_10
            valid_deltaQc = -delta_Qc[delta_Qc < 0] # 只考虑负值部分, 多cycle之后容量会减少
            valid_deltaQc = valid_deltaQc + self.config.EPSILON
            
            # Handle edge case where there are no valid values
            DeltaQc_var[i] = np.log(np.var(valid_deltaQc))
            DeltaQc_min[i] = np.log(np.min(valid_deltaQc))
            
            # 2. ChargingPolicy: ordinal encoding of charging policy
            if 'charge_policy' in cell:
                ChargingPolicy[i] = self._get_c_rate_category(cell)
            else:
                ChargingPolicy[i] = 0.0
        
        # Combine standard features with new features
        X_extended = np.column_stack([X_standard, DeltaQc_var, DeltaQc_min, ChargingPolicy])
        
        return X_extended, y
    
    def _interpolate_to_1000_points(self, V_raw: np.ndarray, Q_raw: np.ndarray, V_interp_grid: np.ndarray) -> np.ndarray:
        """
        将原始的电压-容量数据插值到固定的电压网格上（1000个点）。
        
        Args:
            V_raw: 原始电压数据（作为插值的X轴）。
            Q_raw: 原始容量数据（作为插值的Y轴）。
            V_interp_grid: 目标插值电压网格（新的X轴，即 Vdlin）。
            
        Returns:
            插值后的容量数组，shape (1000,)
        """
        # 1. 检查数据有效性
        if len(V_raw) < 2 or len(Q_raw) < 2:
            print("Warning: Insufficient data points for interpolation. Returning zeros.")
            return np.zeros_like(V_interp_grid) 
        
        # 2. 确保原始电压 V_raw 是单调递增的 (np.interp 的要求)
        # 充电数据理论上是递增的，但为稳健性，进行排序
        sort_indices = np.argsort(V_raw)
        V_sorted = V_raw[sort_indices]
        Q_sorted = Q_raw[sort_indices]

        # 3. 使用 np.interp 进行线性插值
        # left/right 参数确保 V_interp_grid 中超出 V_raw 范围的点使用边界值进行外推，
        # 维持容量曲线的平滑和连续性。
        Q_interp = np.interp(
            x=V_interp_grid, 
            xp=V_sorted, 
            fp=Q_sorted, 
            left=Q_sorted[0],  # 使用第一个点的容量值进行左外推
            right=Q_sorted[-1] # 使用最后一个点的容量值进行右外推
        )
        
        return Q_interp
       