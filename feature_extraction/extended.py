import re
from feature_extraction.standard import StandardFeatureExtractor
from typing import Dict, Tuple, Any
from config import Config
import numpy as np
import pandas as pd

class ExtendedFeatureExtractor(StandardFeatureExtractor):
    """
    Extends the standard extractor with Charging Policy and detailed Thermal features.
    """
    
    def __init__(self, config: Config = None):
        super().__init__(config)
        # 增加新的特征
        self.feature_names.extend([
            # Charging Policy Features
            'policy_avg_crate', 'policy_step1_crate', 'policy_step2_crate', 'policy_switch_soc',
            # Enhanced Thermal Features
            'Tmax_mean', 'Tmin_mean', 'T_variance_mean'
        ])
        
    def _parse_policy(self, policy_str: str) -> dict:
        """
        Parses policy string like "6C(80%)-3.6C" or "3.6C(80%)-3.6C"
        Returns dictionary with numeric features.
        """
        feats = {
            'policy_step1_crate': 0.0,
            'policy_switch_soc': 0.0,
            'policy_step2_crate': 0.0,
            'policy_avg_crate': 0.0
        }
        
        # 常见格式正则匹配: "6C(80%)-3.6C"
        # 解释: Group 1 (Step 1 C-rate), Group 2 (Switch SOC), Group 3 (Step 2 C-rate)
        match = re.search(r'([\d\.]+)C\((\d+)%\)-([\d\.]+)C', policy_str)
        
        if match:
            c1 = float(match.group(1))
            soc = float(match.group(2)) / 100.0
            c2 = float(match.group(3))
            
            feats['policy_step1_crate'] = c1
            feats['policy_switch_soc'] = soc
            feats['policy_step2_crate'] = c2
            # 估算平均倍率 (简化计算: C1 * SOC + C2 * (1-SOC))
            feats['policy_avg_crate'] = c1 * soc + c2 * (1.0 - soc)
        else:
            # 处理单步策略或其他格式 (如 "3.6C-3.6C" 或 "3.6C")
            # 简单回退策略：提取字符串中所有数字取最大值作为 C-rate
            nums = [float(x) for x in re.findall(r'([\d\.]+)', policy_str)]
            if nums:
                feats['policy_avg_crate'] = max(nums)
                feats['policy_step1_crate'] = max(nums)
                
        return feats

    def extract_features(self, battery_data: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        # 1. 先获取基础特征
        X_base, y = super().extract_features(battery_data)
        
        # 2. 准备新特征数组
        N = len(battery_data)
        policy_feats_list = []
        tmax_mean = np.zeros(N)
        tmin_mean = np.zeros(N)
        t_var_mean = np.zeros(N)
        
        for i, key in enumerate(battery_data.keys()):
            cell = battery_data[key]
            
            # --- Policy Features ---
            policy_str = str(cell.get('charge_policy', ''))
            policy_feats_list.append(self._parse_policy(policy_str))
            
            # --- Enhanced Thermal Features ---
            # 计算前100圈 Tmax, Tmin 的统计值 (比 Integral 更敏感)
            # summary['Tmax'] 是一个数组，包含每一圈的最高温
            cycle_end = min(len(cell['summary']['Tmax']), self.config.FEATURE_CYCLE_END)
            # 忽略第0、1圈可能的噪音
            valid_idx = slice(self.config.FEATURE_CYCLE_START, cycle_end)
            
            tmax_mean[i] = np.mean(cell['summary']['Tmax'][valid_idx])
            tmin_mean[i] = np.mean(cell['summary']['Tmin'][valid_idx])
            
            # 尝试计算每一圈内的温度方差并取平均（如果 cycle data 可用）
            # 这里简化处理：直接用 Tmax - Tmin 作为简易的温差指标
            t_var_mean[i] = np.mean(cell['summary']['Tmax'][valid_idx] - cell['summary']['Tmin'][valid_idx])

        # 3. 合并特征
        X_policy = pd.DataFrame(policy_feats_list)
        X_thermal = pd.DataFrame({
            'Tmax_mean': tmax_mean,
            'Tmin_mean': tmin_mean,
            'T_variance_mean': t_var_mean
        })
        
        X_final = pd.concat([X_base, X_policy, X_thermal], axis=1)
        
        # 更新 feature_names 以匹配最终 DataFrame
        self.feature_names = X_final.columns.tolist()
        
        return X_final, y