"""
Data loader module for battery cycle life prediction
"""
import pickle
import numpy as np
from typing import Dict, Any, Tuple
from config import Config

# to remove
BATCH1_REMOVE_CELLS = ['b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22']
BATCH3_REMOVE_CELLS = ['b3c37', 'b3c2', 'b3c23', 'b3c32', 'b3c42', 'b3c43']

# Batch merging parameters
BATCH2_KEYS = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
BATCH1_KEYS = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
ADD_LEN = [662, 981, 1060, 208, 482]

class BatteryDataLoader:
    """Data loader for battery cycle life datasets"""
    
    def __init__(self, config: Config = None, apply_outlier_removal: bool = True):
        """Initialize data loader with configuration
        
        Args:
            config: Configuration object containing data paths and parameters
            apply_outlier_removal: Whether to apply 3-sigma outlier removal (default: True)
        """
        self.config = config if config else Config()
        self.apply_outlier_removal = apply_outlier_removal
        self.batch1 = None
        self.batch2 = None
        self.batch3 = None
        self.bat_dict = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.load_batch_data()
    
    def load_batch_data(self) -> Tuple[Dict, Dict, Dict]:
        """Load all three batch datasets
        
        Returns:
            Tuple of (batch1, batch2, batch3) dictionaries
        """
        print("Loading batch data...")
        
        # Load batch 1
        with open(self.config.BATCH1_PATH, 'rb') as f:
            self.batch1 = pickle.load(f)
        
        # Remove problematic cells from batch 1
        for cell in BATCH1_REMOVE_CELLS:
            if cell in self.batch1:
                del self.batch1[cell]
        
        # Load batch 2
        with open(self.config.BATCH2_PATH, 'rb') as f:
            self.batch2 = pickle.load(f)

        # There are four cells from batch1 that carried into batch2, we'll remove the data from batch2 
        # and put it with the correct cell from batch1
        self.batch1, self.batch2 = self._merge_batch_data()
        
        # Load batch 3
        with open(self.config.BATCH3_PATH, 'rb') as f:
            self.batch3 = pickle.load(f)
        
        # Remove noisy channels from batch 3
        for cell in BATCH3_REMOVE_CELLS:
            if cell in self.batch3:
                del self.batch3[cell]
        
        print(f"Loaded batch1: {len(self.batch1)} cells")
        print(f"Loaded batch2: {len(self.batch2)} cells") 
        print(f"Loaded batch3: {len(self.batch3)} cells")

        self.bat_dict = {**self.batch1, **self.batch2, **self.batch3}
        
        # self.bat_dict = self._remove_first_cycle_data(self.bat_dict)
        # Apply 3-sigma outlier removal if enabled
        if self.apply_outlier_removal:
            print("Applying 3-sigma outlier removal...")
            self.bat_dict = self._remove_outliers_from_dataset(self.bat_dict)
            print("Outlier removal completed.")
        
        return self.bat_dict
    
    def _merge_batch_data(self) -> Dict[str, Any]:
        """Merge batch1 and batch2 data, then combine with batch3
        
        Returns:
            Merged battery dictionary
        """
        # Merge batch2 data into batch1
        batch1 = self.batch1
        batch2 = self.batch2
        for key, bk in enumerate(BATCH1_KEYS):
            batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + ADD_LEN[key]
            for j in batch1[bk]['summary'].keys():
                if j == 'cycle':
                    batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[BATCH2_KEYS[key]]['summary'][j] + len(batch1[bk]['summary'][j])))
                else:
                    batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[BATCH2_KEYS[key]]['summary'][j]))
            last_cycle = len(batch1[bk]['cycles'].keys())
            for j, jk in enumerate(batch2[BATCH2_KEYS[key]]['cycles'].keys()):
                batch1[bk]['cycles'][str(last_cycle + j)] = batch2[BATCH2_KEYS[key]]['cycles'][jk]
                
        # Remove merged cells from batch2
        for cell in BATCH2_KEYS:
            if cell in batch2:
                del batch2[cell]
        
        return batch1, batch2

    # def _remove_first_cycle_data(self, bat_dict: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     移除所有电池的第一个循环数据（Cycle 1）。
    #     - 从详细的循环数据 'cycles' 中移除键 '1'。
    #     - 从 'summary' 数据中移除索引 1（对应 Cycle 1），并修复周期编号。
    #     """
    #     cleaned_dict = {}
    #     for key, cell in bat_dict.items():
    #         # 1. 从详细循环数据中移除
    #         if '1' in cell['cycles']:
    #             del cell['cycles']['1']
            
    #         # 2. 从汇总数据 'summary' 中移除索引 1
    #         if len(cell['summary']['cycle']) > 1:
    #             # 遍历所有 summary 键（'cycle', 'QD', 'IR' 等）
    #             for summary_key in cell['summary'].keys():
    #                 # 确保数组长度足够
    #                 if cell['summary'][summary_key].shape[0] > 1:
    #                     data = cell['summary'][summary_key]
    #                     # 保留索引 0（通常是初始状态）和从索引 2（Cycle 2）开始的数据
    #                     cell['summary'][summary_key] = np.concatenate((data[0:1], data[2:]))
                        
    #                     # 修复 'cycle' 键的编号，使其从 0, 1, 2... 连续
    #                     if summary_key == 'cycle':
    #                         cell['summary'][summary_key] = np.arange(len(cell['summary'][summary_key]))

    #         cleaned_dict[key] = cell
            
    #     return cleaned_dict
    
    def _remove_outliers_3sigma(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers using 3-sigma rule and replace with neighboring mean
        
        Args:
            data: 1D numpy array
            
        Returns:
            Array with outliers replaced by neighboring mean
        """
        if len(data) == 0:
            return data
            
        data = data.copy()
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return data
        
        # Find outliers (beyond 3 sigma)
        outlier_mask = np.abs(data - mean) > 3 * std
        outlier_indices = np.where(outlier_mask)[0]
        
        # Replace each outlier with mean of its neighbors
        for idx in outlier_indices:
            neighbors = []
            
            # Get left neighbor
            if idx > 0:
                neighbors.append(data[idx - 1])
            
            # Get right neighbor
            if idx < len(data) - 1:
                neighbors.append(data[idx + 1])
            
            # Replace with neighbor mean, or keep original if no neighbors
            if len(neighbors) > 0:
                data[idx] = np.mean(neighbors)
        
        return data
    
    def _remove_outliers_from_dataset(self, bat_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply 3-sigma outlier removal to all numerical data in the dataset
        
        Args:
            bat_dict: Battery dictionary
            
        Returns:
            Battery dictionary with outliers removed
        """
        outlier_count = 0
        
        for cell_id, cell_data in bat_dict.items():
            # Process summary data
            if 'summary' in cell_data:
                summary_keys = ['IR', 'QC', 'QD', 'Tavg', 'Tmin', 'Tmax', 'chargetime']
                for key in summary_keys:
                    if key in cell_data['summary']:
                        original = cell_data['summary'][key]
                        cleaned = self._remove_outliers_3sigma(original)
                        outliers = np.sum(np.abs(original - cleaned) > 1e-10)
                        outlier_count += outliers
                        cell_data['summary'][key] = cleaned
            
            # Process cycles data
            if 'cycles' in cell_data:
                cycle_keys = ['I', 'Qc', 'Qd', 'V', 'T', 't', 'dQdV', 'Qdlin', 'Tdlin']
                for cycle_id, cycle_data in cell_data['cycles'].items():
                    for key in cycle_keys:
                        if key in cycle_data:
                            original = cycle_data[key]
                            cleaned = self._remove_outliers_3sigma(original)
                            outliers = np.sum(np.abs(original - cleaned) > 1e-10)
                            outlier_count += outliers
                            cycle_data[key] = cleaned
            
            # Process Vdlin (voltage discharge linear interpolation)
            if 'Vdlin' in cell_data:
                vdlin = cell_data['Vdlin']
                if vdlin.ndim == 2:
                    # Process each cycle (column)
                    for i in range(vdlin.shape[1]):
                        original = vdlin[:, i]
                        cleaned = self._remove_outliers_3sigma(original)
                        outliers = np.sum(np.abs(original - cleaned) > 1e-10)
                        outlier_count += outliers
                        vdlin[:, i] = cleaned
                elif vdlin.ndim == 1:
                    original = vdlin
                    cleaned = self._remove_outliers_3sigma(original)
                    outliers = np.sum(np.abs(original - cleaned) > 1e-10)
                    outlier_count += outliers
                    cell_data['Vdlin'] = cleaned
        
        print(f"Total outliers detected and replaced: {outlier_count}")
        return bat_dict
    
    def split_data(self) -> Tuple[Dict, Dict, Dict]:
        """Split merged data into train, validation, and test sets
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        
        print("Splitting data into train/val/test sets...")
        
        if self.config.USE_RANDOM_SPLIT:
            # Random split based on specified sizes
            return self._random_split()
        else:
            # Original batch-based split
            return self._batch_based_split()
    
    def _batch_based_split(self) -> Tuple[Dict, Dict, Dict]:
        """Original batch-based split method
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        numBat1 = len(self.batch1)
        numBat2 = len(self.batch2)
        numBat3 = len(self.batch3)
        
        # Define indices for splitting
        val_ind = np.hstack((np.arange(0, (numBat1 + numBat2), 2), 83))
        train_ind = np.arange(1, (numBat1 + numBat2 - 1), 2)
        test_ind = np.arange(numBat1 + numBat2, numBat1 + numBat2 + numBat3)
        
        keys_array = list(self.bat_dict.keys())
        
        self.train_data = {keys_array[i]: self.bat_dict[keys_array[i]] for i in train_ind}
        self.val_data = {keys_array[i]: self.bat_dict[keys_array[i]] for i in val_ind}
        self.test_data = {keys_array[i]: self.bat_dict[keys_array[i]] for i in test_ind}
        
        print(f"Train set: {len(self.train_data)} cells (batch-based split)")
        print(f"Validation set: {len(self.val_data)} cells (batch-based split)")
        print(f"Test set: {len(self.test_data)} cells (batch-based split)")
        
        return self.train_data, self.val_data, self.test_data
    
    def _random_split(self) -> Tuple[Dict, Dict, Dict]:
        """Random split within each batch based on specified train/val/test proportions
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        # Get split proportions from config
        train_ratio = self.config.TRAIN_SPLIT
        val_ratio = self.config.VAL_SPLIT
        test_ratio = self.config.TEST_SPLIT if hasattr(self.config, 'TEST_SPLIT') else (1.0 - train_ratio - val_ratio)
        
        # Validate proportions
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Train ({train_ratio}) + Val ({val_ratio}) + Test ({test_ratio}) "
                        f"must sum to 1.0, but got {total_ratio}")
        
        # Set random seed
        np.random.seed(self.config.RANDOM_STATE)
        
        # Initialize data dictionaries
        train_data = {}
        val_data = {}
        test_data = {}
        
        # Process each batch separately to ensure representation from all batches
        batches = [
            ('batch1', self.batch1),
            ('batch2', self.batch2),
            ('batch3', self.batch3)
        ]
        
        for batch_name, batch_data in batches:
            keys_array = list(batch_data.keys())
            n_cells = len(keys_array)
            
            if n_cells == 0:
                continue
            
            # Calculate split sizes for this batch
            n_train = int(np.round(n_cells * train_ratio))
            n_val = int(np.round(n_cells * val_ratio))
            n_test = n_cells - n_train - n_val  # Remaining cells go to test
            
            # Ensure at least one cell in each split if possible
            if n_train == 0 and n_cells > 0:
                n_train = 1
                n_test = max(0, n_test - 1)
            if n_val == 0 and n_cells > 1:
                n_val = 1
                n_test = max(0, n_test - 1)
            
            # Shuffle and split indices
            shuffled_indices = np.random.permutation(n_cells)
            
            train_idx = shuffled_indices[:n_train]
            val_idx = shuffled_indices[n_train:n_train + n_val]
            test_idx = shuffled_indices[n_train + n_val:]
            
            # Add cells to respective datasets
            for idx in train_idx:
                key = keys_array[idx]
                train_data[key] = batch_data[key]
            
            for idx in val_idx:
                key = keys_array[idx]
                val_data[key] = batch_data[key]
            
            for idx in test_idx:
                key = keys_array[idx]
                test_data[key] = batch_data[key]
            
            print(f"{batch_name}: {n_train} train, {n_val} val, {n_test} test (total: {n_cells})")
        
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        
        print(f"\nTotal - Train: {len(self.train_data)} cells, "
            f"Val: {len(self.val_data)} cells, "
            f"Test: {len(self.test_data)} cells")
        print(f"Split ratios: {train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f}")
        print(f"Random seed: {self.config.RANDOM_STATE}")
        
        return self.train_data, self.val_data, self.test_data
    
    def get_all_data(self) -> Dict[str, Any]:
        """Get the complete merged dataset
        
        Returns:
            Complete merged battery dictionary
        """
        return self.bat_dict
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded datasets
        
        Returns:
            Dictionary containing dataset information
        """   
        # Get sample cell info
        sample_key = list(self.bat_dict.keys())[0]
        sample_cell = self.bat_dict[sample_key]
        
        info = {
            'total_cells': len(self.bat_dict),
            'sample_cell_keys': list(sample_cell.keys()),
            'sample_summary_keys': list(sample_cell['summary'].keys()),
            'sample_cycles_count': len(sample_cell['cycles'].keys()),
            'voltage_array_shape': sample_cell['Vdlin'].shape if 'Vdlin' in sample_cell else None
        }
        
        if hasattr(self, 'train_data') and self.train_data is not None:
            info.update({
                'train_cells': len(self.train_data),
                'val_cells': len(self.val_data),
                'test_cells': len(self.test_data)
            })
        
        return info