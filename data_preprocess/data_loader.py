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
    
    def __init__(self, config: Config = None):
        """Initialize data loader with configuration
        
        Args:
            config: Configuration object containing data paths and parameters
        """
        self.config = config if config else Config()
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
    
    def split_data(self) -> Tuple[Dict, Dict, Dict]:
        """Split merged data into train, validation, and test sets
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        
        print("Splitting data into train/val/test sets...")
        
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
        
        print(f"Train set: {len(self.train_data)} cells")
        print(f"Validation set: {len(self.val_data)} cells")
        print(f"Test set: {len(self.test_data)} cells")
        
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