from torch.utils.data import Dataset
import pandas as pd
import h5py
import numpy as np
import torch

class h5file_Dataset(Dataset):
    """
    A generic PyTorch Dataset class to load instance features and coordinates 
    from a single H5 file, guided by a CSV file.
    """
    def __init__(self, csv_path, h5_file_path, split, split_col='split'):
        """
        Args:
            csv_path (string): Path to the csv file with slide IDs, labels, and splits.
            h5_file_path (string): Path to the single H5 file containing all features.
            split (string): The desired data split to load ('train', 'val', or 'test').
            split_col (string): The name of the column in the CSV that contains the split information.
        """
        all_data = pd.read_csv(csv_path)
        
        # Use the specified split column to filter the data
        self.slide_data = all_data[all_data[split_col] == split].reset_index(drop=True)
        
        self.h5_file_path = h5_file_path

    def __len__(self):
        """
        Returns the total number of slides in the dataset split.
        """
        return len(self.slide_data)

    def __getitem__(self, idx):
        """
        Fetches the data for a single slide at the given index.
        """
        slide_id = self.slide_data.loc[idx, 'slide_id']
        label = self.slide_data.loc[idx, 'label']

        # Remove the file extension to match the H5 file key
        slide_id_key = slide_id.split('.')[0]

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            if slide_id_key not in h5_file:
                raise KeyError(f"Slide ID '{slide_id_key}' not found in H5 file.")
            
            features = np.array(h5_file[slide_id_key]['feat'])
            coords = np.array(h5_file[slide_id_key]['coords'])

        return torch.from_numpy(coords), torch.from_numpy(features), label