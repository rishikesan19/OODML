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
    def __init__(self, csv_path, h5_file_path, split):
        """
        Args:
            csv_path (string): Path to the csv file with slide IDs, labels, and splits.
            h5_file_path (string): Path to the single H5 file containing all features.
            split (string): The desired data split to load ('train', 'val', or 'test').
        """
        # Read the CSV and filter for the requested split (e.g., 'train')
        all_data = pd.read_csv(csv_path)
        self.slide_data = all_data[all_data['split'] == split].reset_index(drop=True)
        
        # Store the H5 file path. The file will be opened in __getitem__
        # which is more efficient for multiprocessing with PyTorch's DataLoader.
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
        # Get the slide ID and label for the given index
        slide_id = self.slide_data.loc[idx, 'slide_id']
        label = self.slide_data.loc[idx, 'label']

        # IMPORTANT: Remove the '.tif' extension to match the H5 file key
        slide_id_key = slide_id.replace('.tif', '')

        # Open the H5 file and get the data for the specific slide
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            # Ensure the key exists before trying to access it
            if slide_id_key not in h5_file:
                raise KeyError(f"Slide ID '{slide_id_key}' not found in H5 file.")
            
            features = np.array(h5_file[slide_id_key]['feat'])
            coords = np.array(h5_file[slide_id_key]['coords'])

        # The OODML model expects the entire bag, so we return all features
        return torch.from_numpy(coords), torch.from_numpy(features), label
