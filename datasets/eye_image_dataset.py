import pandas as pd
import os, json, shutil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from collections import Counter, OrderedDict
from torch.utils.data import Dataset
from torchvision.io import read_image
class EyeImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0] + '.jpg')
        # image = read_image(img_path)
        diagnosis = self.annotations.iloc[idx, 1]
        finding = self.annotations.iloc[idx, 1]
        modality = self.annotations.iloc[idx, 3]
        dataset = self.annotations.iloc[idx, 4]
        caption = self.annotations.iloc[idx, 5]

        # if self.transform:
        #     image = self.transform(image)

        return img_path, diagnosis,
    # finding, modality, dataset, caption
    
    def get_entries_by_diagnosis(self, diagnosis, n=None):
        """
        Args:
            diagnosis (string): The diagnosis to filter the dataset entries by.
            n (int, optional): Maximum number of entries to return. If None, all matching entries are returned.

        Returns:
            list: A list of dictionaries, each representing a unique sample with the specified diagnosis and img_name.
        """
        # Filter annotations DataFrame by the given diagnosis using the correct column name
        filtered_df = self.annotations[self.annotations.iloc[:, 1] == diagnosis]

        # Add 'img_name' column to the filtered DataFrame
        filtered_df['img_name'] = filtered_df.iloc[:, 0].astype(str) + '.jpg'

        # Remove duplicates based on 'img_name' and 
        filtered_df.drop_duplicates(subset=['img_name'], inplace=True)

        # Apply limit if n is specified and valid
        if n is not None and isinstance(n, int) and n > 0:
            filtered_df = filtered_df.head(n)

        # Create a list of dictionaries containing 'img_name' 
        entries = filtered_df[['img_name']].to_dict(orient='records')

        return entries
