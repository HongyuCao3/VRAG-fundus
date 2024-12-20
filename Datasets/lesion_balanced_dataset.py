import pandas as pd
import os, json, shutil, sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
from collections import Counter, OrderedDict
from torch.utils.data import Dataset
from torchvision.io import read_image

class LesionBalancedDataset(Dataset):
    def __init__(self, excel_file, root_dir, transform=None, sheet_names = ["CFP", "FFA", "ultrasound", "OCT", "slitlamp"]):
        """
        Args:
            excel_file (string): Path to the Excel file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations_df = pd.read_excel(excel_file, sheet_name=None)  # Read all sheets
        self.root_dir = root_dir
        self.transform = transform
        self.annotations_df = pd.read_excel(excel_file, sheet_name=None)  # Read all sheets
        self.root_dir = root_dir
        self.transform = transform
        samples_dict = OrderedDict()
        
        for sheet_name, df in self.annotations_df.items():
            # select according to sheet_name
            if sheet_name not in sheet_names:
                print("ignore " + sheet_name + " modal")
                continue
            for index, row in df.iterrows():
                img_path = row["impath"].replace("/home/danli/data/public/alldataset", "/home/hongyu/alldataset/images")
                img_path2 = row["impath"].replace("/home/danli/data/public/alldataset", "/home/hongyu/alldataset/images")
                # img_path = os.path.join(self.root_dir, sheet_name, row['Diagnosis'], f"{row['Case number']}.jpg")
                # img_path2 = os.path.join(self.root_dir, sheet_name, row['Diagnosis'], f"{row['Case number']}.png")
                if os.path.exists(img_path):
                    samples_dict[img_path] = {
                        'img_path': img_path,
                        'Q': row["Q"],
                        "A": row["A"]
                    }
                elif os.path.exists(img_path2) :
                    samples_dict[img_path] = {
                        'img_path': img_path2,
                        'Q': row["Q"],
                        "A": row["A"]
                    }
                else:
                    print("Image Not Found: ")
                    print(img_path)
        # Convert the dictionary back into a list of dictionaries
        self.samples = list(samples_dict.values())
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]
        img_path = sample['img_path']
        query = sample['Q']
        answer = sample['A']
        # Load image
        image = read_image(img_path)
        
        if self.transform:
            image = self.transform(image)

        return img_path, query, answer