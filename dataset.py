import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class DRDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = f"{self.image_dir}/{self.data_frame.iloc[idx, 0]}.jpg"  # Adjust file extension if needed
        # image = Image.open(img_name)
        diagnosis = self.data_frame.iloc[idx, 1]
        
        # if self.transform:
        #     image = self.transform(image)

        return img_name, diagnosis

# Usage Example
if __name__ == "__main__":
    # Specify the path to your CSV and image directory
    csv_file = './data/DR/multidr.csv'
    image_dir = './data/DR/multidr'

    # Define transformations if needed
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to desired size
        transforms.ToTensor(),            # Convert image to tensor
    ])

    # Create dataset and dataloader
    dataset = DRDataset(csv_file=csv_file, image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Iterate through the dataloader
    for images, diagnoses in dataloader:
        print(images, diagnoses)