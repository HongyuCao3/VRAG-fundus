import pandas as pd
import pathlib
import torch
import sys
sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")
from collections import Counter, OrderedDict
from typing import Tuple, Any
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from PathManager.DatasetPathManager import DatasetPathManager


class MultiModalClassificationConfig:
    def __init__(self):
        self.path_manager = DatasetPathManager()
        self.dataset_name = "Multimodal VQA Dataset"
        self.dataset_dir = self.path_manager.get_dataset_dir(
            dataset_name=self.dataset_name
        )
        self.DEFAULT_EXCEL_PATH = pathlib.Path.joinpath(
            self.dataset_dir, "Multimodal VQA dataset_1015.xlsx"
        )


class MultiModalClassificationDataset(Dataset):
    def __init__(
        self,
        excel_file: pathlib.Path,
        transform=None,
        sheet_names=["CFP", "FFA", "ultrasound", "OCT", "slitlamp"],
    ):
        """
        Args:
            excel_file (string): Path to the Excel file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations_df = pd.read_excel(
            excel_file, sheet_name=None
        )  # Read all sheets
        self.transform = transform
        self.annotations_df = pd.read_excel(
            excel_file, sheet_name=None
        )  # Read all sheets
        # self.root_dir = root_dir
        self.transform = transform
        self.config = MultiModalClassificationConfig()

        # Use an OrderedDict to preserve insertion order and remove duplicates based on img_path
        samples_dict = OrderedDict()

        for sheet_name, df in self.annotations_df.items():
            # select according to sheet_name
            if sheet_name not in sheet_names:
                print("ignore " + sheet_name + " modal")
                continue
            for index, row in df.iterrows():
                # img_path = os.path.join(self.root_dir, sheet_name, row['Diagnosis'], f"{row['Case number']}.jpg")
                img_path = pathlib.Path.joinpath(
                    self.config.dataset_dir,
                    sheet_name,
                    row["Diagnosis"],
                    f"{row['Case number']}.jpg",
                )
                # img_path2 = os.path.join(self.root_dir, sheet_name, row['Diagnosis'], f"{row['Case number']}.png")
                img_path2 = pathlib.Path.joinpath(
                    self.config.dataset_dir,
                    sheet_name,
                    row["Diagnosis"],
                    f"{row['Case number']}.png",
                )
                if img_path.exists() and img_path not in samples_dict:
                    samples_dict[img_path] = {
                        "img_path": img_path,
                        "diagnosis": row["Diagnosis"],
                    }
                elif img_path2.exists() and img_path2 not in samples_dict:
                    samples_dict[img_path] = {
                        "img_path": img_path2,
                        "diagnosis": row["Diagnosis"],
                    }
        # Convert the dictionary back into a list of dictionaries
        self.samples = list(samples_dict.values())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[str, str]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]
        img_path = sample["img_path"]
        diagnosis = sample["diagnosis"]

        # Load image
        image = read_image(str(img_path))

        if self.transform:
            image = self.transform(image)

        return str(img_path), diagnosis

    def get_entries_by_diagnosis(self, diagnosis):
        """
        Args:
            diagnosis (string): The diagnosis to filter the dataset entries by.

        Returns:
            list: A list of dictionaries, each representing a sample with the specified diagnosis.
        """
        # Filter samples by the given diagnosis and return them as a list
        filtered_samples = [
            sample for sample in self.samples if sample["diagnosis"] == diagnosis
        ]

        return filtered_samples
    
if __name__ == "__main__":
    config = MultiModalClassificationConfig()
    mdcd = MultiModalClassificationDataset(config.DEFAULT_EXCEL_PATH, sheet_names=["CFP"])
    dataloader = DataLoader(dataset=mdcd, batch_size=1, shuffle=False)
    for image_path, diagnosis in dataloader:
        print(f"image path: {image_path}")
        print(f"diagnosis: {diagnosis}")
