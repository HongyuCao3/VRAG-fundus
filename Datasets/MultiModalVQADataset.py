import pandas as pd
import sys
sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")
from torch.utils.data import Dataset, DataLoader
import torch
import pathlib
from typing import Tuple
from PathManager.DatasetPathManager import DatasetPathManager
from torchvision.io import read_image

class MultiModalVQAConfig:
    def __init__(self):
        self.path_manager = DatasetPathManager()
        self.dataset_name = "Multimodal VQA Dataset"
        self.dataset_dir = self.path_manager.get_dataset_dir(
            dataset_name=self.dataset_name
        )
        self.DEFAULT_EXCEL_PATH = pathlib.Path.joinpath(
            self.dataset_dir, "Multimodal VQA dataset_1015.xlsx"
        )

class MultiModalVQADataset(Dataset):
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
        self.transform = transform
        # samples_dict = OrderedDict()
        samples_dict = []
        self.config = MultiModalVQAConfig()
        
        for sheet_name, df in self.annotations_df.items():
            # select according to sheet_name
            if sheet_name not in sheet_names:
                print("ignore " + sheet_name + " modal")
                continue
            for index, row in df.iterrows():
                # img_path = os.path.join(
                #     self.root_dir,
                #     sheet_name,
                #     row["Diagnosis"],
                #     f"{row['Case number']}.jpg",
                # )
                img_path = pathlib.Path.joinpath(
                    self.config.dataset_dir,
                    sheet_name,
                    row["Diagnosis"],
                    f"{row['Case number']}.jpg",
                )
                # img_path2 = os.path.join(
                #     self.root_dir,
                #     sheet_name,
                #     row["Diagnosis"],
                #     f"{row['Case number']}.png",
                # )
                img_path2 = pathlib.Path.joinpath(
                    self.config.dataset_dir,
                    sheet_name,
                    row["Diagnosis"],
                    f"{row['Case number']}.png",
                )
                if img_path.exists():
                    samples_dict.append(
                        {
                            "img_path": img_path,
                            "diagnosis": row["Diagnosis"],
                            "Q": row["Q"],
                            "A": row["A"],
                        }
                    )
                elif img_path2.exists():
                    samples_dict.append(
                        {
                            "img_path": img_path2,
                            "diagnosis": row["Diagnosis"],
                            "Q": row["Q"],
                            "A": row["A"],
                        }
                    )
        # Convert the dictionary back into a list of dictionaries
        # self.samples = list(samples_dict.values())
        self.samples = samples_dict

    def __getitem__(self, idx) -> Tuple[str, str, str, str]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]
        img_path = str(sample["img_path"])
        diagnosis = sample["diagnosis"]
        query = sample["Q"]
        answer = sample["A"]
        # Load image
        image = read_image(img_path)

        if self.transform:
            image = self.transform(image)

        return img_path, diagnosis, query, answer

    def __len__(self):
        return len(self.samples)
    
if __name__ == "__main__":
    config = MultiModalVQAConfig()
    mdvd = MultiModalVQADataset(config.DEFAULT_EXCEL_PATH, sheet_names=["CFP"])
    dataloader = DataLoader(dataset=mdvd, batch_size=1, shuffle=False)
    for img_path, diagnosis, query, answer in dataloader:
        print(f"img path: {img_path}")
        print(f"diagnosis: {diagnosis}")
