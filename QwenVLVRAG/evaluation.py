import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd()))
import tqdm
from QwenVLVRAG.collate import collate_fn
from Datasets.MultiModalClassificationDataset import (
    MultiModalClassificationDataset,
    MultiModalClassificationConfig,
)
from torch.utils.data import DataLoader
from QwenVLVRAG.QwenVL_vrag import QwenVLVRAG


class evaluation:
    def __init__(self):
        pass

    def evaluate_classification(
        self, checkpoint_path: str, batch_size: int = 1, sheet_names=["CFP"]
    ):
        vrag = QwenVLVRAG(checkpoint_path=checkpoint_path)
        config = MultiModalClassificationConfig()
        dataset = MultiModalClassificationDataset(
            config.DEFAULT_EXCEL_PATH, sheet_names=sheet_names
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Iterate through the dataloader
        idx = 0
        query = "what is th diagnosis?"
        for images, diagnosis in tqdm(dataloader):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": images,
                        },
                        {"type": "text", "text": query},
                    ],
                }
            ]
            answer = vrag.inference(messages=messages)
