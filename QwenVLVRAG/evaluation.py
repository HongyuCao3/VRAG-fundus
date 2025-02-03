import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd()))
import tqdm
import json
from Datasets.MultiModalClassificationDataset import (
    MultiModalClassificationDataset,
    MultiModalClassificationConfig,
)
from torch.utils.data import DataLoader
from QwenVLVRAG.QwenVL_vrag import QwenVLVRAG


class evaluation_config:
    root_dir = pathlib.Path("./QwenVRAG/output/")


class evaluation:
    def __init__(self):
        self.config = evaluation_config()

    def evaluate_classification(
        self,
        checkpoint_path: str,
        image_index_folder: pathlib.Path,
        text_emb_folder: pathlib.Path,
        batch_size: int = 1,
        sheet_names=["CFP"],
        use_pics: bool = False,
    ):
        vrag = QwenVLVRAG(checkpoint_path=checkpoint_path)
        dataset_config = MultiModalClassificationConfig()
        dataset = MultiModalClassificationDataset(
            dataset_config.DEFAULT_EXCEL_PATH, sheet_names=sheet_names
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Iterate through the dataloader
        results = []
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
            answer = vrag.inference_rag(
                messages=messages,
                image_index_folder=image_index_folder,
                text_emb_folder=text_emb_folder,
                use_pics=use_pics,
            )
            results.append(
                {
                    "img_name": images,
                    "ground truth": diagnosis,
                    "llm respond": answer,
                }
            )
        # save_result
        result_saving_folder = self.config.root_dir.joinpath(
            dataset_config.dataset_name
        )
        if not result_saving_folder.exists():
            result_saving_folder.mkdir()
        result_saving_path = result_saving_folder.joinpath(
            f"{image_index_folder.name}_{text_emb_folder.name}_usepics_{str(use_pics)}.json"
        )
        with result_saving_path.open("w", encoding="utf-8") as f:
            json.dump(results, f)


if __name__ == "__main__":
    eva = evaluation()
