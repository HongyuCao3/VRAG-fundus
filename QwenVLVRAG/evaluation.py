import pathlib
import sys
sys.path.append(str(pathlib.Path.cwd()))
import torch
from functools import partial
from QwenVLVRAG.sampler import InferenceSampler
from QwenVLVRAG.collate import collate_fn
from Datasets.MultiModalClassificationDataset import MultiModalClassificationDataset, MultiModalClassificationConfig
from QwenVLVRAG.QwenVL_vrag import QwenVLVRAG
    
class evaluation():
    def __init__(self):
        pass
    
    def evaluate_classification(self, checkpoint_path: str, batch_size:int=1, num_workers: int=1):
        vrag = QwenVLVRAG(checkpoint_path=checkpoint_path)
        config = MultiModalClassificationConfig()
        dataset = MultiModalClassificationDataset(config.DEFAULT_EXCEL_PATH, sheet_names=["CFP"])
        dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=vrag.tokenizer),
    )