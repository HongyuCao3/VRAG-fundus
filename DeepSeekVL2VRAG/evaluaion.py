import pathlib
import sys

sys.path.append(str(pathlib.Path.cwd()))
from tqdm import tqdm
import json
from Datasets.MultiModalClassificationDataset import (
    MultiModalClassificationDataset,
    MultiModalClassificationConfig,
)
from Datasets.MultiModalVQADataset import MultiModalVQAConfig, MultiModalVQADataset
from torch.utils.data import DataLoader
from PathManager.EmbPathManager import EmbPathManager, EmbPathConfig
from DeepSeekVL2VRAG.deepseekvl2_vrag import DeepSeekVL2VRAG

class evaluation_config:
    root_dir = pathlib.Path("/home/hongyu/Visual-RAG-LLaVA-Med/QwenVLVRAG/output/")
    
    
class evaluation:
    def __init__(self):
        self.config = evaluation_config()
