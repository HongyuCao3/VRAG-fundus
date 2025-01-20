import sys
sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")
from pathlib import Path
from PathManager.BasePathManager import BasePathManager, BasePathConfig

class DatasetPathConfig(BasePathConfig):
    def __init__(self):
        super().__init__()
        self.dataset_saving_dir = Path.joinpath(self.root_path, "data")
        
class DatasetPathManager(BasePathManager):
    def __init__(self):
        super().__init__()
        self.config = DatasetPathConfig()
        
    def get_dataset_dir(self, dataset_name: str):
        dataset_dir = Path.joinpath(self.config.dataset_saving_dir, dataset_name)
        return dataset_dir