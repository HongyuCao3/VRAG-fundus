from pathlib import Path
from abc import ABC

class BasePathConfig(ABC):
    def __init__(self):
        self.root_path = Path("/home/hongyu/Visual-RAG-LLaVA-Med/")
        self.data_dir = Path.joinpath(self.root_path, "data")
    

class BasePathManager(ABC):
    def __init__(self):
        super().__init__()
        self.config = BasePathConfig()
        
    def get_data_path(self, data_name: str):
        data_path = Path.joinpath(self.config.data_dir, data_name)
        return data_path
    
if __name__ == "__main__":
    bpm = BasePathManager()
    print(bpm.get_data_path("Classic Images"))