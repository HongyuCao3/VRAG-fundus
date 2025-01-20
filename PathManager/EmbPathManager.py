import sys
sys.path.append("/home/hongyu/Visual-RAG-LLaVA-Med")
from pathlib import Path
from PathManager.BasePathManager import BasePathManager, BasePathConfig

class EmbPathConfig(BasePathConfig):
    def __init__(self):
        super().__init__()
        self.emb_saving_dir = Path.joinpath(self.root_path, "KnowledgeBase", "emb_savings")


class EmbPathManager(BasePathManager):
    def __init__(self):
        super().__init__()
        self.config = EmbPathConfig()
        
    def get_emb_dir(self, emb_name: str):
        # TODO:添加emb_name的细分
        emb_path = Path.joinpath(self.config.emb_saving_dir, emb_name)
        return emb_path
    
    def get_image_dir(self, image_name: str):
        img_dir = Path.joinpath(self.config.data_dir, image_name)
        return img_dir
        
if __name__ == "__main__":
    epm = EmbPathManager()
    print(epm.config.emb_saving_dir)