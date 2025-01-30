import torch, json
import pathlib
from fundus_embedding.base_embedding import BaseEmbedding

class TextEmbeddingData(BaseEmbedding):
    def __init__(self, text: str, diagnosis: str, embedding: torch.Tensor=None):
        super().__init__()
        self.text = text
        self.diagnosis = diagnosis
        self.embedding = embedding
        
    def save(self, folder: pathlib.Path):
        # save embedding
        embedding_path = folder.joinpath(f"{self.diagnosis}.pt")
        torch.save(self.embedding, embedding_path)
        
        # save json of information
        info = {
            "diagnosis": self.diagnosis,
            "text": self.text,
            "embedding_path": embedding_path,
        }
        info_path = folder.joinpath(f"{self.diagnosis}.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f)
            
    def load(self, folder: pathlib.Path):
        info_path = folder.joinpath(f"{self.diagnosis}.json")
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
            
        self.diagnosis = info["diagnosis"]
        self.embedding = torch.load(info["embeding_path"])
        self.text = info["text"]