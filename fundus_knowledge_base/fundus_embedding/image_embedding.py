import pathlib
import torch, json
from fundus_embedding.base_embedding import BaseEmbedding

class ImageEmbeddingData(BaseEmbedding):
    def __init__(self, image_path: pathlib.Path=None, embeding: torch.Tensor=None, diagnosis: str=None, discription: str=None):
        """init embedding data

        Args:
            image_path (pathlib.Path): image path
            embedding_path (pathlib.Path, optional): embedding saving path. Defaults to None.
            embeding (torch.Tensor, optional): torch.Tensor. Defaults to None.
            diagnosis (str, optional):.  Defaults to None.
            discription (str, optional): . Defaults to None.
        """
        super().__init__()
        self.image_path = image_path
        self.embedding = embeding
        self.diagnosis = diagnosis
        self.discription = discription
        
    def save(self, folder: pathlib.Path):
        """save the embedding data

        Args:
            folder (pathlib.Path): the path to save torch.Tensor and json
        """
        embedding_path = folder.joinpath(f"{self.image_path.stem}.pt")
        torch.save(self.embedding, embedding_path)
        info = {
            "image_path": str(self.image_path),
            "embedding_path": embedding_path,
            "diagnosis": self.diagnosis,
            "discription": self.discription
        }
        info_path = folder.joinpath(f"{self.image_path.stem}.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f)
        
    def load(self, folder: pathlib.Path, image_path: pathlib.Path):
        """load the embedding data of image

        Args:
            folder (pathlib.Path): _description_
            image_path (pathlib.Path): _description_
        """
        info_path = folder.joinpath(f"{self.image_path.stem}.json")
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        self.diagnosis = info["diagnosis"]
        self.discription = info["discription"]
        self.image_path = image_path
        self.embedding = torch.load(info["embeding_path"])