import pathlib
import json
from torch.types import (
    Number,
)
from torch.nn.functional import cosine_similarity
from KnowledgeBase.Retriever.BaseRetriever import BaseRetriever
from KnowledgeBase.EmbBuilder.MultiDiseaseEmbBuilder import MultiDiseaseEmbBuilder

class TextRetriever(BaseRetriever):
    def __init__(self,  emb_folder: pathlib.Path):
        super().__init__()
        self.emb_builder = MultiDiseaseEmbBuilder()
        self.representations = self.emb_builder.load_text_embeddings(emb_folder)
        
    def find_similar_text(
        self,
        input_img: pathlib.Path,
        k=2,
        layer=11,
    ) -> list[tuple[str, Number]]:
        input_emb = self.emb_builder.get_image_embedding(input_img, layer_index=layer)
        similarities = []
        for txt, rep in self.representations.items():
            if len(rep.shape) == 4:
                rep = rep.mean(dim=(2, 3))  # 全局平均池化
            elif len(rep.shape) == 3:
                rep = rep.mean(dim=2)  # 全局平均池化
            else:
                raise ValueError("Unexpected feature tensor shape")
            
            sim = cosine_similarity(input_emb, rep, dim=1)
            similarities.append((txt, sim.item()))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # 获取最相似图像的原始路径
        similar_txts = [
            (txt, sim)
            for txt, sim in top_k
        ]
        return similar_txts