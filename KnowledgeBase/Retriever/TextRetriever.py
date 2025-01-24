import pathlib
import sys
sys.path.append(r"/home/hongyu/Visual-RAG-LLaVA-Med/")
import json
from torch.types import (
    Number,
)
from torch.nn.functional import cosine_similarity
from KnowledgeBase.Retriever.BaseRetriever import BaseRetriever
from KnowledgeBase.EmbBuilder.MultiDiseaseEmbBuilder import MultiDiseaseEmbBuilder
from PathManager.EmbPathManager import EmbPathManager, EmbPathConfig

class TextRetriever(BaseRetriever):
    def __init__(self,  emb_folder: pathlib.Path):
        super().__init__()
        self.emb_builder = MultiDiseaseEmbBuilder()
        self.representations = self.emb_builder.load_text_embeddings(emb_folder)
        
    def get_similar_texts(
        self,
        input_img: pathlib.Path,
        k=2,
    ) -> list[tuple[str, Number]]:
        input_emb = self.emb_builder.encode_image(input_img)
        similarities = []
        for txt, rep in self.representations.items():
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
    
if __name__ == "__main__":
    pm = EmbPathManager()
    emb_folder = pm.get_emb_dir(pm.config.default_text_emb_name)
    print(emb_folder)
    tr = TextRetriever(emb_folder=emb_folder)
    input_img = pm.config.test_img_path
    similar_txts = tr.get_similar_texts(input_img=input_img)
    print(similar_txts)