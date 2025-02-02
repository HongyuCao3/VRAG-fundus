import sys
sys.path.append(r"/home/hongyu/Visual-RAG-LLaVA-Med/")
from fundus_knowledge_base.emb_builder.BaseEmbBuilder import BaseEmbBuilder
from abc import ABC

class BaseRetriever(ABC):
    def __init__(self):
        super().__init__()
        
        
    def retrieve(self, img_path, emb):
        ret_empty = {"img": [], "txt": [], "score": [], "metadata": []}
        if self.crop_emb:
            ret_c = self.crop_emb.get_detailed_similarities_crop(img_path, self.top_k_c)
        else:
            ret_c = ret_empty
        if self.level_emb:
            ret_l = self.level_emb.get_detailed_similarities(img_path, self.top_k_l, self.layer)
        else:
            ret_l = ret_empty
        return ret_c, ret_l