import pathlib
from typing import Tuple
from torch.types import (
    Number,
)
import json
from torch.nn.functional import cosine_similarity
from KnowledgeBase.Retriever.BaseRetriever import BaseRetriever
from KnowledgeBase.EmbBuilder.MultiDiseaseEmbBuilder import MultiDiseaseEmbBuilder


class ImageRetriever(BaseRetriever):
    def __init__(self, emb_folder: pathlib.Path):
        super().__init__()
        self.emb_builder = MultiDiseaseEmbBuilder()
        self.representations = self.emb_builder.load_image_representations(emb_folder)

    def get_similar_images(
        self,
        input_img: pathlib.Path,
        img_folder: pathlib.Path,
        k=2,
        layer=11,
    ) -> list[Tuple[pathlib.Path, Number]]:
        """get the top k simlar images' path and similarities"""
        # 获取输入图片的嵌入
        input_emb = self.emb_builder.get_image_embedding(input_img, layer_index=layer)

        # 确保输入嵌入是一个二维张量 (batch_size, feature_dim)
        if len(input_emb.shape) == 4:
            input_emb = input_emb.mean(dim=(2, 3))  # 全局平均池化
        elif len(input_emb.shape) == 3:
            input_emb = input_emb.mean(dim=2)  # 全局平均池化
        else:
            raise ValueError("Unexpected feature tensor shape")

        # 加载文件夹中的所有嵌入

        # 计算所有嵌入与输入嵌入的相似度
        similarities = []
        for img_name, rep in self.representations.items():
            # 确保预加载的嵌入也是一个二维张量 (batch_size, feature_dim)
            if len(rep.shape) == 4:
                rep = rep.mean(dim=(2, 3))  # 全局平均池化
            elif len(rep.shape) == 3:
                rep = rep.mean(dim=2)  # 全局平均池化
            else:
                raise ValueError("Unexpected feature tensor shape")

            # 计算余弦相似度
            sim = cosine_similarity(input_emb, rep, dim=1)
            similarities.append((img_name, sim.item()))

        # 按相似度排序并选择前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]

        # 获取最相似图像的原始路径
        similar_images = [
            (pathlib.Path.joinpath(img_folder, img_name), sim)
            for img_name, sim in top_k
        ]
        return similar_images

    def retrieve(
        self, input_img: pathlib.Path, img_folder: pathlib.Path, k: int=2, layer: int=11
    ):
        score = []
        txt = []
        metadata = []
        img = []
        similar_images = self.get_similar_images(
            input_img=input_img, img_folder=img_folder, k=k, layer=layer
        )
        for img_name, sim in similar_images:
            score.append(sim)
            txt.append(img_name.stem)
            metadata.append(img_name)
            img.append(img_name)
        return {"score": score, "txt": txt, "metadata": metadata, "img": img}
