import argparse
import torch
import os
import json
from tqdm import tqdm

from PIL import Image
import math
from transformers import set_seed, logging

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
from llama_index.core import (ServiceContext, 
                               SimpleDirectoryReader,
                               SimpleDirectoryReader,
                               StorageContext,
                               load_index_from_storage,
                               Settings)
from llama_index.core.schema import ImageNode
from llama_index.core.schema import ImageDocument
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class VRAG():
    def __init__(self, args):
        self.model_path = args.model_path
        self.top_k = args.top_k
        self.use_pics = args.use_pics
        self.use_rag = args.use_rag
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, args.model_base, model_name
        )
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.image_folder = args.image_folder
        self.qa_tmpl_str = (
            "Given the provided information, including retrieved contents and metadata, \
            accurately and precisely answer the query without any additional prior knowledge.\n"
            "Please ensure honesty and responsibility, refraining from any racist or sexist remarks.\n"
            "---------------------\n"
            "Context: {context_str}\n"     ## 将上下文信息放进去
            "Metadata: {metadata_str} \n"  ## 将原始的meta信息放进去
            "---------------------\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        if os.path.exists("./data/emb_crop"):
            storage_context = StorageContext.from_defaults(persist_dir="./data/emb_crop")
            self.multi_index = load_index_from_storage(storage_context)
        else:
            self.build_index()
        
    def build_index(self, json_folder="./data/lesion/"):
        # read meta data
        document = []
        # with open(args.meta_data, "r", encoding="UTF-8") as f_m:
        #     meta_data = json.load(f_m)
        document = self.extract_image_data(json_folder)
        # for k, d in meta_data.items():
        #     caption = d["discription"]
        #     img_path = self.image_folder + d["file"]
        #     seg_path = self.image_folder + d["seg"]
        #     document.append([img_path, caption, d])
        # use llama-index to construct index
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        image_nodes = [ImageNode(image_path=json_folder+p, text=t, meta_data=k) for p, t, k in document]
        self.multi_index = MultiModalVectorStoreIndex(image_nodes, show_progress=True)
        # save index
        self.multi_index.storage_context.persist(persist_dir="./data/emb_crop")
        
    def extract_image_data(self, json_folder):
        image_data = []

        # 遍历指定目录下的所有JSON文件
        for filename in os.listdir(json_folder):
            if filename.endswith('.json'):
                file_path = os.path.join(json_folder, filename)
                with open(file_path, 'r') as f:
                    # 加载JSON内容
                    data = json.load(f)
                    
                    # 提取信息并构成元组
                    for text, image_path in data.items():
                        meta_data = {'source_file': filename}  # 可以根据需要添加更多元数据
                        image_data.append((image_path, text, meta_data))

        return image_data
        
    def inference_rag(self, query_str, img_path):
        record_data = {}
        # do retrieval
        retrieve_data = self.multi_index.as_retriever(similarity_top_k=self.top_k, image_similarity_top_k=self.top_k)
        # TODO: 需要了解两个top的参数最佳设置
        txt = []
        score = [] 
        img = [] 
        metadata= []
        # multi modal retrieve
        # img, txt, score, metadata = retrieve_data.retrieve(query_str)
        # image retrieve
        # print(retrieve_data.image_to_image_retrieve(img_path))
        nodes = retrieve_data.image_to_image_retrieve(img_path)
        for node in nodes:
            print(type(node))
            txt.append(node.get_text()) # excudates
            score.append(node.get_score()) # 0.628
            img.append(node.node.image_path)
            metadata.append(node.node.metadata)
            
        record_data.update({"txt":txt})
        record_data.update({"score":score})
        record_data.update({"img":img})
        record_data.update({"org":img_path})
            # img, txt, score, metadata = node
        # txt2img retrieve
        # img, txt, score, metadata = retrieve_data.text_to_image_retrieve(img_path)
        print(score)
        image_documents = [ImageDocument(iamge_path=img_path)]
        image_org = Image.open(img_path)
        images= [image_org]
        if self.use_pics:
            for res_img in img:
                image_documents.append(ImageDocument(image_path=res_img))
                # print(res_img)
                image = Image.open(res_img)
                images.append(image)
        context_str = ",".join(txt)
        metadata_str = metadata
            
        # do inference
        set_seed(0)
        disable_torch_init()
        print(self.use_rag)
        if self.use_rag:
            prompt = self.qa_tmpl_str.format(
                context_str=context_str,
                metadata_str=metadata_str,
                query_str=query_str, 
            )
        else:
            prompt = self.qa_tmpl_str.format(
                context_str="",
                metadata_str="",
                query_str=query_str, 
            )
        print(prompt)
        qs = prompt.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # TODO：考虑这里是否需要将所有图片都放进去
        image_tensor = process_images(images, self.image_processor, self.model.config)[0] 
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        record_data.update({"outputs": outputs})
        return outputs, record_data

    def inference(self):
        set_seed(0)
        disable_torch_init()
        with open(args.question_file, 'r', encoding='utf-8') as file:
            questions = json.load(file)
        # questions = [json.loads(q) for q in open(args.question_file,"r")]
        # do inference
        for line in tqdm(questions):
            qs = line["text"].replace(DEFAULT_IMAGE_TOKEN, '').strip()
            image_file = line["image"]
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
                
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            return outputs
            # TODO:批量测试需要记录上下文信息
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/data/eye_diag.json")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--image-folder", type=str, default="./segmentation/")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--meta-data", type=str, default="/home/hongyu/Visual-RAG-LLaVA-Med/data/segmentation.json")
    parser.add_argument("--use-pics", type=bool, default=False)
    parser.add_argument("--use-rag", type=bool, default=False)
    args = parser.parse_args()
    vrag = VRAG(args)
    # print(vrag.inference())
    # vrag.build_index()
    test_img = "/home/hongyu/DDR/lesion_segmentation/test/image/007-1789-100.jpg"
    query_str_0 = "Can you describe the image in details?"
    query_str_1 = "what's the diagnosis?"
    print(vrag.inference_rag(query_str_1, test_img))
    # vrag.build_index()