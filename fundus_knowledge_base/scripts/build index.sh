# build level
/home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python index_builder.py \
    --level-dir "./data/level/" \
    --persist-dir "./data/level_emb/" 

# build level with rotation
/home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python index_builder.py \
    --level-dir "./data/level_copy/" \
    --persist-dir "./data/level_emb_r/" 


# build all
/home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python index_builder.py \
    --crop-dir "./data/lesion/" \
    --level-dir "./data/level/" \
    --persist-dir "./data/level_crop_emb/" 

# build classic
/home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python index_builder.py \
    --classic-dir "./data/Classic Images/" \
    --persist-dir "./data/classic_emb/" 

# build classic dr
/home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python index_builder.py \
    --classic-dir "./data/Classic Images/" \
    --persist-dir "./data/classic_dr_emb/" 


# build classic dr large base
/home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python index_builder.py \
    --classic-dir "./data/Classic Images/" \
    --persist-dir "./data/classic_dr_emb_large/" \
    --embedding-name "BAAI/bge-large-en-v1.5"

cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python ./fundus_knowledge_base/index_builder.bkup.py \
    --classic-dir "./data/Classic Images/" \
    --persist-dir "./fundus_knowledge_base/emb_savings/mulit_disease_clip_embeddings/" \