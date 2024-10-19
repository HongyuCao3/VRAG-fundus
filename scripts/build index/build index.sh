# build level
/home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python index_builder.py \
    --level-dir "./data/level/" \
    --persist-dir "./data/level_emb/" 


# build all
/home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python index_builder.py \
    --crop-dir "./data/lesion/" \
    --level-dir "./data/level/" \
    --persist-dir "./data/level_crop_emb/" 