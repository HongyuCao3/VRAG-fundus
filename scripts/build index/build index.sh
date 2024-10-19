/home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python index_builder.py \
    --crop-dir "./data/lesion/" \
    --level-dir "./data/level/" \
    --persist-dir "./data/level_emb/" \