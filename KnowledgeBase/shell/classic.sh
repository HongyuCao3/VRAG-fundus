python emb_builder.py \
    --img-path "./data/Classic Images/" \
    --emb-path "./data/classic_emb_mvqa/"


# FFA emb
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
img_dir=${cur_path}"/KnowledgeBase/img_savings/Classic_Images_(FFA)"
emb_path=${cur_path}"/KnowledgeBase/emb_savings/classic_emb_ffa"
python ./KnowledgeBase/emb_builder.py \
    --img-path ${img_dir} \
    --emb-path ${emb_path}

# OCT emb
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
img_dir=${cur_path}"/KnowledgeBase/img_savings/Classic_Images_(OCT)"
emb_path=${cur_path}"/KnowledgeBase/emb_savings/classic_emb_oct"
python ./KnowledgeBase/emb_builder.py \
    --img-path ${img_dir} \
    --emb-path ${emb_path}