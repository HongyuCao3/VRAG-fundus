
# crop & level
crop_emb=crop_emb_clip
level_emb=level_emb_clip
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/internVL2/${level_emb}_${crop_emb}_${m}_${n}_rag_multiturn_${test_num}_pics.json"
res_path="./output/${dataset}/internVL2/analysis/${level_emb}_${crop_emb}_${m}_${n}_rag_multiturn_${test_num}_pics.txt"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis.py \
    --file-path ${output_path} \
    --res-path ${res_path} \
    --level-emb ${level_emb_path}s