# raw
crop_emb=crop_emb_clip
level_emb=level_emb_clip
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/internVL2/raw_${m}_${n}_rag_${test_num}_pics.json"
res_path="./output/${dataset}/internVL2/analysis/raw_${m}_${n}_rag_${test_num}_pics_b.png"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis_multi_turn.py \
    --file-path ${output_path} \
    --res-path ${res_path} \
    --level-emb ${level_emb_path}

# crop & level
crop_emb=crop_emb_clip
level_emb=level_emb_clip
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
test_num=-1
step=4
output_path="./output/${dataset}/internVL2/${level_emb}_${crop_emb}_${m}_${n}_rag_multiturn_${test_num}_pics.json"
res_path="./output/${dataset}/internVL2/analysis/${level_emb}_${crop_emb}_${m}_${n}_rag_multiturn_${test_num}_step_${step}_b.png"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis_multi_turn.py \
    --file-path ${output_path} \
    --res-path ${res_path} \
    --level-emb ${level_emb_path} \
    --step ${step}

# crop & level check
crop_emb=crop_emb_clip
level_emb=level_emb_clip
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
test_num=-1
step=3
mode=MultiTurnCheck
output_path="./output/${dataset}/internVL2/${level_emb}_${crop_emb}_${m}_${n}_rag_multiturn_${test_num}_${mode}.json"
res_path="./output/${dataset}/internVL2/analysis/${level_emb}_${crop_emb}_${m}_${n}_rag_multiturn_${test_num}_${mode}_${step}.png"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis_multi_turn.py \
    --file-path ${output_path} \
    --res-path ${res_path} \
    --level-emb ${level_emb_path} \
    --step ${step}