classic_emb=classic_emb_clip
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med/raw_${m}_${n}_rag_${test_num}3.json"
log_path="./output/${dataset}/llava-med/log/raw_${m}_${n}_rag_${test_num}3.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis_.py \
    --res-path ${output_path}

# classic emb
classic_emb=classic_emb_clip
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.json"
log_path="./output/${dataset}/llava-med/log/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis_.py \
    --res-path ${output_path}