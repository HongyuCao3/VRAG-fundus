# raw
classic_emb=classic_emb_clip
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/internVL2/raw_${m}_${n}_rag_${test_num}_pics3.json"
log_path="./output/${dataset}/internVL2/log/raw_${m}_${n}_rag_${test_num}_pics3.log"
res_path="./output/${dataset}/internVL2/analysis/raw_${m}_${n}_rag_${test_num}_pics.png"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis_.py \
    --file-path ${output_path} \
    --res-path ${res_path} 

# classic emb
classic_emb=classic_emb_clip
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/internVL2/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.json"
log_path="./output/${dataset}/internVL2/log/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.log"
res_path="./output/${dataset}/internVL2/analysis/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.png"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
conda activate llava-med
python analysis_.py \
    --file-path ${output_path} \
    --res-path ${res_path} 
