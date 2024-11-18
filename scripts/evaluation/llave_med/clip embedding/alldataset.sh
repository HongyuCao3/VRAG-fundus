export CUDA_VISIBLE_DEVICES=0,1,2,3

# raw
classic_emb=classic_emb_clip
dataset=ALL
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med/raw_${m}_${n}_rag_${test_num}_2.json"
log_path="./output/${dataset}/llava-med/raw_${m}_${n}_rag_${test_num}_2.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --model-path ${model_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --use-pics True \
    --test-num ${test_num} \
    --mode ${dataset} \
    >${log_path} 2>&1 &

# classic emb
classic_emb=classic_emb_clip
dataset=ALL
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med/${classic_emb}_${m}_${n}_rag_${test_num}_pics2.json"
log_path="./output/${dataset}/llava-med/log/${classic_emb}_${m}_${n}_rag_${test_num}_pics2.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --model-path ${model_path} \
    --classic-emb-path ${classic_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --use-pics True \
    --test-num ${test_num} \
    --mode ${dataset} \
    >${log_path} 2>&1 &