# emb-crop raw 
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
output_path="./output/${dataset}/llava/raw_${m}_${n}.json"
log_path="./output/${dataset}/llava/log/raw_${m}_${n}.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-v1.6-mistral-7b"
nohup python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --crop-emb-path ${crop_emb_path} \
    --level-emb-path ${level_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --model-path ${model_path} \
    --test-num -1 \
    >${log_path} 2>&1 &

# both emb rag 
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
output_path="./output/${dataset}/llava/${crop_emb}_${level_emb}_${m}_${n}_rag.json"
log_path="./output/${dataset}/llava/log/${crop_emb}_${level_emb}_${m}_${n}_rag.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-v1.6-mistral-7b"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --crop-emb-path ${crop_emb_path} \
    --level-emb-path ${level_emb_path} \
    --model-path ${model_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --test-num -1 \
    >${log_path} 2>&1 &

# level-emb rag 
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
output_path="./output/${dataset}/llava/${level_emb}_${m}_${n}_rag.json"
log_path="./output/${dataset}/llava/log/${level_emb}_${m}_${n}_rag.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-v1.6-mistral-7b"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --level-emb-path ${level_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --model-path ${model_path} \
    --use-rag True \
    --test-num -1 \
    >${log_path} 2>&1 &

# crop-emb rag
crop_emb=emb_crop
level_emb=level_emb
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
dataset=DR
m=1
n=1
output_path="./output/${dataset}/llava/${crop_emb}_${m}_${n}_rag.json"
log_path="./output/${dataset}/llava/log/${crop_emb}_${m}_${n}_rag.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-v1.6-mistral-7b"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --crop-emb-path ${crop_emb_path} \
    --model-path ${model_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --test-num 5 \
    >${log_path} 2>&1 &


# level crop combined rag
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
output_path="./output/${dataset}/llava/${crop_emb}_${level_emb}_${m}_${n}_rag.json"
log_path="./output/${dataset}/llava/log/${crop_emb}_${level_emb}_${m}_${n}_rag.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-v1.6-mistral-7b"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --crop-emb-path ${crop_emb_path} \
    --level-emb-path ${level_emb_path} \
    --model-path ${model_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --test-num 5 \
    >${log_path} 2>&1 &
