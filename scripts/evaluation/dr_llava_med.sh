# emb-crop raw llava-med
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
output_path="./output/${dataset}/raw_${m}_${n}.json"
log_path="./output/${dataset}/log/raw_${m}_${n}.log"
nohup python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --crop-emb-path ${crop_emb_path} \
    --level-emb-path ${level_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --test-num -1 \
    >${log_path} 2>&1 &

# both emb rag llava-med
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
output_path="./output/${dataset}/${crop_emb}_${level_emb}_${m}_${n}_rag.json"
log_path="./output/${dataset}/log/${crop_emb}_${level_emb}_${m}_${n}_rag.log"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --crop-emb-path ${crop_emb_path} \
    --level-emb-path ${level_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --test-num -1 \
    >${log_path} 2>&1 &

# level-emb rag llava-med
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
output_path="./output/${dataset}/${level_emb}_${m}_${n}_rag.json"
log_path="./output/${dataset}/log/${level_emb}_${m}_${n}_rag.log"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --level-emb-path ${level_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --test-num -1 \
    >${log_path} 2>&1 &

# crop-emb rag llava-med
crop_emb=emb_crop
level_emb=level_emb
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
dataset=DR
m=1
n=1
output_path="./output/${dataset}/${crop_emb}_${m}_${n}_rag.json"
log_path="./output/${dataset}/log/${crop_emb}_${m}_${n}_rag.log"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --crop-emb-path ${crop_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --test-num 5 \
    >${log_path} 2>&1 &


# level crop combined rag llava-med
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
output_path="./output/${dataset}/${crop_emb}_${level_emb}_${m}_${n}_rag.json"
log_path="./output/${dataset}/log/${crop_emb}_${level_emb}_${m}_${n}_rag.log"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --crop-emb-path ${crop_emb_path} \
    --level-emb-path ${level_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --test-num 5 \
    >${log_path} 2>&1 &


# classic-emb rag llava-med
crop_emb=emb_crop
level_emb=level_emb
classic_emb=classic_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
classic_emb_path="./data/${classic_emb}"
m=1
n=1
output_path="./output/${dataset}/${classic_emb}_${m}_${n}_rag.json"
log_path="./output/${dataset}/log/${classic_emb}_${m}_${n}_rag.log"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --classic-emb-path ${classic_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --test-num -1 \
    >${log_path} 2>&1 &