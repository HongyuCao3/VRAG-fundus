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
output_path="./output/${dataset}/${crop_emb}_${level_emb}_${m}_${n}.json"
python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --crop-emb-path ${crop_emb_path} \
    --level-emb-path ${level_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --test-num 5

# both emb rag llava-med
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
output_path="./output/${dataset}/${crop_emb}_${level_emb}_${m}_${n}_rag.json"
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
    --test-num 5

# level-emb rag llava-med
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
output_path="./output/${dataset}/${level_emb}_${m}_${n}_rag.json"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --level-emb-path ${level_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --test-num 5

# emb-level rag llava-med
crop_emb=emb_crop
level_emb=level_emb
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
dataset=DR
m=1
n=1
output_path="./output/${dataset}/${crop_emb}_${level_emb}_${m}_${n}_rag.json"
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
    --test-num 5


# emb-level-crop rag llava-med
crop_emb=emb_crop
level_emb=level_emb
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
output_path="./output/${dataset}/${crop_emb}_${level_emb}_${m}_${n}_rag.json"
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
    --test-num 5

