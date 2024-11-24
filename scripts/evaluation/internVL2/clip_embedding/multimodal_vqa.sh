export CUDA_VISIBLE_DEVICES=0,1,2,3

# raw
classic_emb=classic_emb_clip
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/internVL2/raw_${m}_${n}_rag_${test_num}_pics2.json"
log_path="./output/${dataset}/internVL2/log/raw_${m}_${n}_rag_${test_num}_pics2.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
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
    --mode ALL \
    >${log_path} 2>&1 &


# classic emb
classic_emb=classic_emb_clip
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/internVL2/${classic_emb}_${m}_${n}_rag_${test_num}_pics.json"
log_path="./output/${dataset}/internVL2/log/${classic_emb}_${m}_${n}_rag_${test_num}_pics.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
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