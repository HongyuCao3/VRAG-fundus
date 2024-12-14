export CUDA_VISIBLE_DEVICES=0,1,2,3

# raw
classic_emb=classic_emb_clip
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med-finetuned/raw_${m}_${n}_rag_${test_num}3.json"
log_path="./output/${dataset}/llava-med-finetuned/log/raw_${m}_${n}_rag_${test_num}3.log"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python evaluation_llava.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --model-path ${model_path} \
    --conv-mode mistral_instruct \
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
output_path="./output/${dataset}/llava-med-finetuned/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.json"
log_path="./output/${dataset}/llava-med-finetuned/log/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.log"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python evaluation_llava.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --model-path ${model_path} \
    --classic-emb-path ${classic_emb_path} \
    --conv-mode mistral_instruct \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --use-pics True \
    --test-num ${test_num} \
    --mode ALL \
    >${log_path} 2>&1 &

# classic emb filter
classic_emb=classic_emb_clip
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med-finetuned/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.json"
log_path="./output/${dataset}/llava-med-finetuned/log/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.log"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python evaluation_llava.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --model-path ${model_path} \
    --classic-emb-path ${classic_emb_path} \
    --conv-mode mistral_instruct \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --use-pics True \
    --test-num ${test_num} \
    --mode ALL \
    >${log_path} 2>&1 &