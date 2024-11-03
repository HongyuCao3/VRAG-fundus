export CUDA_VISIBLE_DEVICES=0,1,2,3

# level rag llava-med finetuned with pics
crop_emb=emb_crop
level_emb=level_emb_clip
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med-finetuned/${level_emb}_${m}_${n}_rag_${test_num}_pics.json"
log_path="./output/${dataset}/llava-med-finetuned/log/${level_emb}_${m}_${n}_rag_${tet_num}_pics.log"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --model-path ${model_path} \
    --level-emb-path ${level_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --use-pics True \
    --test-num ${test_num} \
    >${log_path} 2>&1 &


crop_emb=crop_emb_clip
level_emb=level_emb_clip
dataset=DR
crop_emb_path="./data/${crop_emb}"
level_emb_path="./data/${level_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med-finetuned/${level_emb}_${crop_emb}_${m}_${n}_rag_${test_num}_pics.json"
log_path="./output/${dataset}/llava-med-finetuned/log/${level_emb}_${crop_emb}_${m}_${n}_rag_${tet_num}_pics.log"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --model-path ${model_path} \
    --level-emb-path ${level_emb_path} \
    --crop-emb-path ${crop_emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --use-pics True \
    --test-num ${test_num} \
    >${log_path} 2>&1 &