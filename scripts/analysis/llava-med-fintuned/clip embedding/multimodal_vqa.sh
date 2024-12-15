# raw
classic_emb=classic_emb_clip
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
output_path="./output/${dataset}/llava-med-finetuned/raw_${m}_${n}_rag_${test_num}3.json"
log_path="./output/${dataset}/llava-med-finetuned/log/raw_${m}_${n}_rag_${test_num}3.log"
res_path="./output/${dataset}/llava-med-finetuned/analysis/raw_${m}_${n}_rag_${test_num}3.png"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
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
output_path="./output/${dataset}/llava-med-finetuned/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.json"
log_path="./output/${dataset}/llava-med-finetuned/log/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.log"
res_path="./output/${dataset}/llava-med-finetuned/analysis/${classic_emb}_${m}_${n}_rag_${test_num}_pics3.png"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis_.py \
    --file-path ${output_path} \
    --res-path ${res_path}

# classic_emb filter
classic_emb=classic_emb_clip
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
t_filter=0.5
t_check=-1
model_name="llava-med-finetuned"
save_tmp=${classic_emb}_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_2d_mh
output_path="./output/${dataset}/${model_name}/${save_tmp}.json"
log_path="./output/${dataset}/${model_name}/log/${save_tmp}.log"
res_path="./output/${dataset}/${model_name}.png"
model_path="/home/hongyu/eye_llava_medllava_finetune_mistral"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python analysis_.py \
    --file-path ${output_path} \
    --res-path ${res_path}