export CUDA_VISIBLE_DEVICES=0,1,2,3

# raw
classic_emb=classic_emb_cfp
dataset=MultiModalVQA
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
emb_path=${cur_path}"/KnowledgeBase/emb_savings/"${classic_emb}
model_name="llava-med"
sheet_names="CFP"
m=1
n=1
test_num=-1
save_tmp=raw_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_$
output_path=${cur_path}"/LLavaVRAG/output/${dataset}/${model_name}/${save_tmp}.csv"
log_path=${cur_path}"/LLavaVRAG/output/${dataset}/${model_name}/log/${save_tmp}.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python ./LLavaVRAG/evaluation.py \
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
    --sheet-names ${sheet_names} \
    >${log_path} 2>&1 &

# classic
classic_emb=classic_emb_cfp
dataset=MultiModalVQA
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
emb_path=${cur_path}"/KnowledgeBase/emb_savings/"${classic_emb}
model_name="llava-med"
sheet_names="CFP"
m=1
n=1
test_num=-1
save_tmp=${classic_emb}_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names}
output_path=${cur_path}"/LLavaVRAG/output/${dataset}/${model_name}/${save_tmp}.csv"
log_path=${cur_path}"/LLavaVRAG/output/${dataset}/${model_name}/log/${save_tmp}.log"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/llava-med-v1.5-mistral-7b"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
nohup python ./LLavaVRAG/evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --model-path ${model_path} \
    --conv-mode mistral_instruct \
    --classic-emb-path ${emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --use-pics True \
    --test-num ${test_num} \
    --mode ALL \
    --sheet-names ${sheet_names} \
    >${log_path} 2>&1 &