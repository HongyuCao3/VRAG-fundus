export CUDA_VISIBLE_DEVICES=0,1,2,3

# raw
classic_emb=classic_emb_oct
dataset=LesionBalanced
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
emb_path=${cur_path}"/KnowledgeBase/emb_savings/"${classic_emb}
model_name="finetuned"
sheet_names="New"
m=1
n=1
test_num=-1
t_check=-1
t_filter=-1
save_tmp=raw_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names}
output_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/${save_tmp}.csv"
log_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/log/${save_tmp}.log"
model_path="/home/hongyu/InternVL/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_fulldataset"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate internvl_louwei
nohup python ./InternVLVRAG/evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --model-path ${model_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --use-pics True \
    --test-num ${test_num} \
    --mode ALL \
    --dynamic \
    --sheet-names ${sheet_names} \
    >${log_path} 2>&1 &


# cfp emb
classic_emb=classic_emb_cfp
dataset=LesionBalanced
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
emb_path=${cur_path}"/KnowledgeBase/emb_savings/"${classic_emb}
model_name="finetuned"
sheet_names="New"
m=1
n=1
test_num=-1
t_check=-1
t_filter=-1
save_tmp=${classic_emb}_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names}
output_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/${save_tmp}.csv"
log_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/log/${save_tmp}.log"
model_path="/home/hongyu/InternVL/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_fulldataset"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate internvl_louwei
nohup python ./InternVLVRAG/evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --model-path ${model_path} \
    --classic-emb-path ${emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --use-pics True \
    --test-num ${test_num} \
    --mode ALL \
    --dynamic \
    --sheet-names ${sheet_names} \
    >${log_path} 2>&1 &


# ffa emb
classic_emb=classic_emb_ffa
dataset=LesionBalanced
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
emb_path=${cur_path}"/KnowledgeBase/emb_savings/"${classic_emb}
model_name="finetuned"
sheet_names="New"
m=1
n=1
test_num=-1
t_check=-1
t_filter=-1
save_tmp=${classic_emb}_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names}
output_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/${save_tmp}.csv"
log_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/log/${save_tmp}.log"
model_path="/home/hongyu/InternVL/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_lora_fulldataset"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate internvl_louwei
nohup python ./InternVLVRAG/evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --model-path ${model_path} \
    --classic-emb-path ${emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --use-pics True \
    --test-num ${test_num} \
    --mode ALL \
    --dynamic \
    --sheet-names ${sheet_names} \
    >${log_path} 2>&1 &