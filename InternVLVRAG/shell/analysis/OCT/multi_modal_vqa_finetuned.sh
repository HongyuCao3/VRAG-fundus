# raw
classic_emb=classic_emb_oct
dataset=MultiModal
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
emb_path=${cur_path}"/KnowledgeBase/emb_savings/"${classic_emb}
model_name="finetuned"
sheet_names="OCT"
m=1
n=1
test_num=-1
t_check=-1
t_filter=-1
save_tmp=raw_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names}
output_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/${save_tmp}.json"
log_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/log/${save_tmp}.log"
res_path="${cur_path}/InternVLVRAG/output/${dataset}/${model_name}/analysis/${save_tmp}.png"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate internvl_louwei
python ./AnalysisVRAG/main.py \
    --file-path ${output_path} \
    --res-path ${res_path} \
    --sheet-names ${sheet_names} 

# classic emb
classic_emb=classic_emb_oct
dataset=MultiModal
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
classic_emb_path=${cur_path}"/KnowledgeBase/emb_savings/"${classic_emb}
model_name="finetuned"
sheet_names="OCT"
m=1
n=1
test_num=-1
t_check=-1
t_filter=-1
save_tmp=${classic_emb}_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names}
output_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/${save_tmp}.json"
log_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/log/${save_tmp}.log"
res_path="${cur_path}/InternVLVRAG/output/${dataset}/${model_name}/analysis/${save_tmp}.png"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate internvl_louwei
python ./AnalysisVRAG/main.py \
    --file-path ${output_path} \
    --res-path ${res_path} \
    --sheet-names ${sheet_names} 

# filter
classic_emb=classic_emb_oct
dataset=MultiModal
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
classic_emb_path=${cur_path}"/KnowledgeBase/emb_savings/"${classic_emb}
model_name="finetuned"
sheet_names="OCT"
m=1
n=1
test_num=-1
t_check=-1
t_filter=0.7
save_tmp=${classic_emb}_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names}
output_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/${save_tmp}.json"
log_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/log/${save_tmp}.log"
res_path="${cur_path}/InternVLVRAG/output/${dataset}/${model_name}/analysis/${save_tmp}.png"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate internvl_louwei
python ./AnalysisVRAG/main.py \
    --file-path ${output_path} \
    --res-path ${res_path} \
    --sheet-names ${sheet_names} 

# filter+check
classic_emb=classic_emb_oct
dataset=MultiModal
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
classic_emb_path=${cur_path}"/KnowledgeBase/emb_savings/"${classic_emb}
model_name="finetuned"
sheet_names="OCT"
m=1
n=1
test_num=-1
t_check=0.5
t_filter=0.5
save_tmp=${classic_emb}_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names}
output_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/${save_tmp}.json"
log_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/log/${save_tmp}.log"
res_path="${cur_path}/InternVLVRAG/output/${dataset}/${model_name}/analysis/${save_tmp}.png"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate internvl_louwei
python ./AnalysisVRAG/main.py \
    --file-path ${output_path} \
    --res-path ${res_path} \
    --sheet-names ${sheet_names} 