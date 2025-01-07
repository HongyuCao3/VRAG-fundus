classic_emb=classic_emb_ffa
dataset=MultiModal
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
emb_path=${cur_path}"/KnowledgeBase/emb_savings/"${classic_emb}
model_name="finetuned"
sheet_names="FFA"
m=1
n=1
test_num=-1
t_filter=0.7
t_check=-1
save_tmp=raw_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names}
output_path=${cur_path}"/LLavaVRAG/output/${dataset}/${model_name}/${save_tmp}.csv"
log_path=${cur_path}"/LLavaVRAG/output/${dataset}/${model_name}/log/${save_tmp}.log"
save_path=${cur_path}"/LLavaVRAG/output/${dataset}/${model_name}/analysis/${save_tmp}.png"
python ./AnalysisVRAG/main.py \
    --file-path ${output_path} \
    --res-path ${save_path}  \
    --sheet-names ${sheet_names}

classic_emb=classic_emb_ffa
dataset=MultiModal
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
emb_path=${cur_path}"/KnowledgeBase/emb_savings/"${classic_emb}
model_name="finetuned"
sheet_names="FFA"
m=1
n=1
test_num=-1
t_filter=0.7
save_tmp=${classic_emb}_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names}
output_path=${cur_path}"/LLavaVRAG/output/${dataset}/${model_name}/${save_tmp}.csv"
log_path=${cur_path}"/LLavaVRAG/output/${dataset}/${model_name}/log/${save_tmp}.log"
save_path=${cur_path}"/LLavaVRAG/output/${dataset}/${model_name}/analysis/${save_tmp}.png"
python ./AnalysisVRAG/main.py \
    --file-path ${output_path} \
    --res-path ${save_path}  \
    --sheet-names ${sheet_names}