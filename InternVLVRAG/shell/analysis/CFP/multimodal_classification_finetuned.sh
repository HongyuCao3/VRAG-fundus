classic_emb=classic_emb_cfp
dataset=MultiModal
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
emb_path=${cur_path}"/KnowledgeBase/emb_savings/"${classic_emb}
model_name="internvl"
sheet_names="CFP"
m=1
n=1
test_num=-1
t_check=-1
t_filter=-1
save_tmp=raw_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names}
output_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/${save_tmp}.json"
save_path=${cur_path}"/InternVLVRAG/output/${dataset}/${model_name}/${save_tmp}.png"
map_path="mapping_relationship.xlsx"
conda activate internvl_louwei
python ./InternVLVRAG/analysis.py \
    --res-path ${output_path} \
    --map-path ${map_path} \
    --save-path ${save_path}