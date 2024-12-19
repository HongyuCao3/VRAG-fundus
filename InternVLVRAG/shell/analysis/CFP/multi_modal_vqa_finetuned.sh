classic_emb=classic_emb_mvqa
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
model=internVL2_finetuned
t_check=-1
t_filter=-1
sheet_names=CFP
save_tmp=raw_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_modality_${sheet_names}
output_path="./output/${dataset}/${model}/raw_${m}_${n}_rag_${test_num}_pics3.json"
log_path="./output/${dataset}/${model}/log/raw_${m}_${n}_rag_${test_num}_pics3.log"
res_path="${cur_path}/InternVLVRAG/output/${dataset}/${model_name}/analysis/${save_tmp}.png"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate internvl_louwei
python ./AnalysisVRAG/main.py \
    --file-path ${output_path} \
    --res-path ${res_path}  \
    --sheet-names ${sheet_names}

# filter+check
classic_emb=classic_emb_mvqa
dataset=MultiModal
classic_emb_path="./data/${classic_emb}"
m=1
n=1
test_num=-1
cur_path="/home/hongyu/Visual-RAG-LLaVA-Med"
model=internVL2_finetuned
t_check=0.5
t_filter=0.5
sheet_names=CFP
save_tmp=${classic_emb}_${m}_${n}_rag_${test_num}_filter_${t_filter}_check_${t_check}_2d_mh
output_path="./output/${dataset}/${model}/${save_tmp}.json"
log_path="./output/${dataset}/${model}/log/${save_tmp}.log"
res_path="${cur_path}/InternVLVRAG/output/${dataset}/${model_name}/analysis/${save_tmp}.png"
model_path="/home/hongyu/Visual-RAG-LLaVA-Med/Model/InternVL2-8B"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate internvl_louwei
python ./AnalysisVRAG/main.py \
    --file-path ${output_path} \
    --res-path ${res_path}  \
    --sheet-names ${sheet_names}