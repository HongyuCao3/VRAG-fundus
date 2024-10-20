# emb-crop raw llava-med
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
emb=emb_crop
dataset=DR
emb_path="./data/${emb}"
m=1
n=1
output_path="./output/${dataset}/${emb}_${m}_${n}.json"
python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --emb-path ${emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --test-num 5

# emb-crop rag llava-med
emb=emb_crop
dataset=DR
emb_path="./data/${emb}"
m=1
n=1
output_path="./output/${dataset}/${emb}_${m}_${n}_rag.json"
cd /home/hongyu/Visual-RAG-LLaVA-Med
conda activate llava-med
python evaluation.py \
    --dataset ${dataset} \
    --output-path ${output_path} \
    --emb-path ${emb_path} \
    --chunk-m ${m} \
    --chunk-n ${n} \
    --use-rag True \
    --test-num 5



