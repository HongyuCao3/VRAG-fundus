import os, json

path = "/home/hongyu/Visual-RAG-LLaVA-Med/segmentation"
discription = {}
idx = 0
for file in os.listdir(path):
    base_name = os.path.splitext(file)[0]
    if base_name in discription:
        discription[base_name]["file"] = file
    else:
        discription.update({base_name: {"file": "", "seg": file, "idx": idx, "discription": ""}})
        idx += 1

tgt_path = "/home/hongyu/Visual-RAG-LLaVA-Med/data/segmentation.json"
with open(tgt_path, "w", encoding="UTF-8") as f:
    json.dump(discription, f)