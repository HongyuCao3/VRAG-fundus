import argparse
import pathlib
import sys

sys.path.append(pathlib.Path.cwd())
from HuatuoGPTVRAG.HuatuoGPT.huatuo_cli_demo_stream import load_model


class HuatuoGPTVRAG:
    def __init__(
        self,
        model_name: str = "/home/hongyu/Visual-RAG-LLaVA-Med/Model/HuatuoGPT-7B",
        device: str = "cuda",
        num_gpus: int = 1,
    ):
        model, self.tokenizer = load_model(model_name, device, num_gpus)

        self.model = model.eval()
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="HuatuoGPT")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-gpus", type=str, default="1")
    # parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
