from QwenVLVRAG import load_model_tokenizer

class QwenVLVRAG():
    def __init__(self, checkpoint_path):
        model, tokenizer = load_model_tokenizer()