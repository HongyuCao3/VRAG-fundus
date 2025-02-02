import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenVLVRAG:
    def __init__(self, checkpoint_path):
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, device_map="cuda", trust_remote_code=True
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.eod_id

        prompt = "<img>{}</img>{} Answer:"

    def inference(self, questions, max_new_tokens=128):
        input_ids = self.tokenizer(questions, return_tensors="pt", padding="longest")
        attention_mask = input_ids.attention_mask
        pred = self.model.generate(
            input_ids=input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=self.tokenizer.eod_id,
            eos_token_id=self.tokenizer.eod_id,
        )
        answers = [
            self.tokenizer.decode(
                _[input_ids.size(1) :].cpu(), skip_special_tokens=True
            ).strip()
            for _ in pred
        ]
        return answers
