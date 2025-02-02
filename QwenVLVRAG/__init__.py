
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)

def load_model_tokenizer(checkpoint_path: str, cpu_only: bool=False):
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=True, resume_download=True, revision='master',
    )

    if cpu_only:
        device_map = "cpu"
    else:
        device_map = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
        revision='master',
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        checkpoint_path, trust_remote_code=True, resume_download=True, revision='master',
    )

    return model, tokenizer