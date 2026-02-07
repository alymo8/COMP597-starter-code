import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_OPT_NAME = "facebook/opt-350m"

def load_opt(model_name: str = DEFAULT_OPT_NAME, dtype: str = "fp16"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )

    return model, tokenizer

