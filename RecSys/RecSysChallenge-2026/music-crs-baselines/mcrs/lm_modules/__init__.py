# from .llama import LLAMA_MODEL
# from .qwen import QWEN_MODEL
from .lm import LLM_MODEL

def load_lm_module(lm_type, device, attn_implementation, dtype):
    return LLM_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
    # if lm_type == "Qwen/Qwen3-4B":
    #     return QWEN_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
    # elif lm_type == "meta-llama/Llama-3.2-1B-Instruct":
    #     return LLAMA_MODEL(model_name=lm_type, device=device, attn_implementation=attn_implementation, dtype=dtype)
    # else:
    #     raise ValueError(f"Unsupported LM type: {lm_type}")
