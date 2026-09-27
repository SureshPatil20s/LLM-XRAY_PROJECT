"""
Tokenizer module for LLM-XRay.
Handles text tokenization, token inspection, and token ID mapping.
"""
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer


def load_tokenizer(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> AutoTokenizer:
    """Load and return the Hugging Face tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer


def get_token_details(text: str, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Tokenizes input text and returns detailed token information:
    - input_ids: tensor and list
    - tokens: readable string tokens
    - token_table: list of dicts with index, token, token_id, and character span
    - display_str: formatted 'Token1 | Token2 | ...'
    """
    encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoding["input_ids"][0]  # 1D tensor
    
    token_ids_list = input_ids.tolist()
    
    tokens = []
    token_table = []
    
    for idx, tid in enumerate(token_ids_list):
        # Decode single token safely
        token_str = tokenizer.decode([tid])
        # Also get tokenizer's internal representation if available
        raw_repr = repr(token_str)
        
        # Friendly display label
        if token_str == " ":
            display_label = "[SPACE]"
        elif token_str == "\n":
            display_label = "[NEWLINE]"
        elif token_str == "\t":
            display_label = "[TAB]"
        elif token_str.strip() == "":
            display_label = f"[WS: {raw_repr}]"
        else:
            display_label = token_str
            
        tokens.append(display_label)
        token_table.append({
            "Index": idx,
            "Token": display_label,
            "Clean Text": token_str,
            "Token ID": tid,
            "Length": len(token_str)
        })
        
    display_tokens_str = " | ".join(tokens)
    display_ids_str = " | ".join(str(tid) for tid in token_ids_list)
    
    return {
        "text": text,
        "input_ids": input_ids.unsqueeze(0),  # [1, seq_len]
        "token_ids_list": token_ids_list,
        "tokens": tokens,
        "token_table": token_table,
        "display_tokens_str": display_tokens_str,
        "display_ids_str": display_ids_str,
        "total_tokens": len(token_ids_list)
    }
