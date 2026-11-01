"""
Tokenizer module for LLM-XRay.
Handles text tokenization, token inspection, UTF-8 byte breakdown,
special-token detection, and lexical compression metrics.
"""
import sys
from typing import List, Dict, Any
from transformers import AutoTokenizer


def load_tokenizer(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> AutoTokenizer:
    """Load and return the Hugging Face tokenizer."""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _utf8_hex(text: str) -> str:
    """Return a space-separated hex string of the UTF-8 bytes for `text`."""
    return " ".join(f"0x{b:02X}" for b in text.encode("utf-8"))


def get_token_details(
    text: str,
    tokenizer: AutoTokenizer,
    use_chat_template: bool = False,
) -> Dict[str, Any]:
    """
    Tokenizes input text and returns rich token information.

    When use_chat_template=True the prompt is wrapped in Qwen chat format
    so the pill representation matches what the model actually sees.

    Returns
    -------
    text                  : original input string
    formatted_text        : text that was actually tokenized (may include chat tags)
    input_ids             : [1, seq_len] tensor
    token_ids_list        : plain Python list of int IDs
    tokens                : display-safe label per token
    token_table           : list of dicts (see below)
    display_tokens_str    : "tok1 | tok2 | ..."
    display_ids_str       : "id1 | id2 | ..."
    total_tokens          : int
    total_chars           : int  (chars in formatted_text)
    compression_ratio     : float  chars / token
    special_token_ids     : set of int IDs that are special tokens
    """
    # ── Optionally wrap in chat template ─────────────────────────────────────
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": text}]
        formatted_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        formatted_text = text

    # ── Tokenize (keep special tokens so pills match model input) ─────────────
    encoding = tokenizer(
        formatted_text,
        return_tensors="pt",
        add_special_tokens=True,        # include <|im_start|> etc.
    )
    input_ids = encoding["input_ids"][0]
    token_ids_list: List[int] = input_ids.tolist()

    # Build the set of IDs the tokenizer considers special
    special_token_ids: set[int] = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_token_ids = set(tokenizer.all_special_ids)

    # ── Per-token detail ───────────────────────────────────────────────────────
    tokens: List[str] = []
    token_table: List[Dict[str, Any]] = []

    for idx, tid in enumerate(token_ids_list):
        raw_str: str = tokenizer.decode([tid], skip_special_tokens=False)
        is_special: bool = tid in special_token_ids

        # Human-readable display label
        if is_special:
            display_label = raw_str  # e.g. "<|im_start|>"
        elif raw_str == " ":
            display_label = "▁ (space)"
        elif raw_str == "\n":
            display_label = "↵ (newline)"
        elif raw_str == "\t":
            display_label = "⇥ (tab)"
        elif raw_str.strip() == "":
            display_label = repr(raw_str)
        else:
            display_label = raw_str

        tokens.append(display_label)
        token_table.append(
            {
                "Index":       idx,
                "Token String": display_label,
                "Token ID":    tid,
                "Length":      len(raw_str.encode("utf-8")),   # byte length
                "Special":     is_special,
                "UTF-8 Bytes": _utf8_hex(raw_str),
            }
        )

    total_chars = len(formatted_text)
    total_toks  = len(token_ids_list)
    ratio       = round(total_chars / total_toks, 2) if total_toks > 0 else 0.0

    return {
        "text":               text,
        "formatted_text":     formatted_text,
        "input_ids":          input_ids.unsqueeze(0),
        "token_ids_list":     token_ids_list,
        "tokens":             tokens,
        "token_table":        token_table,
        "display_tokens_str": " | ".join(tokens),
        "display_ids_str":    " | ".join(str(t) for t in token_ids_list),
        "total_tokens":       total_toks,
        "total_chars":        total_chars,
        "compression_ratio":  ratio,
        "special_token_ids":  special_token_ids,
    }
