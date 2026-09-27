"""
Model loader and inference module for LLM-XRay.
Provides internal inspection of embeddings, transformer layers, attention weights,
hidden states, logits, and token-by-token generation.
"""
from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, str]:
    """
    Loads causal LM model with low CPU memory footprint.
    Returns (model, device_str).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (
        torch.float16 if device == "cuda" else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=True
    )
    model.to(device)
    model.eval()
    return model, device


@torch.no_grad()
def inspect_inference_internals(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Runs a single forward pass with output_attentions=True and output_hidden_states=True.
    Extracts all intermediate tensor data needed for LLM-XRay visualizations:
    - Input Embeddings
    - Attention matrices across all layers & heads
    - Hidden states across all layers
    - Output Logits & Softmax Probabilities for the next token
    - Top-k candidate predictions
    """
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
    
    seq_len = input_ids.shape[1]
    
    # 1. Input Embeddings: extract using model's embedding layer
    # For Qwen2: model.model.embed_tokens
    embedding_layer = model.get_input_embeddings()
    input_embeddings = embedding_layer(input_ids) # [1, seq_len, hidden_dim]
    
    # 2. Forward pass with full internal outputs
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True
    )
    
    # outputs.hidden_states is a tuple of (num_layers + 1)
    # Layer 0 is embedding output, Layer 1..N are transformer layer outputs
    hidden_states = [h.detach().cpu().to(torch.float32) for h in outputs.hidden_states]
    
    # outputs.attentions is a tuple of num_layers
    # Each tensor is [batch_size, num_heads, seq_len, seq_len]
    attentions = [a.detach().cpu().to(torch.float32) for a in outputs.attentions]
    
    # 3. Next token Logits & Probabilities
    # Last token logits: [vocab_size]
    last_logits = outputs.logits[0, -1, :].detach().cpu().to(torch.float32)
    probabilities = F.softmax(last_logits, dim=-1)
    
    # Top 15 predictions
    topk_k = 15
    topk_probs, topk_indices = torch.topk(probabilities, k=topk_k)
    top_predictions = []
    for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
        token_str = tokenizer.decode([idx])
        top_predictions.append({
            "token_id": idx,
            "token": token_str if token_str.strip() else repr(token_str),
            "clean_token": token_str,
            "probability": prob,
            "percentage": f"{prob * 100:.2f}%",
            "logit": float(last_logits[idx])
        })

    # Model architecture metadata
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1] if num_layers > 0 else 0
    hidden_dim = hidden_states[0].shape[-1]
    
    return {
        "prompt": prompt,
        "input_ids": input_ids.cpu(),
        "seq_len": seq_len,
        "input_embeddings": input_embeddings.detach().cpu().to(torch.float32),
        "hidden_states": hidden_states,
        "attentions": attentions,
        "last_logits": last_logits,
        "probabilities": probabilities,
        "top_predictions": top_predictions,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "hidden_dim": hidden_dim,
        "vocab_size": model.config.vocab_size
    }


@torch.no_grad()
def generate_step_by_step(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 15,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cpu"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generates response token by token, capturing intermediate decisions,
    selected token probabilities, and runner-up candidate tokens at every step.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    generated_ids = []
    step_history = []
    
    eos_token_id = tokenizer.eos_token_id
    
    for step in range(max_new_tokens):
        outputs = model(input_ids=input_ids, return_dict=True)
        next_token_logits = outputs.logits[0, -1, :].clone().to(torch.float32)
        
        # Apply temperature
        if temperature > 0:
            scaled_logits = next_token_logits / temperature
        else:
            scaled_logits = next_token_logits
            
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Top candidate choices before sampling
        top_probs, top_indices = torch.topk(probs, k=5)
        step_candidates = []
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            step_candidates.append({
                "token_id": idx,
                "token": repr(tokenizer.decode([idx])),
                "prob": p
            })
            
        # Sampling logic
        if temperature == 0:
            # Greedy
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        else:
            # Top-K filtering
            if top_k > 0:
                indices_to_remove = scaled_logits < torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))[0][..., -1, None]
                scaled_logits[indices_to_remove] = -float("Inf")
                
            # Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep at least first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                scaled_logits[indices_to_remove] = -float("Inf")
                
            filtered_probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(filtered_probs, num_samples=1)
            
        next_token_id = next_token.item()
        token_str = tokenizer.decode([next_token_id])
        token_prob = probs[next_token_id].item()
        
        generated_ids.append(next_token_id)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        current_full_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        step_history.append({
            "step": step + 1,
            "token_id": next_token_id,
            "token_str": token_str,
            "display_str": token_str if token_str.strip() else repr(token_str),
            "prob": token_prob,
            "percentage": f"{token_prob * 100:.1f}%",
            "current_text": current_full_text,
            "candidates": step_candidates
        })
        
        if next_token_id == eos_token_id:
            break
            
    full_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return full_response, step_history
