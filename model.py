"""
Model loader and inference module for LLM-XRay.
Provides internal inspection of embeddings, transformer layers, attention weights,
hidden states, logits, and token-by-token generation.
"""
import re
import sys
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
    device: str = "cpu",
    use_chat_template: bool = False
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
    # If chat template is requested, format prompt to match tokenizer and generation
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = prompt

    # Tokenize input
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
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
    use_chat_template: bool = True,
    device: str = "cpu"
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generates response token by token, capturing intermediate decisions,
    selected token probabilities, runner-up candidates, and natural EOS detection
    to detect if the prompt only needed part of the allocated token budget.
    """
    # If chat template is requested, format prompt properly so Qwen knows when to emit <|im_end|>
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    generated_ids = []
    step_history = []
    
    # Set of EOS tokens for Qwen (<|im_end|>: 151645, <|endoftext|>: 151643)
    eos_token_ids = {tokenizer.eos_token_id, 151645, 151643}
    eos_token_ids.discard(None)
    
    stopped_early = False
    stop_reason = "Budget Exhausted: Reached maximum allocated tokens before model finished."

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
        
        # Check for natural EOS termination
        if next_token_id in eos_token_ids:
            stopped_early = True
            stop_reason = f"Natural Completion: Model emitted End-Of-Sequence token ({repr(token_str)}) because the answer was fully complete."
            break

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
            
    full_response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    tokens_used = len(step_history)
    tokens_saved = max(0, max_new_tokens - tokens_used)
    efficiency_pct = round((tokens_saved / max_new_tokens) * 100, 1) if max_new_tokens > 0 else 0
    
    usage_stats = {
        "allocated_budget": max_new_tokens,
        "tokens_used": tokens_used,
        "tokens_saved": tokens_saved,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "efficiency_pct": efficiency_pct,
        "used_full_budget": not stopped_early and tokens_used >= max_new_tokens
    }
    
    return full_response, step_history, usage_stats


TOKEN_TRANSLATIONS = {
    "换句话": "In other words",
    "下面小编": "Editor's intro",
    "回答": "Answer",
    "计算": "Calculate",
    "结果": "Result",
    "总结": "Summary",
    "解释": "Explain",
    "理解": "Understand",
    "第一": "First",
    "首先": "First of all",
    "综上所述": "In summary",
    "因此": "Therefore",
    "zellik": "feature",
}


def clean_and_gloss_token(raw_tok: str) -> str:
    """Cleans up raw token strings, truncates delimiter spam, and appends English glosses."""
    s = raw_tok.strip()
    # Check for long repetitive delimiters
    if len(s) > 15 and all(c in "=-_/*#~` \n\t\r" for c in s):
        return f"[Divider {s[:6]}...]"
    # Check for Chinese/foreign gloss
    stripped = s.strip("'").strip('"')
    if stripped in TOKEN_TRANSLATIONS:
        return f"{stripped} ({TOKEN_TRANSLATIONS[stripped]})"
    return raw_tok if raw_tok.strip() else repr(raw_tok)


@torch.no_grad()
def run_logit_lens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str = "cpu",
    use_chat_template: bool = True,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Applies the unembedding projection (final RMSNorm + LM Head) directly to the 
    hidden state of the last token at every layer (Layer 0 to Layer N).
    Reveals at which transformer layer the output token identity or semantic concept emerges.
    """
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = prompt

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True
    )

    norm = getattr(model, "model", model).norm
    lm_head = model.lm_head

    layer_predictions = []
    for l_idx, h in enumerate(outputs.hidden_states):
        last_h = h[:, -1, :].to(device)
        projected = lm_head(norm(last_h))
        probs = F.softmax(projected[0], dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)

        candidates = []
        for p, idx in zip(top_probs.tolist(), top_indices.tolist()):
            t_str = tokenizer.decode([idx])
            glossed = clean_and_gloss_token(t_str)
            candidates.append({
                "token_id": idx,
                "token": glossed,
                "clean_token": t_str,
                "prob": p,
                "percentage": f"{p * 100:.1f}%"
            })

        top_cand = candidates[0]
        layer_predictions.append({
            "layer": l_idx,
            "layer_name": "Layer 0 (Embeddings)" if l_idx == 0 else f"Layer {l_idx}",
            "top_token": top_cand["token"],
            "top_prob": top_cand["prob"],
            "percentage": top_cand["percentage"],
            "candidates": candidates
        })

    return layer_predictions


@torch.no_grad()
def run_deterministic_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 10,
    device: str = "cpu"
) -> str:
    """
    Executes greedy (temperature=0.0) autoregressive generation for causal intervention tests.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    new_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def analyze_computational_process(
    prompt: str,
    generated_text: str,
    step_history: List[Dict[str, Any]],
    usage_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deconstructs the computational and reasoning process that led to the final conclusion.
    Extracts intermediate calculations (if present), logical reasoning milestones,
    confidence metrics, decision pivot points, and how the model ended at its conclusion.
    """
    text = generated_text.strip()
    
    # 1. Extract Intermediate Calculations
    extracted_calcs = []
    lines = text.split("\n")
    
    for line in lines:
        clean_l = line.strip()
        if not clean_l:
            continue
        # Standardize symbols: \times, ×, \div, ÷
        norm_l = (
            clean_l.replace(r"\times", "*")
            .replace("×", "*")
            .replace(r"\div", "/")
            .replace("÷", "/")
            .replace(r"\[", "")
            .replace(r"\]", "")
        )
        # Match equations like: a * b = c
        matches = re.findall(
            r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)",
            norm_l
        )
        for m in matches:
            op1, op, op2, stated_str = float(m[0]), m[1], float(m[2]), m[3]
            stated_val = float(stated_str)
            gt_val = None
            if op == "+":
                gt_val = op1 + op2
            elif op == "-":
                gt_val = op1 - op2
            elif op == "*":
                gt_val = op1 * op2
            elif op == "/" and op2 != 0:
                gt_val = op1 / op2

            is_correct = (gt_val is not None) and (abs(gt_val - stated_val) < 1e-4)
            gt_formatted = int(gt_val) if gt_val is not None and gt_val.is_integer() else gt_val
            
            # Check chaining (did this step consume output from an earlier step?)
            chained_from = None
            for prev in extracted_calcs:
                if str(prev["stated_result"]) == m[0] or str(prev["stated_result"]) == m[2]:
                    chained_from = prev["raw_equation"]
                    break

            extracted_calcs.append({
                "step_index": len(extracted_calcs) + 1,
                "raw_equation": f"{m[0]} {m[1]} {m[2]} = {stated_str}",
                "operation": f"{m[0]} {m[1]} {m[2]}",
                "op1": m[0],
                "operator": m[1],
                "op2": m[2],
                "stated_result": stated_str,
                "ground_truth": gt_formatted,
                "is_correct": is_correct,
                "chained_from": chained_from
            })

    # 2. Extract Final Conclusion
    conclusion = None
    conclusion_type = "Text Statement"
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        conclusion = boxed.group(1).strip()
        conclusion_type = "LaTeX Boxed Answer"
    else:
        for line in reversed(lines):
            clean_l = line.strip()
            for prefix in ["therefore", "thus", "hence", "final answer", "in conclusion", "the result is", "the answer is"]:
                if clean_l.lower().startswith(prefix):
                    conclusion = clean_l
                    conclusion_type = "Deductive Concluding Statement"
                    break
            if conclusion:
                break
        if not conclusion and lines:
            non_empty = [l.strip() for l in lines if l.strip()]
            if non_empty:
                conclusion = non_empty[-1]
                conclusion_type = "Terminal Sentence"

    # 3. Deconstruct Step-by-Step Reasoning Flow / Milestones
    steps = []
    step_blocks = re.split(r"\n(?=\d+[\.\)]|\*|\-|\#\#|Step \d+)", text)
    if len(step_blocks) > 1:
        for idx, block in enumerate(step_blocks):
            b = block.strip()
            if not b:
                continue
            first_line = b.split("\n")[0][:50]
            steps.append({
                "milestone": idx + 1,
                "title": first_line if any(first_line.lower().startswith(p) for p in ["step", "1.", "2.", "3.", "4.", "5.", "*", "-"]) else f"Milestone {idx + 1}",
                "content": b
            })
    else:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paras and text:
            paras = [text]
        for idx, p in enumerate(paras):
            title = "1. Problem Framing" if idx == 0 else ("Final Synthesis" if idx == len(paras) - 1 else f"Stage {idx + 1}: Intermediate Reasoning")
            steps.append({
                "milestone": idx + 1,
                "title": title,
                "content": p
            })

    # 4. Critical Decision Points (Pivots from step_history)
    decision_pivots = []
    if step_history:
        for s in step_history:
            cands = s.get("candidates", [])
            if len(cands) >= 2:
                top_p = cands[0]["prob"]
                second_p = cands[1]["prob"]
                margin = top_p - second_p
                if margin < 0.35 or top_p < 0.65:
                    decision_pivots.append({
                        "step": s["step"],
                        "chosen_token": s["display_str"],
                        "confidence": s["percentage"],
                        "runner_up_token": cands[1]["token"],
                        "runner_up_prob": f"{second_p * 100:.1f}%",
                        "margin": f"{margin * 100:.1f}%"
                    })

    # 5. Narrative Explanation of How It Ended
    stopping_reason = usage_stats.get("stop_reason", "Completed")
    stopped_early = usage_stats.get("stopped_early", False)
    
    if stopped_early:
        termination_narrative = (
            "The model concluded naturally upon emitting an End-Of-Sequence token (<|im_end|>). "
            "It determined that the premise was satisfied, all necessary intermediate operations were resolved, "
            "and the final conclusion was reached."
        )
    else:
        termination_narrative = (
            "Generation halted because the allocated token budget was exhausted. "
            "The model reached the current point in its computational sequence, but additional token budget "
            "would be required to generate further elaboration."
        )

    return {
        "has_calculations": len(extracted_calcs) > 0,
        "calculations": extracted_calcs,
        "conclusion": conclusion if conclusion else "[No explicit conclusion reached]",
        "conclusion_type": conclusion_type,
        "reasoning_steps": steps,
        "decision_pivots": decision_pivots[:5],
        "stopping_reason": stopping_reason,
        "stopped_early": stopped_early,
        "termination_narrative": termination_narrative,
        "total_steps_executed": len(step_history)
    }


@torch.no_grad()
def run_chain_of_thought(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.4,
    device: str = "cpu",
    reasoning_style: str = "deep_thinking",
) -> dict:
    """
    Forces the model to reason through a problem using a Chain-of-Thought (CoT) system prompt.
    Supports two reasoning styles:
      - "deep_thinking": Instructs the model to write out its internal thoughts inside <think>...</think>
                        tags (as in DeepSeek-R1 / QwQ), then output the final answer.
      - "structured":    Instructs the model with numbered Step 1, Step 2, etc.

    Extracts:
        - raw_text:             the complete unedited generated text
        - thinking_process:     extracted internal thoughts (from <think> or internal scratchpad)
        - answer_text:          clean final answer without thinking tags
        - has_thinking_block:   bool
        - stages:               list of {stage_index, stage_type, title, content, token_count}
        - token_trace:          list of {step, token_id, token_str, prob, percentage, phase, cumulative_text}
        - thinking_tokens_count:int
        - answer_tokens_count:  int
        - final_answer:         extracted final answer line
        - total_tokens:         total tokens generated
        - stopped_naturally:    bool
    """
    if reasoning_style == "deep_thinking":
        cot_system = (
            "You are an advanced reasoning model. When given a problem, you must first think "
            "deeply inside <think> and </think> tags. In your thinking process, analyze the question, "
            "verify logical intermediate steps or arithmetic, and explore edge cases. "
            "After </think>, output your clear final answer."
        )
    else:
        cot_system = (
            "You are a precise analytical reasoning engine. "
            "When given a problem, always reason through it step by step, "
            "showing ALL intermediate steps clearly labeled as Step 1, Step 2, etc. "
            "If the problem involves calculation, show each arithmetic operation explicitly. "
            "End with a clearly labeled 'Final Answer:' line."
        )

    messages = [
        {"role": "system", "content": cot_system},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    eos_ids = {tokenizer.eos_token_id, 151645, 151643}
    eos_ids.discard(None)

    generated_ids: List[int] = []
    token_trace: List[Dict[str, Any]] = []
    stopped_naturally = False

    for step in range(max_new_tokens):
        outputs = model(input_ids=input_ids, return_dict=True)
        logits = outputs.logits[0, -1, :].clone().to(torch.float32)

        scaled = logits / temperature if temperature > 0 else logits
        probs = F.softmax(scaled, dim=-1)
        next_id = torch.argmax(scaled, dim=-1).item()
        token_prob = probs[next_id].item()
        token_str = tokenizer.decode([next_id])

        if next_id in eos_ids:
            stopped_naturally = True
            break

        generated_ids.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1
        )
        cumulative = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Detect phase (inside <think> vs outside)
        lower_cum = cumulative.lower()
        if "<think>" in lower_cum:
            if "</think>" in lower_cum:
                # Check if current token was generated after </think>
                after_think_text = lower_cum.split("</think>")[-1]
                current_phase = "thinking" if len(after_think_text.strip()) == 0 and "</think>" not in token_str.lower() else "answer"
            else:
                current_phase = "thinking"
        else:
            current_phase = "reasoning"

        token_trace.append({
            "step": step + 1,
            "token_id": next_id,
            "token_str": token_str,
            "prob": token_prob,
            "percentage": f"{token_prob * 100:.1f}%",
            "phase": current_phase,
            "cumulative_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        })

    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ---- Extract <think>...</think> if present ----
    thinking_process = ""
    answer_text = raw_text
    has_thinking_block = False

    # Check for <think> in generated text
    if "<think>" in raw_text.lower():
        has_thinking_block = True
        parts = re.split(r"</think>", raw_text, maxsplit=1, flags=re.IGNORECASE)
        think_raw = re.sub(r"^.*?<think>", "", parts[0], flags=re.DOTALL | re.IGNORECASE).strip()
        thinking_process = think_raw
        answer_text = parts[1].strip() if len(parts) > 1 else ""
    elif "<thought>" in raw_text.lower():
        has_thinking_block = True
        parts = re.split(r"</thought>", raw_text, maxsplit=1, flags=re.IGNORECASE)
        think_raw = re.sub(r"^.*?<thought>", "", parts[0], flags=re.DOTALL | re.IGNORECASE).strip()
        thinking_process = think_raw
        answer_text = parts[1].strip() if len(parts) > 1 else ""

    # Count tokens in thinking vs answer
    thinking_tokens_count = sum(1 for t in token_trace if t["phase"] == "thinking")
    answer_tokens_count = len(token_trace) - thinking_tokens_count

    # ---- Parse into labeled stages ----
    stages: List[Dict[str, Any]] = []

    if has_thinking_block and thinking_process:
        # Create thinking stages from the thinking block
        stage_index = 1
        stages.append({
            "stage_index": stage_index,
            "stage_type": "thinking_scratchpad",
            "title": "🧠 Internal Thinking (Scratchpad)",
            "content": thinking_process,
            "token_count": len(thinking_process.split()),
        })
        if answer_text:
            stage_index += 1
            stages.append({
                "stage_index": stage_index,
                "stage_type": "final_answer",
                "title": "🎯 Final Answer",
                "content": answer_text,
                "token_count": len(answer_text.split()),
            })
    else:
        # Standard line-by-line stage parsing
        lines = raw_text.split("\n")
        current_stage: Optional[Dict[str, Any]] = None
        stage_index = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            is_numbered_step = bool(
                re.match(r"^(step\s*\d+[\:\.\)]|\d+[\.\)])", stripped, re.IGNORECASE)
            )
            is_heading = stripped.startswith("##") or (
                stripped.startswith("**") and stripped.endswith("**")
            )
            is_final = bool(
                re.match(
                    r"^(final answer|therefore|thus|hence|in conclusion|the answer is)",
                    stripped, re.IGNORECASE
                )
            )

            if is_final:
                if current_stage:
                    stages.append(current_stage)
                stage_index += 1
                current_stage = {
                    "stage_index": stage_index,
                    "stage_type": "final_answer",
                    "title": "🎯 Final Answer",
                    "content": stripped,
                    "token_count": len(stripped.split()),
                }
                stages.append(current_stage)
                current_stage = None
            elif is_numbered_step or is_heading:
                if current_stage:
                    stages.append(current_stage)
                stage_index += 1
                label = stripped[:60].rstrip(":").strip("*").strip()
                current_stage = {
                    "stage_index": stage_index,
                    "stage_type": "reasoning_step",
                    "title": label,
                    "content": stripped,
                    "token_count": len(stripped.split()),
                }
            else:
                if current_stage:
                    current_stage["content"] += "\n" + stripped
                    current_stage["token_count"] += len(stripped.split())
                else:
                    stage_index += 1
                    current_stage = {
                        "stage_index": stage_index,
                        "stage_type": "context",
                        "title": "📋 Problem Setup",
                        "content": stripped,
                        "token_count": len(stripped.split()),
                    }

        if current_stage:
            stages.append(current_stage)

        # Fallback: treat whole text as paragraphs
        if not stages and raw_text:
            for idx, para in enumerate(
                [p.strip() for p in raw_text.split("\n\n") if p.strip()]
            ):
                stages.append({
                    "stage_index": idx + 1,
                    "stage_type": "reasoning_step" if idx > 0 else "context",
                    "title": f"Stage {idx + 1}",
                    "content": para,
                    "token_count": len(para.split()),
                })

    # Extract clean final answer line
    if has_thinking_block and answer_text:
        final_answer = answer_text
    else:
        final_answer = "[No explicit final answer detected]"
        for s in reversed(stages):
            if s["stage_type"] == "final_answer":
                final_answer = s["content"]
                break
        if final_answer == "[No explicit final answer detected]" and stages:
            final_answer = stages[-1]["content"].split("\n")[-1].strip()

    return {
        "cot_prompt": formatted,
        "raw_text": raw_text,
        "thinking_process": thinking_process,
        "answer_text": answer_text,
        "has_thinking_block": has_thinking_block,
        "stages": stages,
        "token_trace": token_trace,
        "thinking_tokens_count": thinking_tokens_count,
        "answer_tokens_count": answer_tokens_count,
        "final_answer": final_answer,
        "total_tokens": len(generated_ids),
        "stopped_naturally": stopped_naturally,
    }


@torch.no_grad()
def run_structured_explanation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    original_response: str = "",
    max_new_tokens: int = 180,
    temperature: float = 0.2,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Phase 2: Structured Explanation
    Prompts the model for concise steps, assumptions, claims, and evidence.
    Clearly labels the result as a model-generated self-explanation, NOT a verified internal reasoning trace.
    Separates the explanatory rationale from the final answer.
    """
    system_instruction = (
        "You are participating in a mechanistic interpretability protocol. "
        "Explain how the solution is derived using strictly structured sections. "
        "Do not provide conversational filler. Format your entire answer using exactly these headings:\n\n"
        "[STEPS]\n"
        "1. First step...\n"
        "2. Second step...\n\n"
        "[ASSUMPTIONS]\n"
        "- Assumption 1...\n"
        "- Assumption 2...\n\n"
        "[CLAIMS]\n"
        "- Claim 1...\n"
        "- Claim 2...\n\n"
        "[EVIDENCE]\n"
        "- Evidence / calculation for claim 1...\n"
        "- Evidence / calculation for claim 2...\n\n"
        "[FINAL ANSWER]\n"
        "State the final answer concisely here."
    )

    user_query = f"Problem: {prompt}"
    if original_response and original_response.strip():
        user_query += f"\nTarget Answer: {original_response.strip()}"
    user_query += "\n\nProvide the structured explanation with [STEPS], [ASSUMPTIONS], [CLAIMS], [EVIDENCE], and [FINAL ANSWER]."

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_query}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    eos_ids = {tokenizer.eos_token_id, 151645, 151643}
    eos_ids.discard(None)

    generated_ids = []
    stopped_naturally = False

    for _ in range(max_new_tokens):
        outputs = model(input_ids=input_ids, return_dict=True)
        logits = outputs.logits[0, -1, :].clone().to(torch.float32)
        scaled = logits / temperature if temperature > 0 else logits
        next_id = torch.argmax(scaled, dim=-1).item()

        if next_id in eos_ids:
            stopped_naturally = True
            break

        generated_ids.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Section parsing logic using regex and heading splitting
    sections = {
        "steps": [],
        "assumptions": [],
        "claims": [],
        "evidence": [],
        "final_answer": ""
    }

    current_section = None
    lines = raw_text.split("\n")
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        upper = stripped.upper()
        if "[STEPS]" in upper or upper.startswith("### STEPS") or upper.startswith("STEPS:"):
            current_section = "steps"
            continue
        elif "[ASSUMPTIONS]" in upper or upper.startswith("### ASSUMPTIONS") or upper.startswith("ASSUMPTIONS:"):
            current_section = "assumptions"
            continue
        elif "[CLAIMS]" in upper or upper.startswith("### CLAIMS") or upper.startswith("CLAIMS:"):
            current_section = "claims"
            continue
        elif "[EVIDENCE]" in upper or upper.startswith("### EVIDENCE") or upper.startswith("EVIDENCE:"):
            current_section = "evidence"
            continue
        elif "[FINAL ANSWER]" in upper or upper.startswith("### FINAL ANSWER") or upper.startswith("FINAL ANSWER:"):
            current_section = "final_answer"
            continue
            
        if current_section == "final_answer":
            if sections["final_answer"]:
                sections["final_answer"] += " " + stripped
            else:
                sections["final_answer"] = stripped
        elif current_section in ["steps", "assumptions", "claims", "evidence"]:
            clean_item = re.sub(r"^(\d+[\.\)]|\-|\*)\s*", "", stripped).strip()
            if clean_item:
                sections[current_section].append(clean_item)

    # Fallback if strict headers weren't perfectly emitted:
    if not sections["steps"] and not sections["claims"]:
        for l in lines:
            s = l.strip()
            if re.match(r"^\d+[\.\)]", s):
                sections["steps"].append(s)
            elif re.match(r"^(\-|\*)\s*(assume|given|let)", s, re.I):
                sections["assumptions"].append(s)
            elif re.match(r"^(final answer|therefore|thus|hence):?", s, re.I):
                sections["final_answer"] = s

    if not sections["final_answer"]:
        if original_response:
            sections["final_answer"] = original_response
        elif lines:
            sections["final_answer"] = lines[-1].strip()

    disclaimer = (
        "MODEL-GENERATED SELF-EXPLANATION: This text is a post-hoc rationalization generated by the model. "
        "It is NOT verified evidence of internal circuit execution, attention weights, or mechanistic causality. "
        "Treat each step, assumption, claim, and evidence piece strictly as a causal hypothesis to be tested."
    )

    success_criteria_met = bool(
        len(sections["steps"]) > 0 or len(sections["claims"]) > 0
    ) and bool(sections["final_answer"])

    return {
        "raw_explanation": raw_text,
        "steps": sections["steps"],
        "assumptions": sections["assumptions"],
        "claims": sections["claims"],
        "evidence": sections["evidence"],
        "final_answer": sections["final_answer"],
        "disclaimer": disclaimer,
        "is_verified_internal_trace": False,
        "success_criteria_met": success_criteria_met,
        "stopped_naturally": stopped_naturally,
        "tokens_generated": len(generated_ids),
        "prompt_used": formatted
    }

