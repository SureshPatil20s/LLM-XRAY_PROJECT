"""
Phase 1: Complete Output Capture
Mechanistic Investigation Research Protocol module for LLM-XRay.

Captures, validates, and persists:
- Full prompt (raw and chat-template formatted)
- Model response
- Model name, architecture version, device, dtype
- Token limits (max_new_tokens, prompt tokens, generated tokens, total tokens)
- Generation hyperparameters (temperature, top_k, top_p, sampling strategy)
- Exact timestamp (ISO-8601 with timezone)
- Truncation status (boolean is_truncated and detailed reason)
"""
import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")


def get_model_metadata(model: Any, model_name: str, device: str = "cpu") -> Dict[str, Any]:
    """
    Extracts comprehensive model architecture details, version, and configuration.
    """
    config = getattr(model, "config", None)
    
    num_layers = getattr(config, "num_hidden_layers", None)
    num_heads = getattr(config, "num_attention_heads", None)
    hidden_dim = getattr(config, "hidden_size", None)
    vocab_size = getattr(config, "vocab_size", None)
    torch_dtype = str(getattr(model, "dtype", "unknown"))
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else 0
    params_readable = f"{total_params / 1e9:.2f}B" if total_params >= 1e9 else f"{total_params / 1e6:.1f}M"

    return {
        "model_name": model_name,
        "model_type": getattr(config, "model_type", "transformer"),
        "transformers_version": getattr(config, "transformers_version", "unknown"),
        "total_parameters": total_params,
        "total_parameters_readable": params_readable,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "device": device,
        "dtype": torch_dtype,
    }


def capture_output_run(
    prompt: str,
    response: str,
    model_name: str,
    model: Optional[Any] = None,
    formatted_prompt: Optional[str] = None,
    max_new_tokens: int = 15,
    prompt_tokens: int = 0,
    generated_tokens: int = 0,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    use_chat_template: bool = True,
    usage_stats: Optional[Dict[str, Any]] = None,
    step_history: Optional[List[Dict[str, Any]]] = None,
    device: str = "cpu",
    save_to_disk: bool = True,
    custom_notes: str = ""
) -> Dict[str, Any]:
    """
    Executes Phase 1: Complete output capture.
    Records full metadata, prompt, response, settings, timestamp, and truncation status.
    """
    # 1. Timestamps
    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now()
    timestamp_iso = now_local.astimezone().isoformat()
    timestamp_compact = now_local.strftime("%Y%m%d_%H%M%S")

    # 2. Determine truncation status
    # A generation is truncated if it reached the maximum allocated token budget
    # without emitting a natural End-Of-Sequence (EOS) token.
    stopped_early = False
    stop_reason = ""
    if usage_stats:
        stopped_early = usage_stats.get("stopped_early", False)
        stop_reason = usage_stats.get("stop_reason", "")
    elif generated_tokens < max_new_tokens:
        stopped_early = True
        stop_reason = "Completed before budget"

    is_truncated = not stopped_early and (generated_tokens >= max_new_tokens)
    
    if is_truncated:
        truncation_status = "TRUNCATED"
        truncation_reason = (
            f"Response was cut off because generation hit the allocated token limit of {max_new_tokens} tokens "
            f"before reaching an EOS token."
        )
    else:
        truncation_status = "COMPLETE"
        truncation_reason = (
            f"Response completed naturally (emitted EOS token after {generated_tokens} tokens, "
            f"within budget of {max_new_tokens})."
        )

    # 3. Model Metadata
    if model is not None:
        model_info = get_model_metadata(model, model_name, device=device)
    else:
        model_info = {
            "model_name": model_name,
            "device": device
        }

    # 4. Generate unique Run ID
    run_uuid = uuid.uuid4().hex[:8]
    run_id = f"phase1_{timestamp_compact}_{run_uuid}"

    # 5. Build canonical record
    record = {
        "run_id": run_id,
        "protocol_phase": "Phase 1: Complete Output Capture",
        "timestamp": {
            "iso": timestamp_iso,
            "utc": now_utc.isoformat(),
            "local_readable": now_local.strftime("%Y-%m-%d %H:%M:%S")
        },
        "model": model_info,
        "prompt": {
            "raw": prompt,
            "formatted": formatted_prompt if formatted_prompt else prompt,
            "token_count": prompt_tokens,
        },
        "response": {
            "text": response,
            "token_count": generated_tokens,
            "is_truncated": is_truncated,
            "truncation_status": truncation_status,
            "truncation_reason": truncation_reason,
            "stop_reason": stop_reason or ("Hit token budget" if is_truncated else "Natural completion")
        },
        "token_limits": {
            "max_new_tokens_budget": max_new_tokens,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": prompt_tokens + generated_tokens,
            "tokens_saved": max(0, max_new_tokens - generated_tokens),
            "budget_exhausted": is_truncated
        },
        "generation_settings": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "use_chat_template": use_chat_template,
            "sampling_method": "Greedy (Argmax)" if temperature == 0 else f"Top-K ({top_k}) + Top-P ({top_p})"
        },
        "custom_notes": custom_notes,
        "token_trace_summary": [
            {
                "step": s.get("step"),
                "token_id": s.get("token_id"),
                "token_str": s.get("token_str"),
                "prob": round(s.get("prob", 0.0), 4),
                "percentage": s.get("percentage")
            }
            for s in (step_history or [])
        ]
    }

    # 6. Save to disk if requested
    saved_path = None
    if save_to_disk:
        saved_path = save_capture_to_disk(record)
        record["saved_file_path"] = saved_path

    return record


def save_capture_to_disk(record: Dict[str, Any], directory: Optional[str] = None) -> str:
    """
    Saves the output capture record as a formatted JSON artifact.
    """
    target_dir = directory or EXPERIMENTS_DIR
    os.makedirs(target_dir, exist_ok=True)
    
    filename = f"{record['run_id']}.json"
    file_path = os.path.join(target_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
        
    return file_path


def list_saved_captures(directory: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves and summarizes all saved Phase 1 captures sorted by timestamp (newest first).
    """
    target_dir = directory or EXPERIMENTS_DIR
    if not os.path.exists(target_dir):
        return []

    files = [f for f in os.listdir(target_dir) if f.startswith("phase1_") and f.endswith(".json")]
    records = []
    
    for fname in sorted(files, reverse=True):
        fpath = os.path.join(target_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                records.append({
                    "run_id": data.get("run_id"),
                    "timestamp": data.get("timestamp", {}).get("local_readable", "Unknown"),
                    "prompt": data.get("prompt", {}).get("raw", "")[:60] + "...",
                    "response": data.get("response", {}).get("text", "")[:60] + "...",
                    "is_truncated": data.get("response", {}).get("is_truncated", False),
                    "truncation_status": data.get("response", {}).get("truncation_status", "UNKNOWN"),
                    "tokens": f"{data.get('token_limits', {}).get('generated_tokens', 0)}/{data.get('token_limits', {}).get('max_new_tokens_budget', 0)}",
                    "file_path": fpath,
                    "full_data": data
                })
        except Exception:
            continue
            
    return records


def capture_phase2_explanation(
    run_id: str,
    prompt: str,
    original_response: str,
    explanation: Dict[str, Any],
    model_name: str = "",
    save_to_disk: bool = True,
    directory: Optional[str] = None
) -> Dict[str, Any]:
    """
    Persists Phase 2 Structured Explanation record.
    Clearly marks the explanation as a model-generated hypothesis (unverified trace).
    """
    now_local = datetime.now()
    record = {
        "run_id": run_id,
        "phase": "Phase 2: Structured Explanation",
        "timestamp": {
            "iso": now_local.astimezone().isoformat(),
            "local_readable": now_local.strftime("%Y-%m-%d %H:%M:%S")
        },
        "model_name": model_name,
        "problem_prompt": prompt,
        "target_response": original_response,
        "model_generated_explanation": {
            "steps": explanation.get("steps", []),
            "assumptions": explanation.get("assumptions", []),
            "claims": explanation.get("claims", []),
            "evidence": explanation.get("evidence", []),
            "final_answer": explanation.get("final_answer", ""),
            "raw_text": explanation.get("raw_explanation", "")
        },
        "scientific_classification": {
            "is_verified_internal_trace": False,
            "disclaimer": explanation.get("disclaimer", ""),
            "success_criteria_met": explanation.get("success_criteria_met", False),
            "epistemic_status": "UNVERIFIED_HYPOTHESIS"
        },
        "generation_meta": {
            "tokens_generated": explanation.get("tokens_generated", 0),
            "stopped_naturally": explanation.get("stopped_naturally", False)
        }
    }

    if save_to_disk:
        target_dir = directory or EXPERIMENTS_DIR
        os.makedirs(target_dir, exist_ok=True)
        filename = f"phase2_{run_id}.json"
        fpath = os.path.join(target_dir, filename)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        record["saved_file_path"] = fpath

    return record


def list_phase2_explanations(directory: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves all saved Phase 2 explanation records.
    """
    target_dir = directory or EXPERIMENTS_DIR
    if not os.path.exists(target_dir):
        return []

    files = [f for f in os.listdir(target_dir) if f.startswith("phase2_") and f.endswith(".json")]
    records = []
    for fname in sorted(files, reverse=True):
        fpath = os.path.join(target_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                records.append({
                    "run_id": data.get("run_id"),
                    "timestamp": data.get("timestamp", {}).get("local_readable", "Unknown"),
                    "prompt": data.get("problem_prompt", "")[:60] + "...",
                    "final_answer": data.get("model_generated_explanation", {}).get("final_answer", ""),
                    "steps_count": len(data.get("model_generated_explanation", {}).get("steps", [])),
                    "claims_count": len(data.get("model_generated_explanation", {}).get("claims", [])),
                    "file_path": fpath,
                    "full_data": data
                })
        except Exception:
            continue
    return records

