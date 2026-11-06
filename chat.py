"""
Chat orchestration module for LocalGPT Pro.
Handles:
1. Multi-turn conversation context building
2. RAG retrieval integration with per-user document isolation
3. Real-time streaming generation using TextIteratorStreamer
4. X-Ray capture integration
5. Message regeneration and editing
"""
from typing import Dict, Any, List, Optional, Generator, Tuple
from threading import Thread
import torch
from transformers import TextIteratorStreamer

from memory import ConversationMemory
import rag
from xray import capture_xray


def stream_chat_response(
    model: Any,
    tokenizer: Any,
    memory: ConversationMemory,
    user_query: str,
    system_prompt: str = "You are a helpful AI assistant. Explain concepts clearly with examples.",
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    use_rag: bool = True,
    device: str = "cpu",
    uid: Optional[str] = None
) -> Tuple[Generator[str, None, None], List[Dict[str, Any]]]:
    """
    Sets up streaming response generation for the user query.
    1. Optionally retrieves document context via RAG for the specific user.
    2. Builds formatted conversation context.
    3. Spawns generation in a background thread with TextIteratorStreamer.
    4. Yields streamed tokens in real time.
    Returns:
    (generator, sources_list)
    """
    # 1. RAG retrieval if enabled
    rag_context = ""
    sources_list: List[Dict[str, Any]] = []
    if use_rag:
        try:
            rag_context, sources_list = rag.retrieve_context(user_query, top_k=4, uid=uid)
        except Exception as e:
            print(f"[RAG Warning] Retrieval error: {e}")

    # 2. Add user message to memory
    memory.add_user_message(user_query)

    # 3. Build prompt messages using memory
    messages = memory.build_prompt_context(
        system_prompt=system_prompt,
        rag_context=rag_context,
        max_history_turns=8
    )

    # Apply tokenizer chat template
    if hasattr(tokenizer, "apply_chat_template"):
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback manual formatting
        prompt_parts = []
        for m in messages:
            prompt_parts.append(f"{m['role'].capitalize()}: {m['content']}")
        prompt_parts.append("Assistant:")
        formatted_prompt = "\n\n".join(prompt_parts)

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)

    # 4. Set up TextIteratorStreamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else 1.0,
        top_k=top_k if temperature > 0 else 1,
        top_p=top_p if temperature > 0 else 1.0,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.daemon = True
    thread.start()

    def token_generator() -> Generator[str, None, None]:
        for token_text in streamer:
            yield token_text

    return token_generator(), sources_list


def generate_single_turn_with_xray(
    model: Any,
    tokenizer: Any,
    prompt: str,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Runs a single forward pass and returns complete X-Ray inspection data."""
    return capture_xray(model, tokenizer, prompt, device=device)
