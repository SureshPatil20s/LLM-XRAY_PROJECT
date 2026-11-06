"""
UI Components for SURESH.GPT Platform / LocalGPT Pro:
- Welcome greeting & 4 suggested prompt cards (ChatGPT Platform style)
- Source attribution chips
- Token probability chips
- Loading skeleton placeholder
"""
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional


def get_time_based_greeting(name: str = "User") -> str:
    """Returns dynamic time-appropriate greeting."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        period = "Good morning"
    elif 12 <= hour < 17:
        period = "Good afternoon"
    else:
        period = "Good evening"
    return f"{period}, {name}"


def render_welcome_screen(display_name: str = "User") -> Optional[str]:
    """
    Renders ChatGPT-style welcome hero with 4 clean suggested prompt cards.
    Returns clicked prompt text if any, or None.
    """
    greeting = get_time_based_greeting(display_name)

    st.markdown(f"""
    <div style="text-align: center; padding: 40px 10px 24px 10px;">
        <div style="display:inline-flex; align-items:center; justify-content:center; width:48px; height:48px; border-radius:12px; background:#10a37f; margin-bottom:14px; font-size:24px;">
            🤖
        </div>
        <h1 style="font-size: 2.1rem; font-weight: 700; color: #ececec; margin-bottom: 6px; letter-spacing: -0.02em;">
            Hello! How can I assist you today?
        </h1>
        <p style="font-size: 0.95rem; color: #8e8ea0; max-width: 580px; margin: 0 auto;">
            Ask questions, analyze private documents with local RAG, or inspect internal token activations with the LLM X-Ray engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

    prompts = [
        ("🧠 Explain Attention", "How do multi-head attention mechanisms work in Transformers?", "Explain how multi-head attention calculates Query, Key, and Value weights in Transformer models."),
        ("🔬 Inspect Mechanistics", "Inspect internal logits and token probabilities", "What happens inside a language model during next-token prediction? Explain logits and softmax."),
        ("📄 Document Synthesis", "Summarize key findings from research papers", "Summarize the key differences between dense embeddings and sparse keyword retrieval."),
        ("⚡ Python Optimization", "Write performant asynchronous Python code", "Show an example of high-performance concurrent processing in Python using asyncio.")
    ]

    cols = st.columns(2)
    selected_prompt = None

    for idx, (title, desc, full_prompt) in enumerate(prompts):
        col = cols[idx % 2]
        with col:
            st.markdown(f"""
            <div style="background: #1e1f25; border: 1px solid #2e3038; border-radius: 12px; padding: 14px 16px; margin-bottom: 8px;">
                <div style="font-weight: 600; font-size: 0.9rem; color: #f4f4f5; margin-bottom: 3px;">{title}</div>
                <div style="font-size: 0.78rem; color: #8e8ea0; line-height: 1.35;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Use Prompt", key=f"welcome_prompt_{idx}", use_container_width=True):
                selected_prompt = full_prompt

    return selected_prompt


def render_source_chips(sources: List[Dict[str, Any]]) -> None:
    """Renders expandable reference source provenance cards."""
    if not sources:
        return

    st.markdown("""
    <div style="margin-top: 10px; margin-bottom: 6px; font-size: 0.78rem; font-weight: 600; color: #10b981; display: flex; align-items: center; gap: 6px;">
        📚 <span>Verified Reference Sources</span>
    </div>
    """, unsafe_allow_html=True)

    chips_html = '<div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">'
    for s in sources:
        source_name = s.get("source", "Document")
        page_num = s.get("page", 1)
        chips_html += f'<span style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 9999px; padding: 2px 10px; font-size: 0.75rem; color: #6ee7b7;">📄 {source_name} &bull; p. {page_num}</span>'
    chips_html += '</div>'
    st.markdown(chips_html, unsafe_allow_html=True)

    with st.expander("🔍 View extracted source excerpts", expanded=False):
        for s in sources:
            st.markdown(f"**📄 {s.get('source')} (Page {s.get('page')})**")
            st.caption(f'"{s.get("snippet", "")}"')


def render_skeleton() -> None:
    """Renders a skeleton placeholder while weights or search are loading."""
    st.markdown("""
    <div style="background: #23252b; border-radius: 6px; height: 16px; width: 80%; margin-bottom: 8px;"></div>
    <div style="background: #23252b; border-radius: 6px; height: 16px; width: 60%; margin-bottom: 8px;"></div>
    <div style="background: #23252b; border-radius: 6px; height: 16px; width: 90%; margin-bottom: 8px;"></div>
    """, unsafe_allow_html=True)
