"""
LLM X-Ray Laboratory & Claim Verifier Page.
Provides:
- 11-Tab Deep Mechanistic Interpretability
- Internal Activations, Attention Heatmaps, Logits, PCA
- Phase 1, Phase 2, and Phase 3 Claim Verification Pipeline
"""
import streamlit as st
import json
import torch
import pandas as pd
from typing import Dict, Any

import database
from auth.guards import require_login
from ui.theme import apply_theme
from ui.sidebar import render_sidebar
from tokenizer import load_tokenizer, get_token_details
from model import (
    load_model,
    inspect_inference_internals,
    generate_step_by_step,
    analyze_computational_process,
    run_structured_explanation
)
from experiment_tracker import capture_output_run
from visualization import (
    plot_embeddings_pca,
    plot_attention_heatmap,
    plot_hidden_states_pca,
    plot_logits_bar_chart,
    plot_token_generation_timeline,
)
import claim_verifier

# Page Setup
st.set_page_config(
    page_title="LLM X-Ray Lab — SURESH.GPT Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

database.init_db()

# Enforce login guard
user = require_login()

# Apply user theme
active_theme = st.session_state.get("theme", user.get("theme", "dark"))
apply_theme(active_theme)

# Render Sidebar
render_sidebar(user, active_conv_id=st.session_state.get("active_conv_id"), current_page="xray_lab")


# Load Cached Model & Tokenizer
has_cuda = torch.cuda.is_available()
model_choice = "Qwen/Qwen2.5-0.5B-Instruct" if not has_cuda else "Qwen/Qwen2.5-1.5B-Instruct"

@st.cache_resource(show_spinner="Loading Hugging Face Tokenizer...")
def get_cached_tokenizer(name: str):
    return load_tokenizer(name)

@st.cache_resource(show_spinner="Loading Local LLM...")
def get_cached_model(name: str):
    return load_model(name)

try:
    with st.spinner("Initializing LLM tokenizer..."):
        tokenizer = get_cached_tokenizer(model_choice)
    with st.spinner("Initializing LLM weights..."):
        loaded_model, device_str = get_cached_model(model_choice)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


def render_xray_lab():
    # Top navigation bar
    col_back, col_actions = st.columns([2.5, 7.5])
    with col_back:
        if st.button("← Back to Chat Workspace", key="xray_back_btn", type="primary", use_container_width=True):
            try:
                st.switch_page("pages/workspace.py")
            except Exception:
                st.rerun()

    st.markdown("""
    <div style="background: #1e1f25; border: 1px solid #2e3038; border-radius: 16px; padding: 20px 26px; margin: 12px 0 20px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
            <div>
                <h1 style="font-size: 2.1rem; font-weight: 800; color: #ececec; margin: 0; letter-spacing: -0.02em;">
                    🔬 LLM X-RAY LAB
                </h1>
                <div style="font-size: 0.9rem; color: #8e8ea0; margin-top: 4px;">
                    Deep Mechanistic Interpretability • Internal Activation Inspection • Phases 1, 2, and 3 Verification
                </div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                <span style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 9999px; padding: 4px 12px; font-size: 0.75rem; color: #6ee7b7; font-weight: 600;">🧬 Embeddings PCA</span>
                <span style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 9999px; padding: 4px 12px; font-size: 0.75rem; color: #6ee7b7; font-weight: 600;">🎯 Attention Heatmap</span>
                <span style="background: rgba(234, 179, 8, 0.12); border: 1px solid rgba(234, 179, 8, 0.3); border-radius: 9999px; padding: 4px 12px; font-size: 0.75rem; color: #fde047; font-weight: 600;">🔍 Phase 3 Verifier</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.form(key="lab_prompt_form_page", clear_on_submit=False):
        col_inp1, col_inp2 = st.columns([3, 1])
        with col_inp1:
            prompt = st.text_input("Enter prompt for mechanistic inspection:", value="What is AI?", key="lab_prompt_input_page")
        with col_inp2:
            max_lab_tokens = st.slider("Budget Tokens", min_value=5, max_value=60, value=25, key="lab_max_tokens_page")
        run_lab_btn = st.form_submit_button("🚀 Run Deep Lab Forward Pass", type="primary", use_container_width=True)

    if run_lab_btn or "lab_results" not in st.session_state:
        with st.spinner("Executing Transformer forward pass and capturing all intermediate tensors..."):
            tok_details = get_token_details(prompt, tokenizer, use_chat_template=True)
            internals = inspect_inference_internals(loaded_model, tokenizer, prompt, device=device_str, use_chat_template=True)
            generated_text, step_history, usage_stats = generate_step_by_step(
                model=loaded_model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_lab_tokens,
                temperature=0.0,
                top_k=50,
                top_p=0.9,
                use_chat_template=True,
                device=device_str
            )
            process_audit = analyze_computational_process(prompt, generated_text, step_history, usage_stats)
            phase1_capture = capture_output_run(
                prompt=prompt,
                response=generated_text,
                model_name=model_choice,
                model=loaded_model,
                formatted_prompt=tok_details.get("formatted_text"),
                max_new_tokens=max_lab_tokens,
                prompt_tokens=tok_details.get("total_tokens", len(tok_details.get("tokens", []))),
                generated_tokens=len(step_history),
                temperature=0.0,
                top_k=50,
                top_p=0.9,
                use_chat_template=True,
                usage_stats=usage_stats,
                step_history=step_history,
                device=device_str,
                save_to_disk=True
            )
            st.session_state["lab_results"] = {
                "prompt": prompt,
                "tok_details": tok_details,
                "internals": internals,
                "generated_text": generated_text,
                "step_history": step_history,
                "usage_stats": usage_stats,
                "process_audit": process_audit,
                "phase1_capture": phase1_capture
            }

    res = st.session_state["lab_results"]
    tok_details = res["tok_details"]
    internals = res["internals"]
    generated_text = res["generated_text"]
    step_history = res["step_history"]
    process_audit = res["process_audit"]
    phase1_capture = res["phase1_capture"]

    tab_overview, tab_tokens, tab_embeddings, tab_layers, tab_attention, tab_hidden, tab_logits, tab_generation, tab_phase1, tab_phase2, tab_phase3 = st.tabs([
        "💬 Model Overview",
        "🔤 Tokenization",
        "🧬 Embeddings",
        "📐 Layers",
        "🔍 Attention",
        "🧠 Hidden States",
        "📊 Logits",
        "⏩ Generation Flow",
        "📁 Phase 1: Output Capture",
        "📑 Phase 2: Explanation",
        "🔍 Phase 3: Claim Verification"
    ])

    with tab_overview:
        col_ov1, col_ov2 = st.columns([3, 2])
        with col_ov1:
            st.markdown("**Prompt:**")
            st.info(res["prompt"])
            st.markdown("**Generated Response:**")
            st.success(generated_text)
        with col_ov2:
            st.markdown("### 📋 Inference Summary")
            st.markdown(f"""
            - **Model:** `{model_choice}`
            - **Layers:** `{internals['num_layers']}`
            - **Attention Heads:** `{internals['num_heads']}`
            - **Embedding Dimension:** `{internals['hidden_dim']}`
            - **Tokens Generated:** `{len(step_history)}`
            """)

    with tab_tokens:
        st.markdown(f"**Token Sequence:** `{tok_details['display_tokens_str']}`")
        st.dataframe(pd.DataFrame(tok_details.get("token_table", [])), use_container_width=True)

    with tab_embeddings:
        fig_pca = plot_embeddings_pca(internals["input_embeddings"], tok_details["tokens"], tok_details["token_ids_list"])
        st.plotly_chart(fig_pca, use_container_width=True)

    with tab_layers:
        st.metric("Total Layers", internals["num_layers"])
        st.metric("Hidden Size", internals["hidden_dim"])

    with tab_attention:
        attns = internals.get("attentions", [])
        if attns:
            sel_l = st.slider("Layer", 0, len(attns) - 1, 0, key="page_lab_sel_l")
            sel_h = st.slider("Head", 0, attns[0].shape[1] - 1, 0, key="page_lab_sel_h")
            fig_attn = plot_attention_heatmap(attns[sel_l][0, sel_h].numpy(), tok_details["tokens"], sel_l, sel_h)
            st.plotly_chart(fig_attn, use_container_width=True)

    with tab_hidden:
        hids = internals.get("hidden_states", [])
        if hids:
            fig_hid = plot_hidden_states_pca(hids, tok_details["tokens"], [0, len(hids)//2, len(hids)-1])
            st.plotly_chart(fig_hid, use_container_width=True)

    with tab_logits:
        top_preds = internals.get("top_predictions", [])
        if top_preds:
            st.plotly_chart(plot_logits_bar_chart(top_preds), use_container_width=True)
            st.dataframe(pd.DataFrame(top_preds), use_container_width=True)

    with tab_generation:
        st.plotly_chart(plot_token_generation_timeline(step_history), use_container_width=True)

    with tab_phase1:
        st.markdown("### 📁 Phase 1: Complete Output Capture")
        st.json(phase1_capture)

    with tab_phase2:
        st.markdown("### 📑 Phase 2: Structured Explanation Interrogation")
        if st.button("Interrogate Model for Structured Explanation", key="page_p2_interrogate_btn"):
            with st.spinner("Extracting concise steps, assumptions, claims and evidence..."):
                p2_res = run_structured_explanation(loaded_model, tokenizer, res["prompt"], generated_text, device=device_str)
                st.session_state["p2_res"] = p2_res
        if "p2_res" in st.session_state:
            st.json(st.session_state["p2_res"])

    with tab_phase3:
        st.markdown("### 🔍 Phase 3: Claim Verification Pipeline")
        st.caption(
            "Extracts factual claims from the original answer and evaluates them against "
            "deterministic calculation engines, curated knowledge bases, and encyclopedic retrieval. "
            "Zero citations are fabricated; model agreement is not counted as independent evidence."
        )

        if st.button("🚀 Verify Output Claims", key="page_p3_run_verify_btn", type="primary"):
            with st.spinner("Extracting and verifying claims..."):
                p3_res = claim_verifier.execute_phase3_verification(
                    original_answer=generated_text,
                    original_run_id=phase1_capture["run_id"],
                    prompt=res["prompt"],
                    is_truncated=phase1_capture["response"]["is_truncated"]
                )
                st.session_state["p3_res"] = p3_res

        if "p3_res" in st.session_state:
            p3_data = st.session_state["p3_res"]
            summary = p3_data["status_summary"]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Supported", summary.get("SUPPORTED", 0))
            c2.metric("Contradicted", summary.get("CONTRADICTED", 0))
            c3.metric("Partially Supported", summary.get("PARTIALLY_SUPPORTED", 0))
            c4.metric("Insufficient Evidence", summary.get("INSUFFICIENT_EVIDENCE", 0))
            c5.metric("Not Verifiable", summary.get("NOT_VERIFIABLE", 0))

            st.markdown("---")
            st.markdown("#### Extracted Claims Breakdown")
            for c in p3_data["claims"]:
                status = c["verification_status"]
                badge_color = {
                    "SUPPORTED": "#22c55e",
                    "CONTRADICTED": "#ef4444",
                    "PARTIALLY_SUPPORTED": "#f59e0b",
                    "INSUFFICIENT_EVIDENCE": "#64748b",
                    "NOT_VERIFIABLE": "#8b5cf6",
                    "PENDING": "#94a3b8"
                }.get(status, "#94a3b8")

                with st.expander(f"[{status}] {c['claim_id']}: {c['exact_claim_text']}", expanded=True):
                    st.markdown(f"""
                    <div style="background-color:#0f172a; border-left:5px solid {badge_color}; border:1px solid #334155; border-left-width:5px; border-radius:6px; padding:12px;">
                        <span style="font-weight:700; color:#38bdf8;">Type:</span> {c['claim_type']} &nbsp;|&nbsp;
                        <span style="font-weight:700; color:{badge_color};">Status: {status}</span><br>
                        <div style="margin-top:6px;"><b>Explanation:</b> {c['explanation']}</div>
                        <div style="margin-top:4px; font-size:0.85rem; color:#94a3b8;">
                            <b>Method:</b> {c['verification_method']}<br>
                            <b>Source:</b> {c['source_title']} {f"(<a href='{c['source_url']}' target='_blank' style='color:#38bdf8;'>link</a>)" if c.get('source_url') else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.download_button(
                "📥 Download Phase 3 Verification Report (JSON)",
                data=json.dumps(p3_data, indent=2, ensure_ascii=False),
                file_name=f"phase3_{p3_data['run_id']}.json",
                mime="application/json"
            )


# Execute
render_xray_lab()

