"""
LLM X-Ray Inspection Panel for LocalGPT.
Provides deep mechanistic interpretability inspection across 6 tabs:
1. Tokens (token pills, token table, token IDs, byte lengths)
2. Embeddings (vector statistics, 2D PCA projection)
3. Layers (transformer layer structure and hidden dimension)
4. Attention (interactive attention heatmap by layer and head)
5. Hidden States (hidden state progression across depth)
6. Logits & Probabilities (next token candidates bar chart and probability breakdown)
"""
from typing import Dict, Any, List, Optional
import streamlit as st
import torch
import pandas as pd

from model import inspect_inference_internals
from tokenizer import get_token_details
from visualization import (
    plot_embeddings_pca,
    plot_vector_profile,
    plot_attention_heatmap,
    plot_hidden_states_pca,
    plot_logits_bar_chart
)


def capture_xray(
    model: Any,
    tokenizer: Any,
    input_text: str,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes a forward pass with output_attentions=True and output_hidden_states=True,
    capturing all internal activations.
    """
    token_info = get_token_details(input_text, tokenizer, use_chat_template=False)
    internals = inspect_inference_internals(
        model=model,
        tokenizer=tokenizer,
        prompt=input_text,
        device=device,
        use_chat_template=False
    )
    return {
        "token_info": token_info,
        "internals": internals
    }


def render_xray_panel(xray_data: Dict[str, Any], key_prefix: str = "xray") -> None:
    """
    Renders the full 6-tab LLM X-Ray interface directly into Streamlit.
    """
    token_info = xray_data.get("token_info", {})
    internals = xray_data.get("internals", {})

    if not internals:
        st.warning("⚠️ No X-Ray data available for this generation.")
        return

    st.markdown("### 🔬 LLM X-Ray Inspection")
    st.caption("Deep internal inspection of embeddings, attention mechanisms, hidden states, and next-token logits.")

    tab_tok, tab_emb, tab_lay, tab_attn, tab_hid, tab_log = st.tabs([
        "🔤 Tokens",
        "🧬 Embeddings",
        "🧱 Layers",
        "🎯 Attention",
        "🌊 Hidden States",
        "📊 Logits & Probabilities"
    ])

    tokens = token_info.get("tokens", [])
    token_ids = token_info.get("token_ids_list", [])

    # ---------------- 1. TOKENS TAB ----------------
    with tab_tok:
        st.markdown("#### Token Breakdown")
        display_str = token_info.get("display_tokens_str", "")
        if display_str:
            st.info(f"**Token Sequence:**\n\n`{display_str}`")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Tokens", token_info.get("total_tokens", 0))
        col_m2.metric("Total Characters", token_info.get("total_chars", 0))
        comp_ratio = token_info.get("compression_ratio", 1.0)
        col_m3.metric("Chars per Token", f"{comp_ratio:.2f}")

        token_table = token_info.get("token_table", [])
        if token_table:
            st.markdown("##### Token Inspection Table")
            df_tok = pd.DataFrame(token_table)
            st.dataframe(df_tok, use_container_width=True)

    # ---------------- 2. EMBEDDINGS TAB ----------------
    with tab_emb:
        st.markdown("#### Input Embeddings & PCA Projection")
        embeddings = internals.get("input_embeddings")

        if embeddings is not None:
            emb_shape = list(embeddings.shape)
            col_e1, col_e2, col_e3 = st.columns(3)
            col_e1.metric("Sequence Length", emb_shape[1])
            col_e2.metric("Embedding Dimension", emb_shape[2])
            col_e3.metric("Vector Norm (Mean)", f"{embeddings.norm(dim=-1).mean().item():.3f}")

            # 2D PCA Plot
            try:
                fig_pca = plot_embeddings_pca(embeddings, tokens, token_ids)
                st.plotly_chart(fig_pca, use_container_width=True, key=f"{key_prefix}_pca")
            except Exception as e:
                st.error(f"Could not compute PCA visualization: {e}")

            # Single token vector profile
            if len(tokens) > 0:
                sel_tok_idx = st.selectbox(
                    "Inspect individual token embedding vector profile",
                    options=list(range(len(tokens))),
                    format_func=lambda i: f"[{i}] {tokens[i]} (ID: {token_ids[i]})",
                    key=f"{key_prefix}_sel_tok_vec"
                )
                vec_np = embeddings[0, sel_tok_idx].numpy()
                fig_prof = plot_vector_profile(vec_np, tokens[sel_tok_idx], token_ids[sel_tok_idx])
                st.plotly_chart(fig_prof, use_container_width=True, key=f"{key_prefix}_prof")

    # ---------------- 3. LAYERS TAB ----------------
    with tab_lay:
        st.markdown("#### Transformer Architecture")
        num_layers = internals.get("num_layers", 0)
        num_heads = internals.get("num_heads", 0)
        hidden_dim = internals.get("hidden_dim", 0)
        vocab_size = internals.get("vocab_size", 0)

        col_l1, col_l2, col_l3, col_l4 = st.columns(4)
        col_l1.metric("Transformer Layers", num_layers)
        col_l2.metric("Attention Heads", num_heads)
        col_l3.metric("Hidden Size", hidden_dim)
        col_l4.metric("Vocabulary Size", f"{vocab_size:,}")

        st.markdown(
            """
            **Internal Components per Transformer Block:**
            - **Multi-Head Self-Attention (Q, K, V projections)** with RoPE positional embeddings
            - **RMSNorm Pre-normalization**
            - **SwiGLU Feed-Forward Network (MLP)**
            - **Residual Skip Connection**
            """
        )

    # ---------------- 4. ATTENTION TAB ----------------
    with tab_attn:
        st.markdown("#### Multi-Head Self-Attention Heatmap")
        attentions = internals.get("attentions", [])
        if attentions:
            num_layers = len(attentions)
            num_heads = attentions[0].shape[1] if num_layers > 0 else 1

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                sel_layer = st.slider(
                    "Layer", min_value=0, max_value=num_layers - 1, value=min(num_layers // 2, num_layers - 1),
                    key=f"{key_prefix}_sel_layer"
                )
            with col_ctrl2:
                sel_head = st.slider(
                    "Attention Head", min_value=0, max_value=num_heads - 1, value=0,
                    key=f"{key_prefix}_sel_head"
                )
            with col_ctrl3:
                use_log_scale = st.checkbox("Log scale weights", value=False, key=f"{key_prefix}_log_attn")

            attn_matrix = attentions[sel_layer][0, sel_head].numpy()
            
            try:
                fig_attn = plot_attention_heatmap(attn_matrix, tokens, sel_layer, sel_head, log_scale=use_log_scale)
                st.plotly_chart(fig_attn, use_container_width=True, key=f"{key_prefix}_fig_attn")
            except Exception as e:
                st.error(f"Error rendering attention heatmap: {e}")
        else:
            st.info("No attention matrices captured.")

    # ---------------- 5. HIDDEN STATES TAB ----------------
    with tab_hid:
        st.markdown("#### Hidden State Evolution")
        hidden_states = internals.get("hidden_states", [])

        if hidden_states:
            total_layers = len(hidden_states)
            # Default pick first, middle, last layer
            selected_indices = [0, total_layers // 2, total_layers - 1]
            try:
                fig_hid = plot_hidden_states_pca(hidden_states, tokens, selected_layers=selected_indices)
                st.plotly_chart(fig_hid, use_container_width=True, key=f"{key_prefix}_fig_hid")
            except Exception as e:
                st.error(f"Error rendering hidden states PCA: {e}")
        else:
            st.info("No hidden states captured.")

    # ---------------- 6. LOGITS & PROBABILITIES TAB ----------------
    with tab_log:
        st.markdown("#### Next Token Prediction & Probabilities")
        top_predictions = internals.get("top_predictions", [])

        if top_predictions:
            st.markdown("##### Candidate Predictions (Top Candidates)")
            try:
                fig_logits = plot_logits_bar_chart(top_predictions)
                st.plotly_chart(fig_logits, use_container_width=True, key=f"{key_prefix}_fig_logits")
            except Exception as e:
                st.error(f"Error rendering logits bar chart: {e}")

            df_preds = pd.DataFrame(top_predictions)
            display_cols = [c for c in ["token", "probability", "percentage", "logit"] if c in df_preds.columns]
            st.dataframe(df_preds[display_cols], use_container_width=True)
        else:
            st.info("No logits or candidate predictions available.")
