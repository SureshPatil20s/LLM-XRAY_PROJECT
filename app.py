"""
LLM X-RAY: Interactive Visualization of LLM Inference
Streamlit Web Application
"""
import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go

from tokenizer import load_tokenizer, get_token_details
from model import load_model, inspect_inference_internals, generate_step_by_step
from visualization import (
    plot_embeddings_pca,
    plot_vector_profile,
    plot_attention_heatmap,
    plot_hidden_states_pca,
    plot_logits_bar_chart,
    plot_token_generation_timeline,
)

# Page configuration
st.set_page_config(
    page_title="LLM X-Ray — Interactive LLM Inference",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .token-chip {
        display: inline-block;
        background: #334155;
        color: #38bdf8;
        border-radius: 6px;
        padding: 4px 10px;
        margin: 4px;
        font-family: monospace;
        font-weight: 600;
        font-size: 0.95rem;
        border: 1px solid #475569;
    }
    .step-box {
        background: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 12px;
        border-radius: 4px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Choose Model:",
    options=[
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct (Faster on CPU)"
    ],
    index=0,
    help="Qwen2.5-1.5B has 28 Transformer layers and 1536 hidden dimension."
)

actual_model_name = "Qwen/Qwen2.5-0.5B-Instruct" if "0.5B" in model_choice else "Qwen/Qwen2.5-1.5B-Instruct"

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Generation Controls")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.05,
                                help="0 = deterministic greedy search, higher = more random")
top_p = st.sidebar.slider("Top-P (Nucleus Sampling)", min_value=0.1, max_value=1.0, value=0.9, step=0.05,
                          help="Cumulative probability cutoff")
top_k = st.sidebar.slider("Top-K", min_value=0, max_value=100, value=50, step=5,
                          help="Number of highest probability vocabulary tokens to keep")
max_tokens = st.sidebar.slider("Max New Tokens", min_value=5, max_value=60, value=25, step=5,
                               help="Maximum number of tokens to generate in response")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**LLM X-Ray Architecture Flow:**
1. Text Prompt
2. Tokenizer & IDs
3. Input Embeddings
4. 28 Transformer Layers (Self-Attention & MLP)
5. Hidden States
6. LM Head Logits
7. Softmax Probabilities
8. Next Token Sampling
""")


# Cached Resource Loaders
@st.cache_resource(show_spinner="Loading Tokenizer...")
def get_cached_tokenizer(name: str):
    return load_tokenizer(name)


@st.cache_resource(show_spinner="Loading Local Model (First run downloads weights)...")
def get_cached_model(name: str):
    return load_model(name)


# Load tokenizer and model
try:
    with st.spinner("Initializing LLM tokenizer..."):
        tokenizer = get_cached_tokenizer(actual_model_name)
    with st.spinner("Initializing LLM model into memory..."):
        model, device_str = get_cached_model(actual_model_name)
    st.sidebar.success(f"✓ Model loaded on: {device_str.upper()}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# Main Header
st.markdown('<div class="main-header">🔬 LLM X-RAY</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Visualization of Internal Transformer Inference Mechanics • Powered by Hugging Face & Qwen2.5</div>', unsafe_allow_html=True)

# Prompt Input
default_prompt = "What is AI?"
prompt = st.text_input("Enter your prompt:", value=default_prompt, placeholder="Type a prompt to inspect inside the model...")

col_btn1, col_btn2 = st.columns([1, 5])
with col_btn1:
    generate_clicked = st.button("🚀 Run X-Ray & Generate", type="primary", use_container_width=True)

# Session state management for inference results
if "last_results" not in st.session_state:
    st.session_state.last_results = None

if generate_clicked or (st.session_state.last_results is None and prompt):
    with st.spinner("Executing Transformer forward pass and extracting internal tensors..."):
        # 1. Token details
        tok_details = get_token_details(prompt, tokenizer)
        
        # 2. Internal inference pass (embeddings, layers, attention, logits)
        internals = inspect_inference_internals(model, tokenizer, prompt, device=device_str)
        
        # 3. Step-by-step autoregressive generation
        generated_text, step_history = generate_step_by_step(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device_str
        )
        
        st.session_state.last_results = {
            "prompt": prompt,
            "tok_details": tok_details,
            "internals": internals,
            "generated_text": generated_text,
            "step_history": step_history
        }

res = st.session_state.last_results

if res:
    tok_details = res["tok_details"]
    internals = res["internals"]
    generated_text = res["generated_text"]
    step_history = res["step_history"]
    
    # Navigation Tabs matching the project specification
    tab_chat, tab_tokens, tab_embeddings, tab_arch, tab_attention, tab_hidden, tab_logits, tab_generation = st.tabs([
        "💬 Chat & Overview",
        "🔤 Tokenization",
        "🧬 Embeddings",
        "📐 Transformer Layers",
        "🔍 Attention Heatmap",
        "🧠 Hidden States",
        "📊 Logits & Probabilities",
        "⏩ Step-by-Step Generation"
    ])
    
    # ----------------------------------------------------
    # TAB 1: Chat & Overview
    # ----------------------------------------------------
    with tab_chat:
        st.subheader("💬 Model Response")
        
        col_chat1, col_chat2 = st.columns([3, 2])
        with col_chat1:
            st.markdown(f"**Prompt:**")
            st.info(res["prompt"])
            
            st.markdown(f"**Generated Response:**")
            st.success(generated_text if generated_text.strip() else "[No text generated]")
            
        with col_chat2:
            st.markdown("### 📋 Inference Summary")
            st.markdown(f"""
            - **Model**: `{actual_model_name}`
            - **Layers**: `{internals['num_layers']}`
            - **Attention Heads per Layer**: `{internals['num_heads']}`
            - **Embedding Dimension**: `{internals['hidden_dim']}`
            - **Prompt Length**: `{tok_details['total_tokens']}` tokens
            - **Generated Tokens**: `{len(step_history)}` tokens
            - **Vocabulary Size**: `{internals['vocab_size']:,}`
            """)
            
        st.markdown("---")
        st.markdown("### 🔄 Full Pipeline Trace")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("1. Input", f"{len(prompt)} chars")
        c2.metric("2. Tokens", f"{tok_details['total_tokens']} tokens")
        c3.metric("3. Hidden Dim", f"{internals['hidden_dim']} dims")
        c4.metric("4. Layers Passed", f"{internals['num_layers']} layers")
        c5.metric("5. Top Predicted", f"'{internals['top_predictions'][0]['clean_token']}'")

    # ----------------------------------------------------
    # TAB 2: Tokenization
    # ----------------------------------------------------
    with tab_tokens:
        st.subheader("🔤 Tokenization Breakdown")
        st.markdown("""
        Before a neural network can process text, the **Tokenizer** splits text into chunks called **Tokens** 
        and maps each token to a unique numerical **Token ID** from its vocabulary.
        """)
        
        col_tok_a, col_tok_b = st.columns([1, 1])
        with col_tok_a:
            st.markdown("**Tokens Sequence:**")
            chips_html = "".join([f'<span class="token-chip">{t}</span>' for t in tok_details["tokens"]])
            st.markdown(chips_html, unsafe_allow_html=True)
            
            st.markdown("<br>**Pipe Delimited Representation:**", unsafe_allow_html=True)
            st.code(tok_details["display_tokens_str"], language="text")
            
        with col_tok_b:
            st.markdown("**Token IDs Sequence:**")
            st.code(tok_details["display_ids_str"], language="text")
            
        st.markdown("### 📑 Detailed Token Mapping Table")
        df_tokens = pd.DataFrame(tok_details["token_table"])
        st.dataframe(df_tokens, use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: Embeddings
    # ----------------------------------------------------
    with tab_embeddings:
        st.subheader("🧬 Token Embeddings")
        st.markdown("""
        Each **Token ID** looks up a high-dimensional vector in the model's **Input Embedding Matrix**.
        This vector places the token in a continuous geometric semantic space.
        """)
        
        emb_tensor = internals["input_embeddings"] # [1, seq_len, hidden_dim]
        emb_dim = internals["hidden_dim"]
        
        c_e1, c_e2, c_e3, c_e4 = st.columns(4)
        c_e1.metric("Embedding Dimension", f"{emb_dim}")
        c_e2.metric("Sequence Length", f"{emb_tensor.shape[1]}")
        c_e3.metric("Mean Vector Norm", f"{float(torch.norm(emb_tensor[0], dim=-1).mean()):.2f}")
        c_e4.metric("Total Parameters in Matrix", f"{emb_dim * internals['vocab_size']:,}")
        
        st.markdown("### 🗺️ 2D PCA Projection of Input Embeddings")
        st.caption("Principal Component Analysis (PCA) projects the high-dimensional vectors down to 2 dimensions to visualize spatial clustering.")
        fig_pca = plot_embeddings_pca(emb_tensor, tok_details["tokens"], tok_details["token_ids_list"])
        st.plotly_chart(fig_pca, use_container_width=True)
        
        st.markdown("### 🔬 Inspect Single Token Vector")
        selected_token_idx = st.selectbox(
            "Select token to inspect vector values & statistics:",
            options=range(len(tok_details["tokens"])),
            format_func=lambda i: f"Token #{i}: '{tok_details['tokens'][i]}' (ID: {tok_details['token_ids_list'][i]})"
        )
        
        sel_vec = emb_tensor[0, selected_token_idx].cpu().numpy()
        
        # Display vector stats
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Vector Dimension", len(sel_vec))
        s2.metric("Mean Value", f"{float(np.mean(sel_vec)):.4f}")
        s3.metric("Std Dev", f"{float(np.std(sel_vec)):.4f}")
        s4.metric("Min / Max", f"{float(np.min(sel_vec)):.2f} / {float(np.max(sel_vec)):.2f}")
        s5.metric("L2 Norm", f"{float(np.linalg.norm(sel_vec)):.2f}")
        
        fig_vec = plot_vector_profile(sel_vec, tok_details["tokens"][selected_token_idx], tok_details["token_ids_list"][selected_token_idx])
        st.plotly_chart(fig_vec, use_container_width=True)

    # ----------------------------------------------------
    # TAB 4: Transformer Layers (Step 6)
    # ----------------------------------------------------
    with tab_arch:
        st.subheader("📐 Transformer Architecture Visualization")
        st.markdown("""
        The transformer consists of an **Input Embedding layer** followed by a deep stack of identical 
        **Transformer Decoder Blocks** (28 layers in Qwen2.5-1.5B), terminating in a **Final RMSNorm** and **LM Head**.
        """)
        
        col_arch_a, col_arch_b = st.columns([1, 1])
        with col_arch_a:
            st.markdown("### 🏢 Stack Architecture Diagram")
            st.markdown(f"""
            ```
            Input Tokens
                 │
                 ▼
            ┌──────────────────────────────────────────────┐
            │ [ Input Embeddings ] (Dim: {internals['hidden_dim']})             │
            └──────────────────────┬───────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────────┐
            │ Layer 1  : Self-Attention + SwiGLU MLP       │
            └──────────────────────┬───────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────────┐
            │ Layer 2  : Self-Attention + SwiGLU MLP       │
            └──────────────────────┬───────────────────────┘
                                   │
                                   ▼
                                 [...]
                                   │
                                   ▼
            ┌──────────────────────────────────────────────┐
            │ Layer {internals['num_layers']} : Self-Attention + SwiGLU MLP      │
            └──────────────────────┬───────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────────┐
            │ [ Final RMSNorm ]                            │
            └──────────────────────┬───────────────────────┘
                                   │
                                   ▼
            ┌──────────────────────────────────────────────┐
            │ [ LM Head ] ➔ Logits (Vocab Size: {internals['vocab_size']:,})│
            └──────────────────────┬───────────────────────┘
                                   │
                                   ▼
            [ Softmax ] ➔ Next Token Probabilities
            ```
            """)

        with col_arch_b:
            st.markdown("### 🎚️ Inspect a Transformer Layer")
            selected_layer_num = st.slider(
                "Select Transformer Layer to inspect:",
                min_value=1,
                max_value=internals["num_layers"],
                value=12,
                help="Select any layer from Layer 1 to Layer 28"
            )
            
            # Layer index in 0-indexed hidden states (Layer 1 is index 1, Layer 28 is index 28)
            layer_tensor = internals["hidden_states"][selected_layer_num].squeeze(0).numpy() # [seq_len, hidden_dim]
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Layer {selected_layer_num} Specifications</h4>
                <ul>
                    <li><b>Architecture:</b> Pre-RMSNorm Causal Self-Attention Block</li>
                    <li><b>Attention Heads:</b> {internals['num_heads']} Query/Key/Value Heads</li>
                    <li><b>Head Dimension:</b> {internals['hidden_dim'] // max(1, internals['num_heads'])} dims per head</li>
                    <li><b>Feed-Forward Network:</b> SwiGLU (Swish-Gated Linear Unit)</li>
                    <li><b>Residual Connection:</b> <code>x + Sublayer(RMSNorm(x))</code></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Layer {selected_layer_num} Output Activation Statistics (across prompt tokens):**")
            col_st1, col_st2, col_st3 = st.columns(3)
            col_st1.metric("Mean Activation", f"{float(np.mean(layer_tensor)):.4f}")
            col_st2.metric("Activation Std", f"{float(np.std(layer_tensor)):.4f}")
            col_st3.metric("Avg Token Norm", f"{float(np.linalg.norm(layer_tensor, axis=-1).mean()):.2f}")
            
            # Bar chart of token L2 norms at this layer
            token_norms = np.linalg.norm(layer_tensor, axis=-1)
            fig_layer_norms = go.Figure(go.Bar(
                x=tok_details["tokens"],
                y=token_norms,
                marker=dict(color=token_norms, colorscale="Purples"),
                text=[f"{n:.2f}" for n in token_norms],
                textposition="outside"
            ))
            fig_layer_norms.update_layout(
                title=f"Token Vector L2 Norms at Output of Layer {selected_layer_num}",
                xaxis_title="Tokens",
                yaxis_title="Vector Norm (Magnitude)",
                template="plotly_dark",
                height=300,
                margin=dict(l=30, r=30, t=40, b=30)
            )
            st.plotly_chart(fig_layer_norms, use_container_width=True)

    # ----------------------------------------------------
    # TAB 5: Attention Heatmap (Step 7)
    # ----------------------------------------------------
    with tab_attention:
        st.subheader("🔍 Multi-Head Self-Attention Visualization")
        st.markdown("""
        In each Transformer layer, **Multi-Head Self-Attention** calculates how strongly each token in the sequence 
        attends to every previous token:
        $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
        """)
        
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            sel_layer = st.slider("Select Transformer Layer:", 0, internals["num_layers"] - 1, value=min(12, internals["num_layers"] - 1))
        with col_ctrl2:
            sel_head = st.slider("Select Attention Head:", 0, internals["num_heads"] - 1, value=min(0, internals["num_heads"] - 1))
            
        # Extract attention weights for [layer][0, head, :, :]
        attn_matrix = internals["attentions"][sel_layer][0, sel_head].numpy()
        
        fig_attn = plot_attention_heatmap(attn_matrix, tok_details["tokens"], sel_layer, sel_head)
        st.plotly_chart(fig_attn, use_container_width=True)
        
        st.info("💡 **How to read:** Row = Query token (the token looking for context). Column = Key token (the token providing context). Darker blue indicates stronger attention weight.")

    # ----------------------------------------------------
    # TAB 6: Hidden States Across Layers (Step 8)
    # ----------------------------------------------------
    with tab_hidden:
        st.subheader("🧠 Hidden State Representations Across Layers")
        st.markdown("""
        As tokens pass upward from Layer 0 (Input Embedding) through Layer 28, their internal representations 
        evolve from raw lexical identity into rich contextual, syntactic, and semantic concepts.
        """)
        
        # Multiselect layers to compare (Step 8: Layer 1, Layer 10, Layer 20, Layer 28)
        default_layers = [0, min(1, internals["num_layers"]), min(10, internals["num_layers"]), min(20, internals["num_layers"]), internals["num_layers"]]
        chosen_layers = st.multiselect(
            "Select Layers to Compare in PCA Representation Space:",
            options=list(range(internals["num_layers"] + 1)),
            default=list(sorted(set(default_layers))),
            format_func=lambda l: "Layer 0 (Embedding Output)" if l == 0 else f"Layer {l}"
        )
        
        if chosen_layers:
            fig_h_pca = plot_hidden_states_pca(internals["hidden_states"], tok_details["tokens"], chosen_layers)
            st.plotly_chart(fig_h_pca, use_container_width=True)

    # ----------------------------------------------------
    # TAB 6: Logits & Probabilities
    # ----------------------------------------------------
    with tab_logits:
        st.subheader("📊 Output Logits & Softmax Probabilities")
        st.markdown("""
        The final hidden state of the last token is projected by the **LM Head** onto the entire vocabulary 
        (producing unnormalized **Logits**), which pass through **Softmax** to form a probability distribution.
        """)
        
        fig_logits = plot_logits_bar_chart(internals["top_predictions"])
        st.plotly_chart(fig_logits, use_container_width=True)
        
        st.markdown("### 🏆 Top Next-Token Candidate Predictions")
        df_top = pd.DataFrame(internals["top_predictions"])
        st.dataframe(df_top[["token", "probability", "percentage", "logit", "token_id"]], use_container_width=True)

    # ----------------------------------------------------
    # TAB 7: Step-by-Step Generation
    # ----------------------------------------------------
    with tab_generation:
        st.subheader("⏩ Autoregressive Generation Trace")
        st.markdown("""
        LLMs generate responses **one token at a time**. In each iteration, the previously generated token 
        is appended back to the input prompt, and the model predicts the next token.
        """)
        
        fig_gen = plot_token_generation_timeline(step_history)
        st.plotly_chart(fig_gen, use_container_width=True)
        
        st.markdown("### 📜 Generation Timeline & Alternative Candidates")
        for s in step_history:
            with st.expander(f"Step {s['step']}: Selected Token '{s['display_str']}' (Confidence: {s['percentage']})"):
                st.markdown(f"**Sequence so far:**")
                st.code(s["current_text"])
                
                st.markdown("**Top Alternative Candidates Considered at this Step:**")
                cand_df = pd.DataFrame(s["candidates"])
                cand_df["prob"] = cand_df["prob"].apply(lambda p: f"{p*100:.2f}%")
                st.table(cand_df)
