"""
Visualization module for LLM-XRay.
Provides Plotly interactive figures for:
- 2D PCA Embeddings Scatter Plot
- Embedding Vector Profile & Distribution
- Attention Weight Heatmaps
- Hidden State Evolution across Transformer Layers
- Logits and Probability Distributions Bar Charts
- Step-by-Step Token Generation Flow
"""
from typing import List, Dict, Any, Optional
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import torch


def plot_embeddings_pca(
    embeddings: torch.Tensor,
    tokens: List[str],
    token_ids: List[int]
) -> go.Figure:
    """
    Computes 2D PCA on input embedding vectors and displays interactive scatter plot.
    """
    # embeddings shape: [1, seq_len, hidden_dim] -> [seq_len, hidden_dim]
    emb_np = embeddings.squeeze(0).cpu().numpy()
    seq_len = emb_np.shape[0]
    
    if seq_len < 2:
        # Not enough tokens for 2D PCA
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 2 tokens to compute 2D PCA.",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(height=400, template="plotly_dark")
        return fig
        
    n_components = min(2, seq_len)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(emb_np)
    
    if coords.shape[1] == 1:
        pc1 = coords[:, 0]
        pc2 = np.zeros(seq_len)
        var_explained = f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}%"
    else:
        pc1 = coords[:, 0]
        pc2 = coords[:, 1]
        var1 = pca.explained_variance_ratio_[0] * 100
        var2 = pca.explained_variance_ratio_[1] * 100
        var_explained = f"Explained Variance: PC1={var1:.1f}%, PC2={var2:.1f}%"

    fig = go.Figure()
    
    # Add trajectory lines connecting consecutive tokens in sentence order
    fig.add_trace(go.Scatter(
        x=pc1,
        y=pc2,
        mode="lines",
        line=dict(color="rgba(100, 180, 255, 0.4)", width=2, dash="dot"),
        hoverinfo="none",
        showlegend=False
    ))
    
    # Add scatter points
    hover_texts = [
        f"<b>Pos:</b> {i}<br><b>Token:</b> '{t}'<br><b>Token ID:</b> {tid}<br><b>PC1:</b> {x:.3f}<br><b>PC2:</b> {y:.3f}"
        for i, (t, tid, x, y) in enumerate(zip(tokens, token_ids, pc1, pc2))
    ]
    
    fig.add_trace(go.Scatter(
        x=pc1,
        y=pc2,
        mode="markers+text",
        marker=dict(
            size=14,
            color=list(range(seq_len)),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Token Index", len=0.7)
        ),
        text=tokens,
        textposition="top center",
        hovertext=hover_texts,
        hoverinfo="text",
        name="Tokens"
    ))
    
    fig.update_layout(
        title=f"2D PCA Projection of Input Token Embeddings ({var_explained})",
        xaxis_title="Principal Component 1 (PC1)",
        yaxis_title="Principal Component 2 (PC2)",
        template="plotly_dark",
        height=500,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_vector_profile(
    vector: np.ndarray,
    token_str: str,
    token_id: int
) -> go.Figure:
    """
    Plots the dimension distribution & sample values of an embedding vector.
    """
    fig = go.Figure()
    
    # Line plot of raw dimensions
    x_indices = np.arange(len(vector))
    fig.add_trace(go.Scatter(
        x=x_indices,
        y=vector,
        mode="lines",
        line=dict(color="#00e5ff", width=1),
        name="Dimension Values"
    ))
    
    fig.update_layout(
        title=f"Embedding Vector Profile for '{token_str}' (Token ID: {token_id}) — Dim: {len(vector)}",
        xaxis_title="Embedding Dimension Index",
        yaxis_title="Activation Value",
        template="plotly_dark",
        height=320,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    tokens: List[str],
    layer_idx: int,
    head_idx: int
) -> go.Figure:
    """
    Plots the attention matrix heatmap for a specific layer and attention head.
    attention_matrix shape: [seq_len, seq_len]
    """
    fig = go.Figure(data=go.Heatmap(
        z=attention_matrix,
        x=tokens,
        y=tokens,
        colorscale="Blues",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Weight")
    ))
    
    fig.update_layout(
        title=f"Self-Attention Matrix — Layer {layer_idx}, Head {head_idx}",
        xaxis_title="Key Tokens (Attended To)",
        yaxis_title="Query Tokens (Attending From)",
        template="plotly_dark",
        height=520,
        yaxis=dict(autorange="reversed"),  # Query 0 at the top
        margin=dict(l=60, r=60, t=60, b=60)
    )
    return fig


def plot_hidden_states_pca(
    hidden_states: List[torch.Tensor],
    tokens: List[str],
    selected_layers: List[int]
) -> go.Figure:
    """
    Visualizes how token representations evolve across selected transformer layers using 2D PCA.
    """
    # Collect points: for each selected layer, collect seq_len vectors
    vectors = []
    labels = []
    layer_tags = []
    
    for layer in selected_layers:
        if layer < len(hidden_states):
            h = hidden_states[layer].squeeze(0).numpy() # [seq_len, hidden_dim]
            for i, tok in enumerate(tokens):
                vectors.append(h[i])
                labels.append(f"L{layer}: {tok}")
                layer_tags.append(f"Layer {layer}")
                
    if len(vectors) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data points for PCA.", showarrow=False)
        fig.update_layout(template="plotly_dark", height=400)
        return fig
        
    vectors_np = np.array(vectors)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors_np)
    
    fig = go.Figure()
    
    # Group by layer
    for layer in selected_layers:
        indices = [idx for idx, tag in enumerate(layer_tags) if tag == f"Layer {layer}"]
        if not indices:
            continue
        fig.add_trace(go.Scatter(
            x=coords[indices, 0],
            y=coords[indices, 1],
            mode="markers+text",
            marker=dict(size=12),
            text=[tokens[i % len(tokens)] for i in indices],
            textposition="top center",
            name=f"Layer {layer}"
        ))
        
    fig.update_layout(
        title="Evolution of Token Representations Across Transformer Layers (2D PCA)",
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        template="plotly_dark",
        height=520,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_logits_bar_chart(top_predictions: List[Dict[str, Any]]) -> go.Figure:
    """
    Horizontal bar chart showing top next-token candidates and their probabilities.
    """
    # Reverse so highest probability is at the top
    rev = list(reversed(top_predictions))
    
    tokens = [p["token"] for p in rev]
    probs = [p["probability"] * 100 for p in rev]
    labels = [f"{p['probability']*100:.1f}% (ID: {p['token_id']})" for p in rev]
    
    fig = go.Figure(go.Bar(
        x=probs,
        y=tokens,
        orientation="h",
        marker=dict(
            color=probs,
            colorscale="Viridis",
            showscale=False
        ),
        text=labels,
        textposition="outside"
    ))
    
    fig.update_layout(
        title="Next Token Probability Distribution (Top Candidates from LM Head Softmax)",
        xaxis_title="Probability (%)",
        yaxis_title="Predicted Token",
        template="plotly_dark",
        height=450,
        margin=dict(l=80, r=80, t=50, b=40)
    )
    return fig


def plot_token_generation_timeline(step_history: List[Dict[str, Any]]) -> go.Figure:
    """
    Bar chart showing the model confidence (probability) for each sequentially generated token.
    """
    steps = [f"#{s['step']}: {s['display_str']}" for s in step_history]
    probs = [s["prob"] * 100 for s in step_history]
    
    fig = go.Figure(go.Bar(
        x=steps,
        y=probs,
        marker=dict(
            color=probs,
            colorscale="Tealgrn",
            showscale=False
        ),
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside"
    ))
    
    fig.update_layout(
        title="Token-by-Token Generation Confidence",
        xaxis_title="Generated Token Sequence",
        yaxis_title="Selected Token Probability (%)",
        template="plotly_dark",
        height=380,
        yaxis=dict(range=[0, 110]),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
