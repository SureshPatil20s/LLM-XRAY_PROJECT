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
import torch

try:
    from sklearn.decomposition import PCA
except Exception:
    class PCA:
        """
        Lightweight, robust PCA implementation using NumPy SVD.
        Provides identical interface and results to sklearn.decomposition.PCA
        without introducing binary dependencies or DLL policy conflicts.
        """
        def __init__(self, n_components: int = 2):
            self.n_components = n_components
            self.explained_variance_ratio_ = np.zeros(n_components)

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            n_samples, n_features = X.shape
            k = min(self.n_components, n_samples, n_features)
            if k < 1:
                return np.zeros((n_samples, self.n_components))
            mean = np.mean(X, axis=0, keepdims=True)
            X_centered = X - mean
            U, S, _ = np.linalg.svd(X_centered, full_matrices=False)
            coords = U[:, :k] * S[:k]
            total_var = np.sum(S ** 2)
            var_ratios = (S[:k] ** 2) / (total_var if total_var > 0 else 1.0)
            self.explained_variance_ratio_ = np.zeros(self.n_components)
            self.explained_variance_ratio_[:k] = var_ratios
            if coords.shape[1] < self.n_components:
                padding = np.zeros((n_samples, self.n_components - coords.shape[1]))
                coords = np.hstack([coords, padding])
            return coords



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
    head_idx: int,
    log_scale: bool = False
) -> go.Figure:
    """
    Plots the attention matrix heatmap for a specific layer and attention head.
    attention_matrix shape: [seq_len, seq_len]

    Parameters
    ----------
    log_scale : bool
        If True, apply log-normalization so that low-weight attention patterns
        (normally crushed by "attention sink" tokens) become visible.
    """
    # Guard against empty or zero‑size matrices
    if attention_matrix.size == 0:
        # Return an empty figure with a helpful annotation
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No attention data for selected token range.", showarrow=False)
        empty_fig.update_layout(template="plotly_dark", height=400)
        return empty_fig
    
    z = attention_matrix.copy()
    
    if log_scale:
        # Shift into log space: log(w + eps) then rescale to [0, 1]
        eps = 1e-8
        z = np.log(z + eps)
        # Clean any NaN/inf before normalization
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=np.nanmin(z[np.isfinite(z)]) if np.any(np.isfinite(z)) else 0.0)
        if z.size > 0:
            z_min, z_max = z.min(), z.max()
            if z_max - z_min > 0:
                z = (z - z_min) / (z_max - z_min)
            else:
                z = np.zeros_like(z)
        colorbar_title = "Weight (log)"
    else:
        colorbar_title = "Weight"

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=tokens,
        y=tokens,
        colorscale="Blues",
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title=colorbar_title)
    ))

    scale_label = " [log scale]" if log_scale else ""
    fig.update_layout(
        title=f"Self-Attention Matrix — Layer {layer_idx}, Head {head_idx}{scale_label}",
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
            h = hidden_states[layer].squeeze(0).numpy()  # [seq_len, hidden_dim]
            seq_len = h.shape[0]
            for i, tok in enumerate(tokens):
                if i >= seq_len:
                    # Stop when we run out of hidden state vectors (e.g., extra special tokens)
                    break
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
    if not step_history:
        fig = go.Figure()
        fig.add_annotation(text="No tokens generated.", showarrow=False)
        fig.update_layout(template="plotly_dark", height=200)
        return fig

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


TOKEN_GLOSS = {
    "换句话": "In other words",
    "下面小编": "Editor's intro",
    "回答": "Answer",
    "计算": "Calculate",
    "结果": "Result",
    "总结": "Summary",
    "解释": "Explain",
    "第一": "First",
    "首先": "First of all",
    "综上所述": "In summary",
    "因此": "Therefore",
    "zellik": "feature",
}


def get_token_display_with_gloss(tok: str) -> str:
    clean = tok.strip().strip("'").strip('"')
    if clean in TOKEN_GLOSS:
        return f"{clean} [{TOKEN_GLOSS[clean]}]"
    return clean


def plot_logit_lens_progression(layer_predictions: List[Dict[str, Any]]) -> go.Figure:
    """
    Interactive line/scatter chart showing the top token and its confidence emerging across layers.
    Includes English glosses for multilingual tokens in early/mid layers.
    """
    layers = [p["layer"] for p in layer_predictions]
    probs = [p["top_prob"] * 100 for p in layer_predictions]
    display_tokens = [get_token_display_with_gloss(p["top_token"]) for p in layer_predictions]

    hover_texts = []
    for p in layer_predictions:
        cands_formatted = []
        for c in p["candidates"][:3]:
            g = get_token_display_with_gloss(c["token"])
            cands_formatted.append(f"• {g}: {c['percentage']}")
        cands_str = "<br>".join(cands_formatted)
        
        top_g = get_token_display_with_gloss(p["top_token"])
        hover_texts.append(
            f"<b>{p['layer_name']}</b><br>"
            f"Top Prediction: <b>{top_g}</b> ({p['percentage']})<br>"
            f"<b>Top Candidates:</b><br>{cands_str}"
        )

    fig = go.Figure()

    # Base line connecting layers
    fig.add_trace(go.Scatter(
        x=layers,
        y=probs,
        mode="lines+markers+text",
        name="Top Prediction Probability",
        line=dict(color="#38bdf8", width=3, shape="spline"),
        marker=dict(
            size=12,
            color=probs,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Confidence (%)", len=0.8)
        ),
        text=display_tokens,
        textposition="top center",
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    fig.update_layout(
        title="Logit Lens: Emergence of Predicted Token Identity Across Transformer Layers",
        xaxis_title="Transformer Layer (0 = Embedding, N = Final Decoder Block)",
        yaxis_title="Top Token Confidence (%)",
        template="plotly_dark",
        height=480,
        yaxis=dict(range=[0, 115]),
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        margin=dict(l=50, r=50, t=60, b=50)
    )
    return fig

