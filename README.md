# 🔬 LLM X-RAY: Interactive Visualization of LLM Inference (Phase 1)

An interactive, local Large Language Model inspection web application built with **PyTorch**, **Hugging Face Transformers**, **Streamlit**, and **Plotly**.

Instead of treating the LLM as an opaque black box, **LLM X-Ray** opens up the internal mechanics of transformer inference—making every tensor transformation visible, interactive, and easy to understand.

---

## 🧭 The Core Inference Flow

```
Input Prompt
    │
    ▼
[ 1. Tokenizer ] ➔ Splits text into subword tokens
    │
    ▼
[ 2. Token IDs ] ➔ Numerical indices in model vocabulary (e.g. 151,936 tokens)
    │
    ▼
[ 3. Embeddings ] ➔ High-dimensional vectors (e.g. 1536-dim) in semantic space
    │
    ▼
[ 4. Transformer Layers ] ➔ 28 stacked blocks of Multi-Head Self-Attention + MLP
    │
    ├──► [ Self-Attention ] ➔ Query, Key, Value interaction between all tokens
    └──► [ Hidden States ] ➔ Representations evolve from lexical to conceptual
    │
    ▼
[ 5. LM Head (Unembedding) ] ➔ Projects final hidden state to vocabulary dimension
    │
    ▼
[ 6. Logits & Softmax ] ➔ Unnormalized scores converted to probabilities (0% - 100%)
    │
    ▼
[ 7. Sampling / Decoding ] ➔ Next token selected via Greedy / Top-P / Top-K / Temperature
    │
    ▼
[ 8. Response Loop ] ➔ Autoregressively feeds token back to generate full output
```

---

## 📁 Project Structure

```
LLM-XRay/
├── app.py              # Main interactive Streamlit web dashboard
├── model.py            # Model loader, forward pass inspector & step-by-step generator
├── tokenizer.py        # Subword tokenization, token IDs, & text decoding
├── visualization.py    # Plotly interactive figures (PCA, Heatmaps, Bar Charts)
├── test_pipeline.py    # CLI verification script for testing inference locally
├── requirements.txt    # Python package dependencies
└── README.md           # Documentation and guide
```

---

## 🚀 Quickstart Guide

### 1. Requirements & Environment
Make sure Python 3.10+ is installed. Install all dependencies:
```bash
cd C:\Users\sures\.gemini\antigravity-ide\scratch\LLM-XRay
pip install -r requirements.txt
```

### 2. Verify Inference Locally (Step 2)
Run the automated test pipeline to verify tokenization, forward pass, and generation:
```bash
python test_pipeline.py
```

### 3. Launch the Interactive Dashboard (Step 3 to 10)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 What You Can Inspect in the App

1. **💬 Chat & Overview**:
   - Enter any prompt, run inference, and view model configuration summary (layers, heads, dimensions).

2. **🔤 Tokenization**:
   - Inspect subword breakdown, special tokens, character lengths, and Token ID mappings.
   - Interactive table and visual token chips.

3. **🧬 Embeddings**:
   - Explore the input embedding vectors (1536-dimensional for Qwen2.5-1.5B).
   - 2D PCA scatter plot showing semantic spatial clustering of tokens.
   - Vector statistics (mean, std dev, L2 norm, min/max) and 1D dimension profile.

4. **🔍 Attention Heatmaps**:
   - Select any of the **28 Transformer Layers** and **12 Attention Heads**.
   - Interactive $N \times N$ matrix showing how tokens attend to one another (Query vs. Key).

5. **🧠 Hidden States**:
   - Track representation changes across layers (Layer 0, Layer 10, Layer 20, Layer 28).
   - 2D PCA layer comparison illustrating representation trajectory.

6. **📊 Logits & Probabilities**:
   - View top next-token candidate predictions from the LM Head.
   - Probability bar chart with percentage likelihoods and raw logit scores.

7. **⏩ Step-by-Step Generation**:
   - Inspect token-by-token autoregressive generation.
   - See confidence scores and alternative candidate tokens considered at each generation step.

---

## 🎛️ Interactive Controls

- **Model Choice**: Switch between `Qwen/Qwen2.5-1.5B-Instruct` (default) and `Qwen/Qwen2.5-0.5B-Instruct` (fast CPU mode).
- **Temperature**: Controls randomness (0 = deterministic greedy, higher = creative).
- **Top-K & Top-P**: Controls token sampling candidate pool.
- **Max New Tokens**: Controls the length of the generated response.
- **Layer & Head Sliders**: Choose specific attention heads to inspect.
