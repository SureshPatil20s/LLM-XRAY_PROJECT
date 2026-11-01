# 🔬 LLM X-RAY: Interactive Visualization of Transformer Inference

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0%2B-3f4f75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

**LLM X-Ray** is an interactive, local Large Language Model inspection and diagnostic web application. Rather than treating modern LLMs as opaque black boxes, **LLM X-Ray** makes every tensor transformation inside the transformer visible, interactive, and intuitive.

Built with **PyTorch**, **Hugging Face Transformers**, **Streamlit**, and **Plotly**, LLM X-Ray runs completely on local hardware (with automatic CUDA GPU acceleration and CPU fallbacks) without requiring any external cloud APIs.

---

## 🧭 Core Transformer Inference Flow

```
                      Input Prompt (e.g. "What is AI?")run
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Subword Tokenizer                                                   │
│    • Splits prompt into subword tokens (Byte-Pair Encoding / Qwen BPE) │
│    • Maps tokens to numerical Token IDs from 151,936 vocabulary       │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ [1, seq_len]
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. Input Embedding Matrix Lookup                                       │
│    • Looks up continuous vector representation (1536-dim or 896-dim)   │
│    • Places tokens in geometric semantic space                         │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ [1, seq_len, hidden_dim]
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Deep Transformer Decoder Blocks (28 Layers)                         │
│    Each layer applies:                                                 │
│    ├── Pre-RMSNorm                                                     │
│    ├── Multi-Head Self-Attention (Q, K, V cross-token weight matrix)    │
│    ├── Residual Add Connection                                         │
│    ├── Post-Attention RMSNorm                                          │
│    ├── SwiGLU MLP (Feed-Forward Expansion & Projection)                │
│    └── Residual Add Connection                                         │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ [1, seq_len, hidden_dim]
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. Hidden States Evolution                                             │
│    • Trajectory from raw lexical tokens (Layer 0)                      │
│    • Through contextual, semantic, and conceptual synthesis (Layer 28) │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ Last token hidden state [1, hidden_dim]
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. LM Head (Unembedding Projection) & Softmax                          │
│    • Projects hidden vector to entire vocabulary (151,936 logits)      │
│    • Softmax converts raw logits into normalized probabilities (0-100%)│
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 6. Sampling & Autoregressive Decoding Loop                             │
│    • Selects next token via Greedy / Top-P / Top-K / Temperature       │
│    • Detects natural End-of-Sequence tokens (<|im_end|>)               │
│    • Appends new token to sequence and repeats autoregressively        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features & Innovations

- 🖥️ **Hardware Auto-Detection**:
  - Automatically queries CUDA availability.
  - Defaults to `Qwen/Qwen2.5-1.5B-Instruct` with FP16/BF16 when an NVIDIA GPU is detected.
  - Automatically activates `Qwen/Qwen2.5-0.5B-Instruct` in low-memory CPU mode if running without a dedicated GPU.
- 🎯 **Dynamic Token Requirement & Early-Stopping Analysis**:
  - Compares allocated generation budget with tokens actually required by the prompt.
  - Real-time tracking of tokens used, tokens saved, and efficiency percentages.
  - Explicit notification of natural EOS termination (`<|im_end|>`) versus budget exhaustion.
- 🧮 **Robust Zero-Conflict PCA Engine**:
  - Features a custom, standalone SVD-based Principal Component Analysis module in `visualization.py`.
  - Bypasses scikit-learn binary/DLL conflicts on Windows systems while maintaining 100% mathematical fidelity.
- 💬 **Instruct Chat Mode**:
  - Wraps prompts inside standard Qwen ChatML templates (`<|im_start|>user...`) for accurate instruction following and clean EOS termination.
- 📊 **8 Comprehensive Diagnostic Tabs**:
  - Every stage of LLM inference is broken down into an intuitive visual tab with Plotly charts and tables.

---

## 🔬 Dashboard Tour: The 8 Inspection Tabs

| Tab | Feature | Description |
|---|---|---|
| **1. 💬 Chat & Overview** | **Inference Summary & Dynamic Early-Stopping** | View user prompt, generated text, model configuration, full pipeline trace, and dynamic token usage analytics. |
| **2. 🔤 Tokenization** | **Subword Breakdown & Token IDs** | Visual token chips, delimiter-formatted strings, and a complete DataFrame mapping token indices, clean text, IDs, and lengths. |
| **3. 🧬 Embeddings** | **Semantic Space & Vector Profile** | Interactive 2D PCA scatter plot showing spatial clustering of input tokens, sequential trajectory lines, and single-token vector statistics (norm, mean, min/max). |
| **4. 📐 Transformer Layers** | **Architecture & Layer Activation Stats** | Full architectural block diagram, layer selector slider (Layers 1–28), activation norm bar charts, and SwiGLU MLP specifications. |
| **5. 🔍 Attention Heatmap** | **Multi-Head Self-Attention Matrices** | $N \times N$ interactive attention heatmaps for any selected layer and head, showing Query-to-Key token relationships. |
| **6. 🧠 Hidden States** | **Cross-Layer Representation Evolution** | Multi-layer PCA projection tracking how token representation geometry transforms from Layer 0 to Layer 28. |
| **7. 📊 Logits & Probabilities** | **LM Head Softmax Distribution** | Horizontal bar chart of top-15 next-token candidates, probability percentages, and raw logit scores. |
| **8. ⏩ Step-by-Step Generation** | **Autoregressive Trace & Candidate Pool** | Step-by-step token generation confidence timeline, early-stopping conditions, and expandable step inspectors displaying runner-up candidates. |

---

## 📁 Project Structure

```
LLM-XRAY_PROJECT/
├── app.py                # Main interactive Streamlit dashboard application
├── model.py              # Model loader, forward pass inspector, and autoregressive generator
├── tokenizer.py          # Hugging Face tokenizer wrapper, subword splitter, and token ID mapper
├── visualization.py      # Plotly chart builders and standalone PCA decomposition engine
├── test_pipeline.py      # Automated CLI pipeline verification script
├── requirements.txt      # Python library dependencies
└── README.md             # Project documentation and update log
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python **3.10+** (Python 3.10, 3.11, or 3.12 recommended)
- Git

### 2. Installation
Clone or navigate to the project directory:
```bash
cd c:\Users\sures\Desktop\Academia\LLM-XRAY_PROJECT
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

> **Note on PyTorch & Hardware:**  
> If you have an NVIDIA GPU with CUDA support, install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Verify Inference with the CLI Test Script
Before launching the web UI, verify that the tokenizer and model load properly on your system:
```bash
python test_pipeline.py
```
*Optional: Test with the lighter 0.5B model:*
```bash
python test_pipeline.py Qwen/Qwen2.5-0.5B-Instruct
```

### 4. Launch the Interactive Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎛️ Generation Controls & Parameter Guide

- **Active Model**: Toggle between `Qwen2.5-1.5B-Instruct` (full quality, best with GPU) and `Qwen2.5-0.5B-Instruct` (ultra-lightweight, best for CPU).
- **Temperature** (`0.0` to `1.5`):
  - `0.0`: Deterministic greedy decoding (always selects highest probability token).
  - `0.7`: Balanced creativity and coherence.
  - `> 1.0`: High randomness and creative diversity.
- **Top-P (Nucleus Sampling)** (`0.1` to `1.0`): Keeps the smallest set of tokens whose cumulative probability exceeds `top_p`.
- **Top-K** (`0` to `100`): Limits the candidate pool to the top $K$ tokens.
- **Max New Tokens** (`5` to `100`): Maximum token generation budget. (Capped at 100 on CPU to ensure UI responsiveness).
- **Instruct Mode (Smart Stopping)**: Enforces ChatML formatting so the model can naturally conclude answers with `<|im_end|>`.

---

## 📊 Technical Architecture Specifications

| Property | Qwen/Qwen2.5-1.5B-Instruct | Qwen/Qwen2.5-0.5B-Instruct |
|---|---|---|
| **Parameter Count** | ~1.54 Billion | ~0.49 Billion |
| **Transformer Layers** | 28 | 24 |
| **Attention Heads (Q)** | 12 | 14 |
| **KV Heads (GQA)** | 2 | 2 |
| **Hidden Dimension ($d_{\text{model}}$)** | 1,536 | 896 |
| **Intermediate Dimension (MLP)** | 8,960 | 4,864 |
| **Vocabulary Size** | 151,936 | 151,936 |
| **Native Context Length** | 32,768 tokens | 32,768 tokens |
| **Activation Function** | SwiGLU | SwiGLU |
| **Normalization** | RMSNorm | RMSNorm |
| **Position Embeddings** | RoPE (Rotary Position Embeddings) | RoPE (Rotary Position Embeddings) |

---

## 🚀 Project Phase 2 Upgrade: LocalGPT (ChatGPT-Style AI with RAG & LLM X-Ray)

**LocalGPT** upgrades the foundational LLM X-Ray system into a practical, multi-turn AI assistant with persistent conversation history, local document retrieval (RAG), streaming generation, and one-click deep mechanistic inspection.

### Key Capabilities in Phase 2
- 💬 **Multi-Turn Conversations**: Context-aware dialogue history managed via SQLite (`database.py` and `memory.py`).
- 📁 **Conversation Management**: Create new chats, resume past discussions, rename chats, and delete conversations in the sidebar.
- ⚡ **Real-Time Streaming Generation**: Autoregressive streaming of tokens via `TextIteratorStreamer` directly into Streamlit's chat interface.
- 📄 **Document RAG (PDF, DOCX, TXT)**: Local document ingestion using `pypdf`, `python-docx`, dense embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`), and `FAISS` vector search.
- 📑 **Source Attribution & Citations**: Automatic citation display showing source documents, page numbers, and relevant text excerpts.
- 🎛️ **Message Controls**: Copy responses, regenerate answers with preserved conversation context, and edit user queries.
- 🔬 **Integrated LLM X-Ray Mode**: One-click `[🔍 X-Ray]` button on any assistant message to inspect:
  - **Tokens**: Token breakdown, IDs, and compression metrics.
  - **Embeddings**: Vector norms and 2D PCA semantic projection.
  - **Layers**: Architectural parameter analysis.
  - **Attention**: Interactive Query-Key attention heatmaps with layer and head sliders.
  - **Hidden States**: Multi-layer hidden state evolution.
  - **Logits & Probabilities**: Next-token candidate probabilities and raw logits.

---

## 📝 Changelog & Recent Updates

This section is continually maintained as new capabilities and features are added to **LLM X-Ray**:

### [v2.0.0] - Project Phase 2: LocalGPT & RAG Integration
- **Full ChatGPT-Style Interface**: Deployed multi-turn chat with conversation memory, sidebar history, and edit/regenerate/copy message controls.
- **RAG Document Engine**: Integrated `pypdf`, `python-docx`, `FAISS`, and `SentenceTransformers` with traceable source citations.
- **Real-Time Token Streaming**: Real-time generation streaming with `st.write_stream`.
- **Integrated X-Ray Inspection**: Inline mechanistic X-Ray inspection panel for any chat response.
- **Phase 3 Claim Verification**: Curated ground truth and deterministic verification engine integrated into the research laboratory.

### [v1.2.0] - Dynamic Token Intelligence & Robust PCA
- **Dynamic Token Analytics**: Added real-time tracking of token budget allocation, tokens actually needed, tokens saved, and early-stopping rationale.
- **Natural EOS Detection**: Added handling for `<|im_end|>` and `<|endoftext|>` stopping tokens.
- **Zero-Dependency PCA**: Built a standalone NumPy SVD Principal Component Analysis engine in [visualization.py](file:///c:/Users/sures/Desktop/Academia/LLM-XRAY_PROJECT/visualization.py) to prevent Windows DLL conflicts.
- **Hardware Auto-Detection**: Integrated dynamic GPU/CPU detection to select optimal model weights automatically.

### [v1.1.0] - Deep Transformer Architecture Inspection
- **Layer & Head Inspection**: Added interactive slider to inspect each of the 28 Transformer Decoder blocks.
- **Activation Statistics**: Displaying mean, standard deviation, and L2 norm across token vectors at any selected layer.
- **Multi-Head Attention Heatmap**: Implemented Query-Key attention matrix visualizer with Blues colormap.
- **Hidden State Trajectory**: Added multi-layer PCA visualization comparing representation evolution across layers 0, 1, 10, 20, and 28.

### [v1.0.0] - Initial Release (Phase 1)
- Subword tokenization and Token ID mapping.
- Input embeddings inspection and 2D PCA semantic projection.
- Logits and Softmax probability distribution bar chart.
- Autoregressive token-by-token generation trace with top alternative candidates.
- Streamlit web interface with custom CSS design.

---

## 🤝 Maintenance & Adding New Features

When extending LLM X-Ray with new components:
1. **Adding Visualizations**: Implement new Plotly figures in [visualization.py](file:///c:/Users/sures/Desktop/Academia/LLM-XRAY_PROJECT/visualization.py) and export them for Streamlit.
2. **Adding Tensor Extractors**: Expand `inspect_inference_internals()` in [model.py](file:///c:/Users/sures/Desktop/Academia/LLM-XRAY_PROJECT/model.py) to extract additional internal layers (e.g., MLP gate activations, residual norms, KV cache).
3. **Adding UI Controls**: Add sidebar widgets or new tabs in [app.py](file:///c:/Users/sures/Desktop/Academia/LLM-XRAY_PROJECT/app.py).
4. **Update Documentation**: Record any newly introduced features, modules, or configurations under the **Changelog & Recent Updates** section of this README.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
