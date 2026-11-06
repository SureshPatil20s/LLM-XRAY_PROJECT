# 🤖 SURESH.GPT Pro (LocalGPT Pro) — Multi-User Local AI Assistant with RAG & LLM X-Ray

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-ff4b4b.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-green.svg)](https://github.com/facebookresearch/faiss)
[![Firebase](https://img.shields.io/badge/Firebase-Admin%20%26%20Auth-orange.svg)](https://firebase.google.com/)

**SURESH.GPT Pro (Phase 3: LocalGPT Pro)** is a secure, multi-user local AI assistant. It integrates ChatGPT-style conversational capabilities, persistent per-user document RAG (Retrieval-Augmented Generation), Firebase Authentication for identity, and deep mechanistic interpretability via the **LLM X-Ray Laboratory**.

---

## 🌟 What's New in Phase 3 (LocalGPT Pro)

| Capability | Phase 2 (LocalGPT) | Phase 3 (LocalGPT Pro) |
|---|---|---|
| **Identity & Access** | Single local user | **Firebase Authentication** (Google Sign-In, Email/Password, Password Reset, Email Verification) |
| **Data Isolation** | Shared conversations & index | **Strict Per-User Isolation**: Chats, messages, document uploads, and FAISS vector indices are private to each user |
| **UI Aesthetics** | Basic dark theme | **Minimalism + Liquid Glass + Glassmorphism**: Light/Dark theme switch, custom avatars, floating chat dock |
| **Welcome Experience** | Empty chat screen | **Dynamic Greeting** ("Good evening, Suresh") + 4 interactive suggested prompt cards |
| **Document Management**| Sidebar file uploader | **Dedicated Document Library**: Multi-file drag-and-drop, stage progress bar, page/chunk counts, deletion |
| **Conversation Organization** | Simple list | **Search box**, **Pinned chats**, and **Date groupings** (Today, Yesterday, Previous 7 Days, Older) |
| **Chat Export** | None | **Export to Markdown (.md)** and **Formatted PDF (.pdf)** via ReportLab |
| **LLM X-Ray Access** | Fixed inspection panel | **Inline Collapsible Side Drawer** next to individual messages + Full 11-tab Mechanistic Lab |
| **User Preferences** | Session-only settings | **Persistent User Settings**: Custom default system prompt, temperature, top-k, top-p, and max tokens |

---

## 🏗️ Architecture & Security Model

```
Browser / Streamlit UI
  │
  ├─► [Firebase Auth Web / Admin SDK] ──► Token Verification (Backend never trusts raw UID)
  │
  └─► Local Laptop Runtime (100% On-Device AI)
        ├── Qwen2.5 Local LLM (0.5B CPU fallback / 1.5B GPU)
        ├── SQLite DB (localgpt.db with users, conversations, messages, documents, settings)
        └── data/users/<firebase_uid>/
              ├── documents/     (User's uploaded PDF/TXT/DOCX)
              ├── faiss_index/   (User's private FAISS IndexFlatIP & chunks)
              └── exports/       (Generated Markdown & ReportLab PDFs)
```

> **Security Guarantee:** Firebase is used **only** for identity and token authentication. Prompts, document contents, embeddings, model weights, and conversation histories never leave your laptop.

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/LLM-XRAY_PROJECT.git
cd LLM-XRAY_PROJECT

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Firebase (Optional for Live Firebase Auth)

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your Firebase Web App credentials in `.env` and place your `serviceAccountKey.json` in the project root:
```env
FIREBASE_API_KEY=your_api_key_here
FIREBASE_AUTH_DOMAIN=localgpt-pro.firebaseapp.com
FIREBASE_PROJECT_ID=localgpt-pro
FIREBASE_APP_ID=1:1234567890:web:abcdef
GOOGLE_APPLICATION_CREDENTIALS=serviceAccountKey.json
```

*(Note: If Firebase credentials are not yet configured, the app includes a seamless built-in offline test switcher with personas like Suresh and Priya to test data isolation immediately!)*

### 3. Launch the Application

```bash
streamlit run app.py
```

---

## 🧪 Testing and Verification

Run the full automated test suites:

```bash
# 1. Phase 3 LocalGPT Pro Multi-User & Export Test Suite
python test_phase3_localgpt_pro.py

# 2. Phase 2 LocalGPT Pipeline Tests
python test_localgpt_pipeline.py

# 3. Phase 1 Mechanistic & Claim Verification Tests
python test_phase3_verification.py
```

All tests execute on CPU with zero network dependency on external LLM APIs.
