# 🤖 SureshGPT — Phase 3: ChatGPT-Style AI Platform

[![Next.js](https://img.shields.io/badge/Next.js-16.4-black.svg?logo=next.js)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-19.3-blue.svg?logo=react)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6.svg?logo=typescript)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-v4-38B2AC.svg?logo=tailwind-css)](https://tailwindcss.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Groq Cloud](https://img.shields.io/badge/Groq-LPU_Hosted_Inference-F55036.svg)](https://groq.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Ready-336791.svg?logo=postgresql)](https://www.postgresql.org/)

An enterprise-grade, multi-user ChatGPT-style web platform built with **Next.js (App Router)**, **React 19**, **TypeScript**, **Tailwind CSS**, and a high-performance **FastAPI** backend. Powered by hosted cloud LLMs (**Groq Cloud**, **Llama 3.3 70B**, **Qwen 2.5 32B**) with zero local GPU required.

---

## 🚀 How to Run the Project

### Option 1: 1-Click Launch (Windows — Recommended)
Simply double-click the launch script in the project root:
```cmd
run_phase3.bat
```
This automatically starts both the **FastAPI Backend** and the **Next.js Frontend** in separate terminal windows.

---

### Option 2: Manual Terminal Launch

#### Step 1: Start the FastAPI Backend
Open a terminal in the project root and run:
```bash
python -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

#### Step 2: Start the Next.js Frontend
Open a second terminal, navigate into the `client` directory, and start the development server:
```bash
cd client
npm run dev
```
- **Web Application:** [http://localhost:3000](http://localhost:3000)

---

## ⚙️ Configuration (`.env`)

In the project root, open [.env](file:///c:/Users/sures/Desktop/Academia/LLM-XRAY_PROJECT/.env):

```ini
# 1. Groq Hosted LLM API Key (Free at https://console.groq.com/keys)
# If left blank, the platform runs in local simulation mode for testing
GROQ_API_KEY=gsk_your_groq_api_key_here

# 2. Default Model
DEFAULT_MODEL=llama-3.3-70b-versatile

# 3. Database (Optional for Cloud PostgreSQL)
# Leave blank to automatically use local SQLite (zero configuration needed)
# Or enter a PostgreSQL connection string (e.g. Supabase, Neon, Render):
# DATABASE_URL=postgresql://user:password@host:5432/dbname

# 4. Security
SECRET_KEY=suresh_gpt_super_secret_jwt_key_2026_phase3_production
```

---

## 🌟 Core Features

### 1. Modern ChatGPT-Style UI (No Streamlit)
- Sleek ChatGPT dark theme (`#121212`, `#212121`, emerald accents).
- Sidebar with **+ New chat**, real-time search, pinned chats, and date-sorted conversation history.
- Markdown and code block syntax highlighting with **1-click Copy Code** feedback.
- Message actions: Copy full response, Regenerate answer, and Thumbs Up / Down rating.

### 2. Hosted LLM Inference & Free Multimodal Models
- **Llama 3.3 70B & Qwen 2.5 32B:** Ultra-fast streaming (300+ tokens/sec) via Groq Cloud API.
- **AI Image Generator (Free Turbo):** Type `/image <prompt>` to generate 1024×1024 high-resolution images inline with zero API key required.
- **Multimodal Vision & OCR:** Attach images (`.png`, `.jpg`, `.jpeg`, `.webp`) for automatic text extraction, document transcription, and visual analysis.

### 3. Document Knowledge Base (RAG)
- Multi-format file ingestion supporting **PDF, DOCX, TXT, CSV, and JSON**.
- User-isolated semantic retrieval with provenance citation chips showing exact file name, page numbers, similarity scores, and source excerpts.
- Real-time **RAG Toggle** switch in the top header to enable or disable document retrieval on the fly.

### 4. Authentication & Data Privacy
- **User Registration & Login:** Salted bcrypt password hashing with JWT access tokens.
- **Google OAuth:** 1-click Google authentication integration.
- **Ephemeral Guest Mode:** Visitors can chat immediately without logging in; guest chats remain temporary and do not clutter the database.
- **Persistent Storage:** Registered users get persistent, cross-device conversation histories stored in PostgreSQL / SQLite.

---

## 🧪 Testing & Verification

Run the automated end-to-end test suite to verify all backend routes, auth flows, streaming, and RAG retrieval:

```bash
python test_phase3_e2e.py
```

Test coverage includes:
- `✓ Health Check & Model Catalog`
- `✓ Bcrypt Registration & JWT Login`
- `✓ Google OAuth Sign-In`
- `✓ Guest Seamless Session`
- `✓ Multi-format Document Ingestion & Chunking`
- `✓ Real-time SSE Chat Streaming with RAG Provenance Citations`
- `✓ Free Image Generation (Turbo Diffusion)`
- `✓ Free Vision & OCR (Llama 3.2 Vision)`

---

## 📁 Project Structure

```text
LLM-XRAY_PROJECT/
├── client/                     # Next.js 16 + React 19 + TypeScript Frontend
│   ├── src/
│   │   ├── app/                # App router (page.tsx, layout.tsx, globals.css)
│   │   ├── components/         # Sidebar, Navbar, ChatArea, ChatInput, Modals
│   │   ├── context/            # AuthContext (JWT & session state)
│   │   ├── lib/                # API client with SSE streaming reader
│   │   └── types/              # Strict TypeScript interfaces
│   └── package.json
│
├── server/                     # FastAPI Python Backend
│   └── app/
│       ├── api/                # auth.py, chat.py, documents.py routers
│       ├── core/               # config.py, security.py (JWT + bcrypt)
│       ├── db/                 # session.py (PostgreSQL & SQLite engine)
│       ├── models/             # entities.py (SQLAlchemy ORM models)
│       ├── schemas/            # schemas.py (Pydantic v2 validation models)
│       ├── services/           # llm_service.py (Groq & Image), rag_service.py
│       └── main.py             # FastAPI application entry point
│
├── run_phase3.bat              # 1-Click Windows launcher for both servers
├── run_backend.bat             # FastAPI backend runner script
├── run_frontend.bat            # Next.js frontend runner script
├── test_phase3_e2e.py          # Automated verification test suite
├── .env                        # Environment configuration
└── README.md                   # Project documentation
```
