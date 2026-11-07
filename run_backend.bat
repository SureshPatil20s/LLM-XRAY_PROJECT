@echo off
title SureshGPT FastAPI Backend
echo ===================================================
echo Starting SureshGPT Backend (FastAPI + Groq + RAG)
echo URL: http://localhost:8000
echo Docs: http://localhost:8000/docs
echo ===================================================
python -m uvicorn server.app.main:app --host 0.0.0.0 --port 8000 --reload
pause

