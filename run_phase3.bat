@echo off
title Launch SureshGPT Phase 3
echo ===================================================
echo Launching SureshGPT Phase 3 Platform
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo ===================================================
start cmd /k "run_backend.bat"
timeout /t 3 /nobreak >nul
start cmd /k "run_frontend.bat"
echo Both servers started!

