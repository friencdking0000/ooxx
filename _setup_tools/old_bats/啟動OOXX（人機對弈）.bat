@echo off
set PYTHONIOENCODING=utf-8
set ARENA_ASR_URL=https://asr.qianpro.shop
set CAM_ALWAYS_DEPTH=1
cd /d "%USERPROFILE%\Desktop\OOXX-robot-v1.1\Arena"
powershell -NoProfile -ExecutionPolicy Bypass -File start_server.ps1 
echo Starting... please wait 25 seconds
timeout /t 25 /nobreak >nul
start "" "http://127.0.0.1:18080/?k=gmZflp8irkqH"