@echo off
chcp 65001 >nul
set "ROOT=%USERPROFILE%\Desktop\OOXX-robot-v1.1"
set "PY=%USERPROFILE%\miniconda3\envs\ooxx\python.exe"
if not exist "%PY%" set "PY=python"
set "OOXX_PYTHON=%PY%"
set PYTHONIOENCODING=utf-8
set ARENA_ASR_URL=https://asr.qianpro.shop
cd /d "%ROOT%"
title OOXX arm monitor
"%PY%" deploy.py --console
timeout /t 3 >nul
