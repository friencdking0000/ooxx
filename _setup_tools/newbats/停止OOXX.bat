@echo off
chcp 65001 >nul
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'server\.py|cam_worker\.py|camera_worker\.py|jog\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
echo OOXX stopped (server, camera, arm monitor, jog).
timeout /t 3 >nul
