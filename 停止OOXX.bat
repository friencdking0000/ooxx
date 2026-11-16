@echo off
chcp 65001 >nul
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'server\.py|cam_worker\.py|camera_worker\.py|jog\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
timeout /t 2 >nul
rem 用系統管理員或經 ssh 開的伺服器：一般權限讀不到命令列、上面會默默略過，所以看 18080 還有沒有人聽
for /f %%p in ('powershell -NoProfile -Command "(Get-NetTCPConnection -State Listen -LocalPort 18080 -ErrorAction SilentlyContinue).OwningProcess"') do (
  echo 對弈伺服器停不掉（PID %%p 還在跑、占著相機）：它是用系統管理員權限或經 ssh 開的，這個視窗沒權限關。
  echo → 對這個檔案按右鍵「以系統管理員身分執行」，或回到開它的視窗／ssh 關掉。
  pause
  exit /b 1
)
echo OOXX stopped (server, camera, arm monitor, jog).
timeout /t 3 >nul
