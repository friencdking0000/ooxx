[Console]::OutputEncoding = [Text.Encoding]::UTF8
$d = Join-Path $env:USERPROFILE 'Desktop'
$a = Join-Path $d 'OOXX-robot-v1.1\Arena'
$k = (Get-Content (Join-Path $a 'access.txt')).Trim()
$url = "http://127.0.0.1:18080/?k=$k"
$start = @"
@echo off
set PYTHONIOENCODING=utf-8
set ARENA_ASR_URL=https://asr.qianpro.shop
cd /d "%USERPROFILE%\Desktop\OOXX-robot-v1.1\Arena"
powershell -NoProfile -ExecutionPolicy Bypass -File start_server.ps1 __ARGS__
echo Starting... please wait 25 seconds
timeout /t 25 /nobreak >nul
start "" "$url"
"@
$stop = @"
@echo off
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { `$_.CommandLine -match 'server\.py|cam_worker\.py' } | ForEach-Object { Stop-Process -Id `$_.ProcessId -Force }"
echo OOXX server stopped.
timeout /t 3 >nul
"@
[IO.File]::WriteAllText((Join-Path $d '啟動OOXX（人機對弈）.bat'), $start.Replace('__ARGS__', ''), [Text.Encoding]::ASCII)
[IO.File]::WriteAllText((Join-Path $d '啟動OOXX（雙臂）.bat'), $start.Replace('__ARGS__', '--use-a'), [Text.Encoding]::ASCII)
[IO.File]::WriteAllText((Join-Path $d '停止OOXX.bat'), $stop, [Text.Encoding]::ASCII)
[IO.File]::WriteAllText((Join-Path $d 'OOXX對弈網頁.url'), "[InternetShortcut]`r`nURL=$url`r`n", [Text.Encoding]::ASCII)
New-Item -ItemType Directory -Force (Join-Path $d 'OOXX-robot-v1.1\_setup_tools') | Out-Null
foreach ($f in 'rt.py','cam.py','chk.py','camtest.py','park.py','b.py','svc.py','probe.ps1','ci.ps1','remote_setup.ps1','remote_dry.ps1','startsrv.ps1','openurl.ps1','cam_0.jpg','cam_3.jpg','OOXX-robot-v1.1-20261115.zip') {
    $p = Join-Path $d $f
    if (Test-Path $p) { Move-Item $p (Join-Path $d 'OOXX-robot-v1.1\_setup_tools\') -Force -ErrorAction SilentlyContinue }
}
Get-ChildItem $d | Where-Object { $_.Name -match 'OOXX' } | ForEach-Object { $_.Name }
