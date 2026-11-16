[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$py = Join-Path $env:USERPROFILE 'miniconda3\envs\ooxx\python.exe'
$a = Join-Path $env:USERPROFILE 'Desktop\OOXX-robot-v1.1\Arena'
New-Item -ItemType Directory -Force (Join-Path $a 'out') | Out-Null
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'server\.py|cam_worker\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep 2
$cmd = 'cmd /c "set PYTHONIOENCODING=utf-8&& set ARENA_ASR_URL=https://asr.qianpro.shop&& set CAM_ALWAYS_DEPTH=1&& "' + $py + '" -u server.py > out\server_run.log 2> out\server_err.log"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $a }
"started pid " + $r.ProcessId + " rc " + $r.ReturnValue
Start-Sleep 30
Get-Content (Join-Path $a 'out\server_run.log') -Encoding UTF8 -Tail 6
Set-Location $a
& $py ctl.py
