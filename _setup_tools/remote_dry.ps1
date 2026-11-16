[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$py = Join-Path $env:USERPROFILE 'miniconda3\envs\ooxx\python.exe'
$a = Join-Path $env:USERPROFILE 'Desktop\OOXX-robot-v1.1\Arena'
Set-Location $a
New-Item -ItemType Directory -Force out | Out-Null
$p = Start-Process -FilePath $py -ArgumentList '-u','server.py','--dry' -WorkingDirectory $a -WindowStyle Hidden -PassThru -RedirectStandardOutput out\dry_server.log -RedirectStandardError out\dry_server_err.log
Start-Sleep 15
Get-Content out\dry_server.log -Tail 2 -Encoding UTF8
& $py dry_run.py 2>&1 | Select-Object -Last 8
Stop-Process -Id $p.Id -Force
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'server\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
