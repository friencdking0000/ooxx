$py = Join-Path $env:USERPROFILE 'miniconda3\envs\ooxx\python.exe'
$a = Join-Path $env:USERPROFILE 'Desktop\OOXX-robot-v1.1\Arena'
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'server\.py|cam_worker\.py|jog\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep 2
$cmd = 'cmd /c "set PYTHONIOENCODING=utf-8&& "' + $py + '" -u jog.py > out\jog.log 2>&1"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $a }
Start-Sleep 6
Get-Content (Join-Path $a 'out\jog.log') -Encoding UTF8 -Tail 3
$act = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c start "" "http://127.0.0.1:18090/"'
$pr = New-ScheduledTaskPrincipal -UserId "$env:COMPUTERNAME\$env:USERNAME" -LogonType Interactive
Register-ScheduledTask -TaskName 'ooxx_open_jog' -Action $act -Principal $pr -Force | Out-Null
Start-ScheduledTask -TaskName 'ooxx_open_jog'; Start-Sleep 3; Unregister-ScheduledTask -TaskName 'ooxx_open_jog' -Confirm:$false
'opened'
