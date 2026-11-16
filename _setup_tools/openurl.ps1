$k = (Get-Content (Join-Path $env:USERPROFILE 'Desktop\OOXX-robot-v1.1\Arena\access.txt')).Trim()
$url = "http://127.0.0.1:18080/?k=$k"
$act = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c start `"`" `"$url`""
$pr = New-ScheduledTaskPrincipal -UserId "$env:COMPUTERNAME\$env:USERNAME" -LogonType Interactive
Register-ScheduledTask -TaskName 'ooxx_open_url' -Action $act -Principal $pr -Force | Out-Null
Start-ScheduledTask -TaskName 'ooxx_open_url'
Start-Sleep 4
(Get-ScheduledTaskInfo -TaskName 'ooxx_open_url').LastTaskResult
Unregister-ScheduledTask -TaskName 'ooxx_open_url' -Confirm:$false
"opened $url"
