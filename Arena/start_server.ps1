# 以獨立程序啟動對弈伺服器（輸出寫到 out\server_run.log / server_err.log）。
#   .\start_server.ps1            雙臂（人機對弈、機器對下）
#   .\start_server.ps1 --b-only   舊的 B 單臂路徑
#   經 ssh 遠端呼叫時這裡的 Start-Process 子程序會跟著連線結束被關：改用 Invoke-CimMethod Win32_Process Create
#   .\start_server.ps1 --dry      乾跑（不連手臂、不開相機）
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force (Join-Path $here 'out') | Out-Null
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'server\.py|cam_worker\.py' -and $_.CommandLine -notmatch 'Console' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
Start-Sleep 2
# 用哪個 Python：環境變數 OOXX_PYTHON ＞ conda 的 ooxx 環境 ＞ 系統的 python
$py = $env:OOXX_PYTHON
if (-not $py) {
    foreach ($c in @("$env:USERPROFILE\miniconda3\envs\ooxx\python.exe", "$env:USERPROFILE\anaconda3\envs\ooxx\python.exe",
                     "D:\anaconda3\envs\ooxx\python.exe", "C:\ProgramData\miniconda3\envs\ooxx\python.exe")) {
        if (Test-Path $c) { $py = $c; break }
    }
}
if (-not $py) { $py = 'python' }
$a = @('-u', 'server.py') + $args
Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $here -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $here 'out\server_run.log') `
    -RedirectStandardError (Join-Path $here 'out\server_err.log')
"已啟動：$py $($a -join ' ')　紀錄：$here\out\server_run.log"
