[Console]::OutputEncoding = [Text.Encoding]::UTF8
"SAC state (0=off 1=on 2=eval): " + (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' -ErrorAction SilentlyContinue).VerifiedAndReputablePolicyState
"--- CI events"
Get-WinEvent -LogName 'Microsoft-Windows-CodeIntegrity/Operational' -MaxEvents 6 -ErrorAction SilentlyContinue | ForEach-Object { $_.TimeCreated.ToString('HH:mm:ss') + ' id=' + $_.Id + ' ' + ($_.Message -replace '\s+',' ').Substring(0, [Math]::Min(260, ($_.Message -replace '\s+',' ').Length)) }
"--- AppLocker / policies"
(Get-CimInstance -Namespace root\Microsoft\Windows\DeviceGuard -ClassName Win32_DeviceGuard -ErrorAction SilentlyContinue | Select-Object CodeIntegrityPolicyEnforcementStatus, UsermodeCodeIntegrityPolicyEnforcementStatus | Out-String).Trim()
Get-ChildItem 'C:\Windows\System32\CodeIntegrity\CiPolicies\Active' -ErrorAction SilentlyContinue | ForEach-Object { $_.Name + ' ' + $_.LastWriteTime }
"--- DLL zone"
$d = Join-Path $env:USERPROFILE 'miniconda3\envs\ooxx\Lib\site-packages\pyorbbecsdk'
Get-ChildItem $d -Filter *.dll | Select-Object -First 4 | ForEach-Object { $_.Name + ' sig=' + (Get-AuthenticodeSignature $_.FullName).Status }
Get-ChildItem $d -Filter *.pyd | ForEach-Object { $_.Name + ' sig=' + (Get-AuthenticodeSignature $_.FullName).Status }
"last boot: " + (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
