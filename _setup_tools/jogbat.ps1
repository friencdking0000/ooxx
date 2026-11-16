$d = Join-Path $env:USERPROFILE 'Desktop'
$bat = "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -File ""%USERPROFILE%\Desktop\OOXX-robot-v1.1\_setup_tools\startjog.ps1""`r`n"
[IO.File]::WriteAllText((Join-Path $d 'XYZ manual jog.bat'), $bat, [Text.Encoding]::ASCII)
'bat ok'
