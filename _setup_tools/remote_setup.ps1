$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$d = Join-Path $env:USERPROFILE 'Desktop'
Set-Location $d
Expand-Archive (Join-Path $d 'OOXX-robot-v1.1-20261115.zip') -DestinationPath $d -Force
"unzipped: " + (Test-Path (Join-Path $d 'OOXX-robot-v1.1\Arena\server.py'))
$conda = Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'
$envs = & $conda env list
if (-not ($envs -match '\envs\ooxx')) {
    & $conda create -y -n ooxx python=3.11 2>&1 | Select-Object -Last 4
}
$py = Join-Path $env:USERPROFILE 'miniconda3\envs\ooxx\python.exe'
"python: " + (& $py --version)
& $py -m pip install --disable-pip-version-check -r (Join-Path $d 'OOXX-robot-v1.1\requirements.txt') 2>&1 | Select-Object -Last 6
& $py -c "import numpy, cv2, opencc, cryptography, pyorbbecsdk; print('imports ok', numpy.__version__, cv2.__version__)"
