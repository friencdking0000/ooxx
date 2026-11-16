$d = Join-Path $env:USERPROFILE 'Desktop'
Get-ChildItem $d -Filter *.bat | Where-Object { (Get-Content $_.FullName -Raw) -match 'ARENA_ASR_URL' -and (Get-Content $_.FullName -Raw) -notmatch 'CAM_ALWAYS_DEPTH' } | ForEach-Object {
    $t = (Get-Content $_.FullName -Raw) -replace 'set ARENA_ASR_URL=https://asr.qianpro.shop', "set ARENA_ASR_URL=https://asr.qianpro.shop`r`nset CAM_ALWAYS_DEPTH=1"
    [IO.File]::WriteAllText($_.FullName, $t, [Text.Encoding]::ASCII); "patched bat"
}
