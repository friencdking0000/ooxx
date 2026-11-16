[Console]::OutputEncoding = [Text.Encoding]::UTF8
"--- NIC"
Get-NetAdapter | Where-Object Status -eq 'Up' | ForEach-Object { $_.Name + ' | ' + $_.InterfaceDescription + ' | ' + $_.LinkSpeed }
Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' } | ForEach-Object { $_.InterfaceAlias + ' ' + $_.IPAddress + '/' + $_.PrefixLength }
"--- NEIGHBORS"
Get-NetNeighbor -AddressFamily IPv4 | Where-Object { $_.State -ne 'Unreachable' -and $_.LinkLayerAddress -notmatch '^(FF-FF|00-00-00|01-00)' } | ForEach-Object { $_.InterfaceAlias + ' ' + $_.IPAddress + ' ' + $_.LinkLayerAddress + ' ' + $_.State }
"--- USB"
Get-PnpDevice -PresentOnly | Where-Object { $_.Class -in 'Camera','Image','USBDevice' -or $_.FriendlyName -match 'Orbbec|Gemini|Dobot|IPEVO|Realtek USB|CH340|Serial' } | ForEach-Object { $_.Status + ' | ' + $_.Class + ' | ' + $_.FriendlyName }
"--- ARM PORTS"
foreach ($ip in '192.168.1.6','192.168.2.7','192.168.2.6','192.168.1.7') { $c = New-Object Net.Sockets.TcpClient; $ok = $c.ConnectAsync($ip, 29999).Wait(1200); $ip + ':29999 ' + ($ok -and $c.Connected); $c.Close() }
