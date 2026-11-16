import socket, sys, time
sys.stdout.reconfigure(encoding="utf-8")
for name, ip in (("A", "192.168.1.6"), ("B", "192.168.2.7")):
    for port in (30004, 30006, 30005):
        try:
            s = socket.create_connection((ip, port), timeout=3); s.settimeout(4)
            t = time.time(); n = 0
            try:
                while time.time() - t < 3:
                    d = s.recv(4096)
                    if not d: break
                    n += len(d)
            except socket.timeout:
                pass
            print(name, port, "3 秒收到 %d bytes" % n); s.close()
        except Exception as e:
            print(name, port, "失敗", e)
    try:
        s = socket.create_connection((ip, 29999), timeout=3); s.settimeout(3)
        s.send(b"RobotMode()"); print(name, "29999:", s.recv(200).decode(errors="replace").strip()[:60]); s.close()
    except Exception as e:
        print(name, "29999 失敗", e)
