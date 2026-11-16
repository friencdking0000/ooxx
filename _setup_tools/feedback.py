"""唯讀：只讀手臂 30004（不通就 30005）的回饋封包，不送任何指令。"""
import sys, socket, os, json
sys.stdout.reconfigure(encoding="utf-8")
R = os.path.join(os.environ["USERPROFILE"], "Desktop", "OOXX-robot-v1.1", "MG400")
sys.path.insert(0, R); os.chdir(R)
import numpy as np
from dobot_api import MyType
out = {}
for name, ip in (("A", "192.168.1.6"), ("B", "192.168.2.7")):
    for port in (30004, 30005):
        try:
            s = socket.create_connection((ip, port), timeout=3); s.settimeout(3)
            buf = b""
            while len(buf) < 1440:
                c = s.recv(1440 - len(buf))
                if not c: break
                buf += c
            s.close()
            if len(buf) < 1440: continue
            r = np.frombuffer(buf, dtype=MyType)[0]
            if int(r["test_value"]) != 0x123456789abcdef: continue
            out[name] = {"port": port, "robot_mode": int(r["robot_mode"]), "tcp_xyzr": [round(float(v), 1) for v in r["tool_vector_actual"][:4]],
                         "joints": [round(float(v), 1) for v in r["q_actual"][:4]], "DO1_vacuum": int(r["digital_outputs"]) & 1,
                         "speed_scaling": float(r["speed_scaling"]) if "speed_scaling" in r.dtype.names else None}
            break
        except Exception as e:
            out.setdefault(name, {})["err_%d" % port] = str(e)[:80]
print(json.dumps(out, ensure_ascii=False))
