import json, sys, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
A = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1\Arena"
k = open(A + r"\access.txt").read().strip()
s = json.load(urllib.request.urlopen("http://127.0.0.1:18080/api/state?k=" + k, timeout=30))
print("phase", s["phase"], "| status", s["status"], "| human", s.get("human"), "| arm_error", s.get("arm_error"), "| counts", s.get("counts"))
print("board", "".join(c if c != " " else "." for c in s["board"]), "seen", "".join(c if c != " " else "." for c in (s["seen"] or [" "] * 9)))
for m in s["messages"][-3:]: print("  ", m.get("text"))
