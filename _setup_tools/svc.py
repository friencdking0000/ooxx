import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1\Arena"
sys.path.insert(0, R); os.chdir(R)
import services
t = time.time()
try:
    w = services.tts("新局開始，我下圈，你下叉。")
    print("TTS ok %d bytes %.1fs" % (len(w), time.time() - t))
except Exception as e:
    print("TTS FAIL", str(e)[:150])
t = time.time()
try:
    fn = [f for f in dir(services) if "intent" in f.lower() or "llm" in f.lower() or "chat" in f.lower()]
    print("llm funcs", fn)
except Exception as e:
    print(e)
import urllib.request
try:
    r = urllib.request.urlopen(services.ASR_URL + "/v1/health", timeout=6); print("ASR health", r.status)
except Exception as e:
    print("ASR 連不到：", str(e)[:100])
