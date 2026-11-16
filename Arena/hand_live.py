"""手掌偵測實測（只開相機，手臂不動）：前 6 秒拍參考（不要伸手），之後 40 秒內把手伸到棋盤上方。

    python hand_live.py         # 結果印出來，標註圖存 out/hand_*.jpg
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Calib"))
import hand                                   # noqa: E402
from common import OrbbecCamera               # noqa: E402
from world import load                        # noqa: E402

sys.path.insert(0, HERE)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
cal = load("B")
cam = OrbbecCamera(depth=True)
print("拍參考（不要伸手）…", flush=True)
refs = []
t0 = time.time()
while len(refs) < 8 and time.time() - t0 < 15:
    refs.append(cam.grab(depth=True, timeout=10)[1])
ref = hand.reference(refs)
print("參考好了：棋盤中心深度 %.0f mm。現在把手掌張開伸到棋盤上方約 5 公分。" % ref[360, 640], flush=True)
t0, k, prev, stable = time.time(), 0, None, 0
while time.time() - t0 < 40:
    img, d = cam.grab(depth=True, timeout=10)
    palm, why = hand.detect(d, ref, cal)
    if palm:
        same = hand.same_place(palm, prev)
        stable = stable + 1 if (same and why is None) else 0
        print("%5.1fs 掌心 px=%s 離棋盤 %.0f mm 半寬 %.0f mm B座標 (%.0f, %.0f)%s%s" % (
            time.time() - t0, palm["px"], palm["h"], palm["half_mm"], palm["xy"][0], palm["xy"][1],
            "" if why is None else "  ← " + why, "  [穩定]" if stable >= 1 else ""), flush=True)
        prev = palm
        if k < 6 and (why is None or k < 3):
            cv2.imwrite(os.path.join(HERE, "out", "hand_%d.jpg" % k), hand.draw(img, palm, why))
            k += 1
    else:
        print("%5.1fs %s" % (time.time() - t0, why), flush=True)
        prev = None
    time.sleep(0.5)
cam.close()
