"""等使用者整理盤面：棋盤連續 3 次讀到全空 → 用深度量兩疊高度 → 印出顆數（不自動改計數檔，讓人確認）。

    python watch_tidy.py [逾時秒數，預設 600]
"""
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import find_pieces                      # noqa: E402
from common import OrbbecCamera          # noqa: E402

TIMEOUT = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
PIECE = 3.7


def measure(cam, stk):
    ds, img = [], None
    for _ in range(6):
        img, d = cam.grab(depth=True, skip=1, timeout=15)
        ds.append(d)
    d = np.pad(np.median(np.stack(ds), axis=0), 120)      # 料區可能靠近畫面邊緣：補邊（0＝無效）
    out = {}
    for name, v in stk["stocks"].items():
        u, w = int(v["px"][0]) + 120, int(v["px"][1]) + 120
        top = d[w - 12:w + 13, u - 12:u + 13]
        top = float(np.median(top[top > 0]))
        yy, xx = np.mgrid[-110:110, -110:110]
        rr = np.hypot(xx, yy)
        ring = d[w - 110:w + 110, u - 110:u + 110]
        table = float(np.median(ring[(rr > 75) & (rr < 105) & (ring > 0)]))
        out[name] = ((table - top) / PIECE, top, table)
    return out, img


def main():
    plan = json.load(open(os.path.join(HERE, "cell_plan.json"), encoding="utf-8"))
    stk = json.load(open(os.path.join(HERE, "stock_B.json"), encoding="utf-8"))
    cam = OrbbecCamera(depth=True)
    try:
        t0, empty, last = time.time(), 0, None
        print("監看中：棋盤清空後自動量料區…", flush=True)
        while time.time() - t0 < TIMEOUT:
            img = cam.grab(skip=1, timeout=15)
            b = find_pieces.read_board(img, plan)
            s = "None" if b is None else "".join(c if c != " " else "." for c in b)
            if s != last:
                print("%5.0fs 盤面 %s" % (time.time() - t0, s), flush=True)
                last = s
            empty = empty + 1 if (b is not None and all(c == " " for c in b)) else 0
            if empty >= 3:
                time.sleep(3.0)                    # 等手離開料區
                res, im = measure(cam, stk)
                for name, (n, top, table) in res.items():
                    print("%s：頂面 %.0f mm、桌面 %.0f mm → %.1f 顆（約 %d 顆）" % (name, top, table, n, round(n)), flush=True)
                cv2.imwrite(os.path.join(HERE, "out", "tidy_done.jpg"), im)
                return
            time.sleep(1.5)
        print("逾時，棋盤還沒清空", flush=True)
    finally:
        cam.close()


if __name__ == "__main__":
    main()
