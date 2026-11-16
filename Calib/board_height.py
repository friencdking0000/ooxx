"""棋盤高度圖：棋盤清空後，用深度量每格中心的棋盤面，存成「比格 4 高幾 mm」（board_dz.json），
robot.place 放開棋子時依格修正 Z（棋盤不平：2026-11-14 實測格 5、7、8 那側高 1～3 mm）。

    python board_height.py            # 等棋盤清空（最多 15 分鐘）再量
    python board_height.py --now      # 不等，直接量（棋盤要是空的）
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import find_pieces                       # noqa: E402
from common import OrbbecCamera          # noqa: E402

OUT = os.path.join(HERE, "board_dz.json")


def main():
    plan = json.load(open(os.path.join(HERE, "cell_plan.json"), encoding="utf-8"))
    cam = OrbbecCamera(depth=True)
    try:
        if "--now" not in sys.argv:
            t0, empty = time.time(), 0
            print("等棋盤清空…", flush=True)
            while time.time() - t0 < 900:
                b = find_pieces.read_board(cam.grab(skip=1, timeout=15), plan)
                empty = empty + 1 if (b is not None and all(c == " " for c in b)) else 0
                if empty >= 3:
                    break
                time.sleep(1.5)
            else:
                print("逾時：棋盤沒有清空", flush=True)
                return
            time.sleep(3.0)                                  # 等手離開
        ds = [cam.grab(depth=True, skip=1, timeout=15)[1] for _ in range(10)]
    finally:
        cam.close()
    d = np.median(np.stack(ds), axis=0)
    depth = {}
    for i in range(9):
        u, v = [int(a) for a in plan[str(i)]["px"]]
        yy, xx = np.mgrid[-30:31, -30:31]
        patch = d[v - 30:v + 31, u - 30:u + 31]
        m = (np.hypot(xx, yy) < 30) & (patch > 0)
        depth[i] = float(np.median(patch[m]))
    dz = {str(i): round(depth[4] - depth[i], 1) for i in range(9)}         # 正＝比格 4 高（離相機近）
    json.dump(dz, open(OUT, "w", encoding="utf-8"), indent=1)
    print("棋盤面深度（mm）：", " ".join("%d:%.1f" % (i, depth[i]) for i in range(9)), flush=True)
    print("比格 4 高（mm）：", " ".join("%d:%+.1f" % (i, float(dz[str(i)])) for i in range(9)), flush=True)
    print("已存 %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
