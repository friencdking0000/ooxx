"""找棋盤上每顆棋子的實際中心與「可吸取點」（吸盤整個落在綠色符號或黑底平面內，不跨交界）。

    python find_pieces.py          # 印每顆的格號、種類、中心（影像／B 座標）、吸取偏移；存 out/pieces.jpg
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stocks                          # noqa: E402
from world import load, px_to_arm      # noqa: E402

CUP_R = 13          # 吸盤半徑（px，棋盤面高度）
MARGIN = 1
PIECE_TOP = stocks.STD_TOP_DEPTH
R_BOARD_PX = stocks.disc_r_px(load("B")["fx"], PIECE_TOP)   # 棋盤上棋子圓盤外緣的預期半徑（約 52.6 px）
# 棋子貼了透明膠帶會反光：黑變亮（V 到 ~110）、綠變淡（S 到 ~40）；棋盤木頭 V ~200、色相偏黃（<40）
DARK_V_MAX = 130
GREEN_S_MIN = 40


def pieces(img, plan, cells=None, blank_ok=False):
    """cells：目前 9 格中心（影像 px，例如 board.find_grid 的結果）；沒給就用校正時的 cell_plan。
    blank_ok：看不出 O／X 的棋子（翻面、背面朝上）也算，kind 為「?」——但要找得到大小對的霍夫圓，手或陰影不算。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S = stocks.SCALE                                    # 棋子大小／原本那批（棋子校正量的）
    if stocks.PROFILE:                                  # 換了一批棋子：顏色用棋子校正量的
        green, body = stocks.symbol_mask(hsv), stocks.piece_mask(hsv)
    else:
        green = cv2.inRange(hsv, (40, GREEN_S_MIN, 50), (95, 255, 255))
        body = cv2.bitwise_or(green, cv2.inRange(hsv, (0, 0, 0), (180, 255, DARK_V_MAX)))
    disc = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    dist = cv2.distanceTransform(disc, cv2.DIST_L2, 5)
    # 每顆棋子中心是距離轉換的一個峰（半徑約 55 px）；膠帶線寬約 25 px，峰值只有 ~12，不會被當成棋子
    peaks = (dist >= 38 * S) & (dist == cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stocks._odd(71 * S),) * 2)))
    ys, xs = np.nonzero(peaks)
    centers = []
    for x, y in sorted(zip(xs, ys), key=lambda q: -dist[q[1], q[0]]):
        if all(np.hypot(x - cx, y - cy) > 70 * S for cx, cy in centers):
            centers.append((x, y))
    model_r = {}
    try:                                                 # 模型偵測到、但顏色門檻沒抓到的棋子也補進來（新光線）
        import ox_model
        for a, b, r_, _, _ in ox_model.detect(img):
            if all(np.hypot(a - cx, b - cy) > 55 * S for cx, cy in centers):
                centers.append((int(round(a)), int(round(b))))
                model_r[(int(round(a)), int(round(b)))] = float(r_)
    except Exception:
        pass
    out = []
    cells = ({i: np.array(c, float) for i, c in enumerate(cells)} if cells is not None
             else {i: np.array(plan[str(i)]["px"], float) for i in range(9)})
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (CUP_R + MARGIN) + 1,) * 2)
    H, W = img.shape[:2]
    for (u, v) in centers:
        u, v = float(u), float(v)
        rad = model_r.get((int(u), int(v)), float(dist[int(v), int(u)]))
        rad = rad if rad >= 30 * S else 53.0 * S
        cell = min(cells, key=lambda i: np.hypot(*(cells[i] - (u, v))))
        if np.hypot(*(cells[cell] - (u, v))) > 75:
            continue                                   # 不在棋盤格上（料區等）
        hc = stocks.piece_center(img, u, v, R_BOARD_PX)   # 圓盤中心＝霍夫圓心（跟料疊同一套；綠色只分 O／X）
        if hc is not None:
            u, v = hc
        if any(np.hypot(u - q["px"][0], v - q["px"][1]) < 55 * S for q in out):
            continue                                   # 兩個候選收斂到同一顆：只算一次
        cell = min(cells, key=lambda i: np.hypot(*(cells[i] - (u, v))))
        if np.hypot(*(cells[cell] - (u, v))) > 75:
            continue                                   # 霍夫把中心移到棋盤外（料疊邊緣等）：不算棋盤上的
        kind = stocks.kind_of(img, u, v, rad)          # O／X 用環形取樣（中心反光也不怕）
        if kind == "?" and not (blank_ok and hc is not None):
            continue                                   # 看不出 O／X：blank_ok（翻面）時要有霍夫圓才算，否則不算
        # 可吸取區：棋子內縮（吸盤半徑＋邊距），且吸盤圓整個在綠或整個在黑；只在棋子周圍的小框裡算
        R = int(rad) + CUP_R + MARGIN + 2
        x0, y0 = max(0, int(u) - R), max(0, int(v) - R)
        x1, y1 = min(W, int(u) + R + 1), min(H, int(v) + R + 1)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.circle(m, (int(round(u)) - x0, int(round(v)) - y0), int(rad - 2), 255, -1)
        g = green[y0:y1, x0:x1]
        inner = cv2.erode(m, ker)
        gz = cv2.erode(cv2.bitwise_and(g, m), ker)
        dz = cv2.erode(cv2.bitwise_and(cv2.bitwise_not(g), m), ker)
        ys, xs = np.nonzero(cv2.bitwise_and(inner, cv2.bitwise_or(gz, dz)))
        if len(xs) == 0:
            pick = None
        else:
            j = int(np.argmin((xs + x0 - u) ** 2 + (ys + y0 - v) ** 2))
            pick = (float(xs[j] + x0), float(ys[j] + y0))
        out.append(dict(cell=cell, kind=kind, px=(u, v), r=rad, pick=pick))
    return sorted(out, key=lambda p: p["cell"])


def read_board(img, plan):
    """以棋子為主讀盤：找每顆棋子實際中心 → 看中心是綠（X）還是黑（O）→ 歸到最近的格子。
    同一格有兩顆（手、手臂或棋子疊在一起）回傳 None（這張不可信）。格線找得到就用目前的格子中心。"""
    cells = None
    try:                                     # 格線找到、而且跟校正時的格子中心差不多（棋盤只是被輕推）才用
        from board import find_grid
        g = np.array(find_grid(img)["cells"], float)
        P = np.array([plan[str(i)]["px"] for i in range(9)], float)
        if np.hypot(*(g - P).T).max() < 45:
            cells = g
    except Exception:
        pass
    b = [" "] * 9
    for p in pieces(img, plan, cells):
        if b[p["cell"]] != " ":
            return None
        b[p["cell"]] = p["kind"]
    return b


def main():
    import json
    from common import Camera
    plan = json.load(open(os.path.join(HERE, "cell_plan.json"), encoding="utf-8"))
    cal = load("B")
    cam = Camera()
    try:
        img = cam.grab(skip=3, timeout=20)
    finally:
        cam.close()
    vis = img.copy()
    for p in pieces(img, plan):
        c = px_to_arm(cal, p["px"], depth=PIECE_TOP)
        line = "格 %d %s 中心 px (%.0f, %.0f) → B (%.1f, %.1f)" % (p["cell"], p["kind"], *p["px"], *c)
        if p["pick"]:
            q = px_to_arm(cal, p["pick"], depth=PIECE_TOP)
            line += "  吸取偏移 (%.1f, %.1f) mm" % tuple(q - c)
            cv2.circle(vis, tuple(int(a) for a in p["pick"]), CUP_R, (0, 0, 255), 2)
        else:
            line += "  ⚠ 找不到完整平面的吸取點"
        print(line)
        cv2.circle(vis, tuple(int(a) for a in p["px"]), int(p["r"]), (255, 0, 0), 2)
        cv2.putText(vis, "%d%s" % (p["cell"], p["kind"]), (int(p["px"][0]) - 20, int(p["px"][1]) - 30), 0, 1.0, (255, 0, 255), 3)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    cv2.imwrite(os.path.join(HERE, "out", "pieces.jpg"), vis)


if __name__ == "__main__":
    main()
