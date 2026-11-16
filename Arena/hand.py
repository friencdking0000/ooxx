"""手掌偵測（花式交接：手臂把棋子放到使用者攤開的手掌上）。

用 Gemini 2 對齊到彩色的深度（mm）：
1. 先拍「參考深度」（沒伸手時，手臂停在停放點）；
2. 之後比參考近 25 mm 以上的區域＝新出現的東西（手＋手臂），取最大一塊；
3. 掌心＝那塊區域裡離邊緣最遠的點（距離轉換最大值；手掌比前臂寬，所以會落在手掌上）；
4. 掌心附近的深度中位數 → 手掌離棋盤多高；像素＋深度 → B 座標。

只回報「看到什麼」，能不能放、要不要等由呼叫端（server／robot）判斷。
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Calib"))
from world import px_to_arm                  # noqa: E402

D_BOARD = 540.0                               # 相機到棋盤面（mm，2026-11-14）
DIFF_MM = 25.0                                # 比參考近這麼多才算新物體
ZONE = (400, 110, 900, 620)                   # 掌心要在這個影像範圍（棋盤及周邊；料區在 x≈930 以外）
H_MIN, H_MAX = 30.0, 100.0                    # 手掌離棋盤面的高度（mm）：太低會壓到棋子、太高手臂要抬太高靠近相機
PALM_HALF_MM = (22.0, 75.0)                   # 掌心到邊緣的距離（手掌半寬）合理範圍


def reference(depths):
    """多張深度取中位數當參考（0＝無效值會被忽略）。"""
    st = np.stack([np.where(d > 0, d, np.nan) for d in depths])
    ref = np.nanmedian(st, axis=0)
    return np.nan_to_num(ref, nan=0.0).astype(np.float32)


def detect(depth, ref, cal, exclude=None):
    """回傳 (palm, reason)。palm＝{px, r_px, depth, h, xy, half_mm}；沒看到時 palm=None、reason 說明原因。"""
    valid = (depth > 0) & (ref > 0)
    mask = (valid & (ref - depth > DIFF_MM)).astype(np.uint8)
    if exclude is not None:
        mask[exclude > 0] = 0                       # 手臂自己（交接時停在手旁邊）
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3000:
            continue
        comp = (labels == i).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        x0, y0, x1, y1 = ZONE
        dz = np.zeros_like(dist)
        dz[y0:y1, x0:x1] = dist[y0:y1, x0:x1]       # 掌心只在交接區裡找
        r = float(dz.max())
        if r <= 0:
            continue
        v, u = np.unravel_index(int(np.argmax(dz)), dz.shape)
        if best is None or r > best[0]:
            best = (r, u, v, comp)
    if best is None:
        return None, "交接區裡沒看到手"
    r, u, v, comp = best
    yy, xx = np.ogrid[:depth.shape[0], :depth.shape[1]]
    disk = ((xx - u) ** 2 + (yy - v) ** 2 <= (0.5 * r) ** 2) & (comp > 0) & (depth > 0)
    if disk.sum() < 30:
        return None, "手掌深度讀不到"
    d = float(np.median(depth[disk]))
    h = D_BOARD - d
    half = r * d / cal["fx"]
    palm = dict(px=(int(u), int(v)), r_px=r, depth=d, h=h, half_mm=half,
                xy=px_to_arm(cal, (u, v), depth=d))
    if not (PALM_HALF_MM[0] <= half <= PALM_HALF_MM[1]):
        return palm, "看到的東西不像手掌（半寬 %.0f mm）" % half
    if h < H_MIN:
        return palm, "手太低了，請抬高一點"
    if h > H_MAX:
        return palm, "手太高了，請放低一點"
    return palm, None


def same_place(p, q, tol_xy=15.0, tol_h=15.0):
    return p is not None and q is not None and float(np.hypot(*(np.asarray(p["xy"]) - q["xy"]))) <= tol_xy \
        and abs(p["h"] - q["h"]) <= tol_h


def draw(img, palm, reason=None):
    """在彩色影像上畫交接區與掌心（給網頁看）。"""
    out = img.copy()
    x0, y0, x1, y1 = ZONE
    cv2.rectangle(out, (x0, y0), (x1, y1), (87, 119, 217), 2)
    if palm:
        col = (60, 180, 75) if reason is None else (40, 40, 220)
        cv2.circle(out, palm["px"], int(palm["r_px"]), col, 3)
        cv2.circle(out, palm["px"], 6, col, -1)
        cv2.putText(out, "h=%.0fmm" % palm["h"], (palm["px"][0] + 10, palm["px"][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    return out
