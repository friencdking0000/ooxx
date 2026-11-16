"""相機被碰歪之後的自動對齊（不用重做手眼校正）。
原理：棋盤跟手臂底座固定在同一個檯面上，手眼校正時 9 格中心在影像上的位置記在 cell_plan.json。
相機被碰動後，用現在畫面的棋盤格線求一個「現在影像 → 校正時影像」的相似變換（平移＋旋轉＋縮放），
存成 Arena/cam_align.json；相機子程式每張影像（彩色、深度）都先套這個變換再輸出，後面的程式完全不用改。
限制：只對桌面這個平面準；棋盤本身若被移動（相對手臂）這個方法會跟著錯，那就要重做手眼校正。

    from align import estimate, load, save, compose
    python align.py <影像>      # 印出估計的變換
"""
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(os.path.dirname(HERE), "Arena", "cam_align.json")
I = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def load(strict=False):
    """沒有檔案＝不修正（單位矩陣）。strict=True 時檔案壞掉會丟例外（相機子程式用：保留上一個有效值）。"""
    if not os.path.exists(FILE):
        return I.copy()
    try:
        with open(FILE, encoding="utf-8") as f:
            return np.array(json.load(f)["M"], float).reshape(2, 3)
    except (OSError, ValueError, KeyError):
        if strict:
            raise
        return I.copy()


def save(M, note=""):
    """先寫暫存檔再換名：相機子程式隨時會讀，不能讓它讀到寫一半的檔。"""
    from common import atomic_write
    atomic_write(FILE, json.dumps({"M": np.asarray(M, float).tolist(), "t": time.time(), "note": note}))


def compose(M_new, M_old):
    """先套 M_old 再套 M_new。"""
    A = np.vstack([M_new, [0, 0, 1]]) @ np.vstack([M_old, [0, 0, 1]])
    return A[:2]


def estimate(img):
    """回傳 (M, 偏移量 px, 殘差 px)；棋盤格線找不到或不合理回 None。img 是目前（可能已套過舊變換）的畫面。"""
    from board import find_grid
    plan = json.load(open(os.path.join(HERE, "cell_plan.json"), encoding="utf-8"))
    ref = np.array([plan[str(i)]["px"] for i in range(9)], np.float32)
    try:
        cur = np.array(find_grid(img)["cells"], np.float32)
    except Exception:
        return None
    if cur.shape != (9, 2):
        return None
    M, _ = cv2.estimateAffinePartial2D(cur, ref, method=cv2.LMEDS)
    if M is None:
        return None
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    res = float(np.hypot(*((cur @ M[:, :2].T + M[:, 2]) - ref).T).max())
    shift = float(np.hypot(*(cur - ref).T).mean())
    if not (0.93 <= scale <= 1.07) or abs(rot) > 6 or res > 8 or shift > 160:
        return None                                    # 不像是「相機被碰了一下」：不要亂套
    return M, shift, res


def estimate_from_pieces(img, min_n=3, ps=None):
    """棋盤上有棋子、格線被蓋住時的粗對齊（只有平移）：每顆棋子中心相對格子中心的位移取中位數。
    前提是棋子大致放在格子正中央：手臂放的會在正中央，人放的有誤差，用中位數抵銷。
    2026-11-15 實測：相機被碰後 9 顆棋子（手臂放的、人放的都一樣）一致往左偏約 14 px，底座也同方向偏 → 是相機動了。
    回傳 (T, (dx, dy), 顆數, 離散程度 px) 或 None（棋子太少、各顆偏的方向不一致、或偏太多不像相機被碰）。"""
    import find_pieces
    plan = json.load(open(os.path.join(HERE, "cell_plan.json"), encoding="utf-8"))
    if ps is None:                                     # 呼叫端已經找過棋子就直接用，不要再跑一次
        try:
            ps = find_pieces.pieces(img, plan)
        except Exception:
            return None
    if len(ps) < min_n:
        return None
    d = np.array([[p["px"][0] - plan[str(p["cell"])]["px"][0], p["px"][1] - plan[str(p["cell"])]["px"][1]] for p in ps], float)
    med = np.median(d, axis=0)
    spread = float(np.median(np.hypot(*(d - med).T)))
    if spread > 8.0 or np.hypot(*med) > 45.0:
        return None
    T = np.array([[1.0, 0.0, -med[0]], [0.0, 1.0, -med[1]]])
    return T, (float(med[0]), float(med[1])), len(ps), spread


def apply(M, img, depth=False):
    if M is None or np.allclose(M, I, atol=1e-6):
        return img
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST if depth else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


if __name__ == "__main__":
    im = cv2.imread(sys.argv[1])
    r = estimate(im)
    print("找不到合理的棋盤格線" if r is None else "平均偏 %.1f px，對齊後殘差 %.1f px\n%s" % (r[1], r[2], np.round(r[0], 4)))
