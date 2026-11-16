"""相機支架的禁入區，用「相機座標」（mm，相機主點正下方為原點，影像 +x／+y 方向）定義。

2026-11-15 用深度量（Arena/out/post_depth.png）：兩側都是「直立的柱子＋貼桌的腳座」，跟相機是同一個架子。
- 影像上那條往畫面外延伸的「橫桿」其實是柱子往上、透視拉長；舊檢查（plan_cells.hits_post）在影像上框矩形，
  把柱子從桌面到頂端的投影都框進去，再拿懸停高度的方盒投影去比，會把離柱子還有 10 公分的格 0、3、6 擋掉。
- 舊檢查還用舊的相機校正：架子相對手臂移了約 31 mm 之後就不準（可能放過真的會撞的位置）。
- 這裡改成比「水平距離」：柱子是直的，所以只看水平位置；手臂從上往下看是底座→吸盤的一條線段
  （MG400 的連桿都在 J1 方向的直立平面上），線段上每一點都要離支架夠遠。
  吸盤端要留方盒（半寬 22 mm，轉到斜角最壞 31 mm）＋餘裕；連桿段留連桿半寬＋餘裕。
- 支架位置每次開始都用深度重新量（measure），架子或相機會被碰動；量到之前 clearance 不能用。
"""
import json
import os
import time

import numpy as np

from world import load as _load

LAST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posts_last.json")   # 上次量到的支架（量不到時沿用）

_cal = _load("B")
CX, CY, FX = float(_cal["cx"]), float(_cal["cy"]), float(_cal["fx"])   # 原始影像（未對齊）的主點與焦距（跟 calib_*.json 同一份）
POSTS_CAM = None                                 # 支架外框（相機座標 mm：x0, x1, y0, y1）；measure() 量到後填入
BOX_CLEAR = 31.0 + 20.0                          # 吸盤端：方盒對角半寬＋餘裕
LINK_CLEAR = 25.0 + 20.0                         # 連桿：半寬＋餘裕
POST_REF = {"左側支架": (-214.0, 0.0), "右側支架": (286.0, 5.0)}   # 2026-11-15 深度量的柱子中心（相機座標 mm），只當搜尋範圍的中心
POST_FOOT = {"左側支架": (-56.0, 62.0), "右側支架": (-55.0, 72.0)}  # 腳座前後範圍（相對柱子中心 Y，量到的 2～98% 再加 5 mm）


def px_to_cam(px, depth):
    """原始影像像素＋深度 → 相機座標水平位置（mm）。px、depth 可以是陣列（逐元素）。"""
    return np.array([(px[0] - CX) * depth / FX, (px[1] - CY) * depth / FX])


def cam_to_px(cam, depth):
    """px_to_cam 的反函數：相機座標水平位置（mm）＋深度 → 原始影像像素。"""
    return np.asarray(cam, float) * FX / depth + [CX, CY]


def near_depth(ds):
    """幾張深度 → 每個像素取「第二近」的有效值（至少兩張量到才算，單張雜訊尖點不採用），不到兩張＝0。
    2026-11-16 新場地：支架高處隔一張才出現一次（其他張同一像素量到後面的桌面 ~540 mm），
    中位數會被桌面蓋掉、整根柱子不見。柱子是那裡最近的東西，取近的只會把它量大（禁入區更保守）。"""
    s = np.stack(ds).astype(np.float32)
    s[s <= 0] = np.inf
    s.sort(axis=0)
    m = s[1] if len(s) > 1 else s[0]
    m[~np.isfinite(m)] = 0.0
    return m


def measure(d, d_board, Minv):
    """用深度量兩側支架（相機座標 mm），存進 POSTS_CAM 並回傳；量不到回 None（呼叫端不要動手臂）。
    只取「桌面以上 60 mm」的點——直立柱子；料疊（6 顆約 23 mm）、棋子、腳座（約 40 mm）都比這矮——
    而且只在 POST_REF ±90 mm 內找（旁邊的袋子、鍵盤、人不算）；柱子要夠細（X 寬 < 45 mm）才採用。
    禁入框＝柱子中心往外 30 mm、往棋盤 12 mm（柱子＋腳座），前後照 POST_FOOT（腳座長度）。
    d 是對齊後（cam_align）的深度，Minv 是對齊的反仿射（對齊後 → 原始）：像素要先換回原始影像才能套相機內參
    （2026-11-15 沒換，柱子算近了 20 mm，把格子全擋掉）。"""
    global POSTS_CAM
    table = d_board + 12.0
    yy, xx = np.nonzero((d > 0) & (d < table - 60))
    z = d[yy, xx].astype(float)
    u = Minv[0, 0] * xx + Minv[0, 1] * yy + Minv[0, 2]
    v = Minv[1, 0] * xx + Minv[1, 1] * yy + Minv[1, 2]
    X, Y = px_to_cam((u, v), z)
    out = {}
    for name, (rx, ry) in POST_REF.items():
        k = (np.abs(X - rx) < 90) & (np.abs(Y - ry) < 90)
        if k.sum() < 300:
            print("%s：深度點太少（%d）" % (name, k.sum()), flush=True)
            return None
        cx, cy = float(np.median(X[k])), float(np.median(Y[k]))
        w = float(np.percentile(X[k], 95) - np.percentile(X[k], 5))
        if w > 45:
            print("%s：不像一根柱子（寬 %.0f mm，可能有東西擋在旁邊）" % (name, w), flush=True)
            return None
        sgn = 1.0 if cx > 0 else -1.0                              # 往外＝離棋盤
        a, b = cx + sgn * 30.0, cx - sgn * 12.0
        f0, f1 = POST_FOOT[name]
        out[name] = (min(a, b), max(a, b), cy + f0, cy + f1)
    POSTS_CAM = out
    from common import atomic_write
    atomic_write(LAST, json.dumps(dict(t=time.strftime("%Y-%m-%d %H:%M:%S"), posts=out), ensure_ascii=False, indent=1))
    return out


def use_last():
    """深度量不到時：載入上次量到的支架到 POSTS_CAM，回傳量的時間；沒有存檔回 None（呼叫端不要動手臂）。
    支架跟相機是同一個架子，相機座標不會變；架子被拆裝過就刪掉 posts_last.json 重量。"""
    global POSTS_CAM
    try:
        with open(LAST, encoding="utf-8") as f:
            s = json.load(f)
        POSTS_CAM = {k: tuple(float(x) for x in v) for k, v in s["posts"].items()}
        return s["t"]
    except (OSError, ValueError, KeyError):
        return None


def rect_dist(p, r):
    x0, x1, y0, y1 = r
    return float(np.hypot(max(x0 - p[0], 0.0, p[0] - x1), max(y0 - p[1], 0.0, p[1] - y1)))


def clearance(base_cam, tcp_cam):
    """手臂底座、吸盤（相機座標）→ (最近的支架, 餘裕 mm)；餘裕 < 0 就是會碰到。"""
    if POSTS_CAM is None:
        raise RuntimeError("支架還沒量（posts.measure），不能判斷會不會撞")
    base_cam, tcp_cam = np.asarray(base_cam, float), np.asarray(tcp_cam, float)
    best = (None, np.inf)
    for name, r in POSTS_CAM.items():
        m = rect_dist(tcp_cam, r) - BOX_CLEAR
        m = min(m, min(rect_dist(base_cam + t * (tcp_cam - base_cam), r) for t in np.linspace(0, 0.95, 20)) - LINK_CLEAR)
        if m < best[1]:
            best = (name, m)
    return best
