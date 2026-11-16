"""料區自動偵測（使用者：放料區要可以在各個地方）：棋盤外桌面上的每一疊棋子，用彩色找位置、深度量高度、環形取樣判 O／X。

    python stocks.py        # 拍一張、印出每一疊（位置、顆數、O/X、吸盤接觸高度），存 out/stocks.jpg

回傳的每一疊：
  kind   "O"／"X"（看最上面那顆）
  px     影像中心；xy  B 座標中心
  n      顆數（(桌面深度 − 頂面深度) / 每顆厚度，四捨五入）
  top    頂面深度 mm；table  旁邊桌面深度 mm；th  每顆厚度 mm
  z_touch  B 吸盤碰到頂面時的 Z（由深度直接算，不靠計數推算）
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from world import load, px_to_arm      # noqa: E402

Z_TOUCH_BOARD = -154.82                  # B 碰到格 4 標準棋子頂面
STD_TOP_DEPTH = 540.0 - 3.7              # 格 4 標準棋子頂面離相機
PIECE_MM = 3.9                           # 平均厚度（貼膠帶後 3.6～4.4）
DARK_V_MAX = 130                         # 黑底（透明膠帶反光會變亮到 ~110）
GREEN = ((40, 25, 40), (95, 255, 255))   # 綠色（反光時飽和度會掉到 ~30）
DISC_R_MM = 41.0                         # 霍夫量到的圓盤外緣半徑（2026-11-15 棋盤上 52.0～53.4 px @537 mm）
PIECE_R_MM = 42.0
SEARCH_PX = 38.0                         # 霍夫圓心離候選點（顏色／深度／模型的峰）最多這麼遠（0.7 個半徑；疊歪 15 px 也找得到）
HOUGH_R_TOL_PX = 3.5                     # 霍夫半徑跟預期差超過這麼多不採用（被直邊撐大、或看到疊的側壁）
PIECE_HSV = [GREEN, ((0, 0, 0), (180, 255, DARK_V_MAX))]   # 棋子（本體＋符號＋背面）的顏色
SYMBOL_HSV = [GREEN]                     # 圈／叉符號的顏色
SCALE = 1.0                              # 棋子大小／原本那批：px 門檻（距離轉換峰、間距）跟著縮放
_TPL = None                              # 棋子校正存的圈／叉樣板（piece_templates.npz）

# 換了一批棋子：Calib/piece_profile.json（piece_calib.py 量的）覆蓋上面的值
import piece_profile                     # noqa: E402
PROFILE = piece_profile.load()


def _ranges(rs):
    return [tuple(tuple(int(x) for x in b) for b in r) for r in rs]


def make_tpl(o, x):
    """圈的樣板直接用；叉的樣板轉 0～84 度（每 6 度）各一張，比對時取最像的。"""
    c = ((x.shape[1] - 1) / 2, (x.shape[0] - 1) / 2)
    return {"O": [o], "X": [cv2.warpAffine(x, cv2.getRotationMatrix2D(c, a, 1.0), x.shape[::-1]) for a in range(0, 90, 6)]}


if PROFILE:
    PIECE_MM = float(PROFILE.get("piece_mm", PIECE_MM))
    DISC_R_MM = float(PROFILE.get("disc_r_mm", DISC_R_MM))
    PIECE_R_MM = float(PROFILE.get("piece_r_mm", DISC_R_MM + 1.0))
    SCALE = DISC_R_MM / 41.0
    SEARCH_PX *= SCALE
    PIECE_HSV = _ranges(PROFILE.get("piece_hsv", PIECE_HSV))
    SYMBOL_HSV = _ranges(PROFILE.get("symbol_hsv", SYMBOL_HSV))
    if PROFILE.get("templates") and os.path.exists(os.path.join(PROFILE["_dir"], PROFILE["templates"])):
        _z = np.load(os.path.join(PROFILE["_dir"], PROFILE["templates"]))
        _TPL = make_tpl(_z["O"], _z["X"])


def _odd(k):
    k = int(round(k))
    return k + 1 - k % 2


def in_ranges(hsv, ranges):
    m = cv2.inRange(hsv, *ranges[0])
    for r in ranges[1:]:
        m = cv2.bitwise_or(m, cv2.inRange(hsv, *r))
    return m


def piece_mask(hsv):
    return in_ranges(hsv, PIECE_HSV)


def symbol_mask(hsv):
    return in_ranges(hsv, SYMBOL_HSV)


TPL_N = 48                               # 樣板邊長（px）


def tpl_crop(sym, u, v, r):
    """符號遮罩（0／255）以 (u, v) 為中心、半徑 r 的圓盤 → TPL_N×TPL_N 浮點（0～1），0.85 半徑以外清 0。"""
    s = TPL_N / (2.0 * r)
    M = np.float32([[s, 0, TPL_N / 2 - s * u], [0, s, TPL_N / 2 - s * v]])
    t = cv2.warpAffine(sym.astype(np.float32) / 255.0, M, (TPL_N, TPL_N), flags=cv2.INTER_LINEAR)
    yy, xx = np.mgrid[:TPL_N, :TPL_N] - (TPL_N - 1) / 2
    t[np.hypot(xx, yy) > 0.85 * TPL_N / 2] = 0
    return t


def sym_crop(img, u, v, r):
    """影像上 (u, v) 那顆的符號樣子（tpl_crop）。只轉附近的 HSV。"""
    R = int(r) + 3
    H, W = img.shape[:2]
    x0, y0 = max(0, int(u) - R), max(0, int(v) - R)
    roi = img[y0:min(H, int(v) + R + 1), x0:min(W, int(u) + R + 1)]
    return tpl_crop(symbol_mask(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)), u - x0, v - y0, r)


def _ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / d if d > 1e-6 else 0.0


def tpl_kind(img, u, v, r, tpl=None):
    """用棋子校正的樣板判圈／叉：沒符號（翻面）回「?」，分不清回 None（交給顏色規則）。"""
    tpl = tpl or _TPL
    if tpl is None:
        return None
    t = sym_crop(img, u, v, r)
    if t.mean() < 0.03:
        return "?"
    so, sx = max(_ncc(t, a) for a in tpl["O"]), max(_ncc(t, a) for a in tpl["X"])
    if max(so, sx) < 0.3 or abs(so - sx) < 0.08:
        return None
    return "O" if so > sx else "X"


def disc_peaks(img, min_r=38, sep=70):
    """棋子顏色圓盤的中心（距離轉換峰值）與半徑。min_r、sep 是原本那批棋子的 px，照 SCALE 縮放。"""
    min_r, sep = min_r * SCALE, sep * SCALE
    disc = cv2.morphologyEx(piece_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    dist = cv2.distanceTransform(disc, cv2.DIST_L2, 5)
    peaks = (dist >= min_r) & (dist == cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(71 * SCALE),) * 2)))
    ys, xs = np.nonzero(peaks)
    out = []
    for x, y in sorted(zip(xs, ys), key=lambda q: -dist[q[1], q[0]]):
        if all(np.hypot(x - a, y - b) > sep for a, b, _ in out):
            out.append((float(x), float(y), float(dist[y, x])))
    return out


def disc_r_px(fx, depth):
    return DISC_R_MM * fx / depth


def hough_discs(img, u, v, r_exp, search):
    """(u, v) 附近 search px 內、半徑跟預期差 HOUGH_R_TOL_PX 內的霍夫圓 [(u, v, r, 圓周完整度), …]。
    圓周完整度＝圓周上有邊緣（Canny，徑向 ±2 px 內）的比例，0～1。"""
    R = int(search + 1.3 * r_exp)
    H, W = img.shape[:2]
    x0, y0 = max(0, int(u) - R), max(0, int(v) - R)
    roi = img[y0:min(H, int(v) + R + 1), x0:min(W, int(u) + R + 1)]
    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 1.2)
    c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=4, param1=120, param2=0.80,
                         minRadius=int(r_exp - HOUGH_R_TOL_PX), maxRadius=int(np.ceil(r_exp + HOUGH_R_TOL_PX)))
    if c is None:
        return []
    edge = cv2.dilate(cv2.Canny(gray, 60, 120), np.ones((5, 5), np.uint8))   # 膨脹 2 px＝徑向 ±2 px 內有邊緣就算
    th = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    out = []
    for a, b, r in c[0]:
        if abs(r - r_exp) > HOUGH_R_TOL_PX or np.hypot(a + x0 - u, b + y0 - v) > search:
            continue
        xs = np.round(a + r * np.cos(th)).astype(int); ys = np.round(b + r * np.sin(th)).astype(int)
        ok = (xs >= 0) & (ys >= 0) & (xs < edge.shape[1]) & (ys < edge.shape[0])
        if ok.mean() < 0.5:                              # 大半個圓在畫面／搜尋框外：量不出完整度
            continue
        cover = float(np.count_nonzero(edge[ys[ok], xs[ok]])) / ok.sum()   # 只算看得到的那段，靠邊的圓不吃虧
        out.append((float(a + x0), float(b + y0), float(r), cover))
    return out


def piece_center(img, u0, v0, r_exp, search=SEARCH_PX):
    """棋子（料疊最上面那顆）的中心＝霍夫圓的圓心。只看圓，不看綠色（綠色只用來分 O／X，使用者 2026-11-15）。
    - 只收半徑跟這個深度預期的外緣半徑（disc_r_px）差 HOUGH_R_TOL_PX 內的圓：直邊撐大的圓、疊的外輪廓都不收。
    - 候選點附近找到好幾個圓（同一顆的重複偵測，或料疊歪了、上下兩顆的圓都看得到）時，取圓周最完整的那個：
      最上面那顆不會被遮，圓周整圈都看得到；下面那顆被上面那顆蓋住一大段（歪 15 px 時約 45% 看不到）。
    找不到圓回 None（呼叫端沿用候選點；放完量、偏了挪正會修）。"""
    cs = hough_discs(img, u0, v0, r_exp, search)
    if not cs:
        return None
    best = max(cs, key=lambda c: c[3])
    return best[0], best[1]


def kind_of(img, u, v, r):
    """先照給的中心判；判成 X 時再用「綠色重心」當中心判一次：疊歪的料疊，最上面那顆的中心會跟整疊輪廓中心差到 30 px，
    O 的綠環落進內圈就被當成 X（2026-11-15：貼膠帶的 O 疊被當 X 疊去吸）。X 不管中心放哪都不會有整圈綠，所以只會把錯的 X 改回 O。"""
    try:                                                 # 訓練好的模型優先（不怕光線變化）；附近沒偵測到才用下面的顏色規則
        import ox_model
        km = ox_model.kind_at(img, u, v)
        if km is not None:
            return km
    except Exception:
        pass
    kt = tpl_kind(img, u, v, r)                          # 棋子校正學到的樣子（換了一批棋子時）
    if kt is not None:
        return kt
    k = _kind_at(img, u, v, r)
    if k != "X":
        return k
    R = int(r)
    y0, x0 = max(0, int(v) - R), max(0, int(u) - R)
    g = symbol_mask(cv2.cvtColor(img[y0:int(v) + R + 1, x0:int(u) + R + 1], cv2.COLOR_BGR2HSV))
    ys, xs = np.nonzero(g)
    if len(xs) < 30:
        return k
    for _ in range(2):                                   # 重心 → 以重心為中心再取一次（把被裁掉的那半圈補回來）
        cu, cv_ = x0 + xs.mean(), y0 + ys.mean()
        y0, x0 = max(0, int(cv_) - R), max(0, int(cu) - R)
        g = symbol_mask(cv2.cvtColor(img[y0:int(cv_) + R + 1, x0:int(cu) + R + 1], cv2.COLOR_BGR2HSV))
        ys, xs = np.nonzero(g)
        if len(xs) < 30:
            return k
    cu, cv_ = x0 + xs.mean(), y0 + ys.mean()
    if 5 < np.hypot(cu - u, cv_ - v) < 0.8 * r and _kind_at(img, cu, cv_, r) == "O":
        return "O"
    return k


def _kind_at(img, u, v, r):
    """O／X：0.4～0.65 半徑的環切 12 片，每片有沒有綠。O 的綠環幾乎 12 片都有；X 的叉只在約 4 片（每條叉經過兩次）。
    不明確時看中心：X 中心是綠、O 中心是黑洞（反光太亮就不靠中心）。整顆幾乎沒綠＝「?」（例如包鋁箔）。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = symbol_mask(hsv)
    H, W = green.shape
    sec = np.zeros(12)
    cnt = np.zeros(12)
    for k in np.linspace(0.42, 0.65, 6):
        for a in np.linspace(0, 2 * np.pi, 144, endpoint=False):
            x, y = int(round(u + k * r * np.cos(a))), int(round(v + k * r * np.sin(a)))
            if 0 <= y < H and 0 <= x < W:
                j = int(a / (2 * np.pi) * 12) % 12
                cnt[j] += 1
                sec[j] += green[y, x] > 0
    frac = sec / np.maximum(cnt, 1)
    yy, xx = np.mgrid[-int(r):int(r) + 1, -int(r):int(r) + 1]
    patch = green[max(0, int(v) - int(r)):int(v) + int(r) + 1, max(0, int(u) - int(r)):int(u) + int(r) + 1]
    if patch.size == 0 or (patch > 0).mean() < 0.05:
        return "?"
    full = int((frac >= 0.35).sum())
    if full >= 10:
        return "O"
    # 內圈（半徑 0.32 以內）：X 的叉一定穿過這裡，O 是黑洞。環被膠帶反光遮掉一半的 O 會被上面的規則當成 X（2026-11-15 實測）
    ri = max(3, int(0.32 * r))
    y0, x0 = int(round(v)), int(round(u))
    gi = green[max(0, y0 - ri):y0 + ri + 1, max(0, x0 - ri):x0 + ri + 1]
    hi = hsv[max(0, y0 - ri):y0 + ri + 1, max(0, x0 - ri):x0 + ri + 1]
    if gi.size:
        inner = float((gi > 0).mean())
        glare = float(((hi[..., 2] > 140) & (hi[..., 1] < 45)).mean())
        if inner < 0.08 and glare < 0.5 and full >= 3:
            return "O"
    if full <= 7:
        return "X"
    c = np.median(hsv[int(v) - 4:int(v) + 5, int(u) - 4:int(u) + 5].reshape(-1, 3), axis=0)
    if c[2] > 140 and c[1] < 45:                          # 中心反光，看不出來：環比較滿就當 O
        return "O" if full >= 9 else "X"
    if PROFILE:
        return "X" if symbol_mask(np.uint8(c).reshape(1, 1, 3))[0, 0] else "O"
    return "X" if (40 <= c[0] <= 95 and c[1] >= 40 and c[2] >= 60) else "O"


def _ring_median(d, u, v, r0, r1):
    yy, xx = np.mgrid[-r1:r1 + 1, -r1:r1 + 1]
    rr = np.hypot(xx, yy)
    y0, x0 = int(v) - r1, int(u) - r1
    patch = np.zeros(rr.shape, np.float32)
    ys, xs = slice(max(0, y0), min(d.shape[0], y0 + rr.shape[0])), slice(max(0, x0), min(d.shape[1], x0 + rr.shape[1]))
    patch[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = d[ys, xs]
    m = (rr >= r0) & (rr < r1) & (patch > 0)
    return float(np.median(patch[m])) if m.sum() > 20 else None


def detect(img, depth, cal, board_cells_px, board_margin=80):
    """棋盤外每一疊：候選＝彩色圓盤峰值 ∪ 深度凸起峰值；用「緊鄰四周的桌面」量高度（≥2.5 mm 才算），
    頂面要看起來是棋子（黑／綠佔多數），O／X 用該深度下的真實棋子半徑做環形取樣。"""
    d = depth.astype(np.float32)
    H, W = img.shape[:2]
    hull = cv2.convexHull(np.asarray(board_cells_px, np.float32))
    cands = [(u, v) for u, v, _ in disc_peaks(img, min_r=30)]
    valid = d > 0
    fill = np.where(valid, d, np.median(d[valid]))
    bg = cv2.dilate(fill, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (181, 181)))
    m = (((bg - fill) > 2.5) & valid).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    pk = (dist >= 30 * SCALE) & (dist == cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(71 * SCALE),) * 2)))
    cands += [(float(x), float(y)) for y, x in zip(*np.nonzero(pk))]
    try:                                                 # 模型偵測到的棋子也當候選（顏色門檻在新光線下漏抓時補上）
        import ox_model
        cands += [(float(a), float(b)) for a, b, _, _, _ in ox_model.detect(img)]
    except Exception:
        pass
    piece_col = piece_mask(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    out, used = [], []
    for u, v in cands:
        if any(np.hypot(u - a, v - b) < 55 * SCALE for a, b in used):
            continue
        if u < 70 or v < 70 or u > W - 70 or v > H - 70:
            continue
        if cv2.pointPolygonTest(hull, (u, v), True) > -board_margin:
            continue
        top0 = _ring_median(d, u, v, 0, 14) or STD_TOP_DEPTH   # 只用來估霍夫的預期半徑；候選點上剛好是深度洞也照找
        hc = piece_center(img, u, v, disc_r_px(cal["fx"], top0))   # 最上面那顆的圓盤中心＝霍夫圓心
        if hc is not None:
            u, v = hc
        if any(np.hypot(u - a, v - b) < 55 * SCALE for a, b in used):
            continue
        top = _ring_median(d, u, v, 0, 14)
        if top is None:
            continue
        r = PIECE_R_MM * cal["fx"] / top                    # 這個深度下棋子在影像上的半徑
        table = _ring_median(d, u, v, int(r) + 15, int(r) + 40)
        if table is None or not (1.5 <= table - top <= 10 * PIECE_MM + 4):   # 單顆平躺（4 mm）深度常只量到 2 mm 多，2.5 會漏（2026-11-15）；外觀檢查在下面把關
            continue
        yy, xx = np.mgrid[-int(r):int(r) + 1, -int(r):int(r) + 1]
        inside = np.hypot(xx, yy) <= 0.7 * r
        patch = piece_col[max(0, int(v) - int(r)):int(v) + int(r) + 1, max(0, int(u) - int(r)):int(u) + int(r) + 1]
        if patch.shape != inside.shape or (patch[inside] > 0).mean() < 0.5:
            continue                                        # 頂面不像棋子（袋子、耳機盒、膠帶台、鋁箔…）
        used.append((u, v))
        n = max(1, int(round((table - top) / PIECE_MM)))
        kind = kind_of(img, u, v, r)
        out.append(dict(kind=kind, px=(u, v), r=r, top=top, table=table, n=n, th=(table - top) / n,
                        xy=px_to_arm(cal, (u, v), depth=top), z_touch=Z_TOUCH_BOARD + (STD_TOP_DEPTH - top)))
    return out


def main():
    import json
    from common import OrbbecCamera
    cal = load("B")
    plan = json.load(open(os.path.join(HERE, "cell_plan.json"), encoding="utf-8"))
    cam = OrbbecCamera(depth=True)
    try:
        img = cam.grab(skip=3, timeout=15)
        ds = [cam.grab(depth=True, skip=1, timeout=15)[1] for _ in range(5)]
    finally:
        cam.close()
    d = np.median(np.stack(ds), axis=0)
    vis = img.copy()
    for s in detect(img, d, cal, [plan[str(i)]["px"] for i in range(9)]):
        print("%s 疊：%d 顆（每顆 %.1f mm）影像 (%.0f,%.0f) → B (%.1f, %.1f)，接觸 Z=%.1f" % (
            s["kind"], s["n"], s["th"], *s["px"], *s["xy"], s["z_touch"]))
        cv2.circle(vis, (int(s["px"][0]), int(s["px"][1])), int(s["r"]), (0, 0, 255), 3)
        cv2.putText(vis, "%s x%d" % (s["kind"], s["n"]), (int(s["px"][0]) - 30, int(s["px"][1]) - int(s["r"]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    cv2.imwrite(os.path.join(HERE, "out", "stocks.jpg"), vis)


if __name__ == "__main__":
    main()
