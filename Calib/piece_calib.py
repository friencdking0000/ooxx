"""棋子校正（桌面「棋子校正.bat」）：現場換了一批棋子（大小、厚度、顏色、圖案不同），讓系統重新認得圈／叉。手臂不會動。

步驟（照畫面提示做）：
  1. 棋盤清空 → Enter（拍空棋盤：量棋盤面深度，也確認木頭不會被當成棋子）
  2. 九格中央各放一顆、正面朝上：四個角和正中間放圈，其他四格放叉 → Enter
  3. 九顆都翻面（背面朝上；機器下棋時棋子是翻面的）→ Enter；不做輸入 s
  4. 印出量到的大小、厚度、顏色和認對幾顆 → 輸入 y 存成 Calib/piece_profile.json（舊的備份成 .bak）
存好要重啟對弈伺服器才生效（最後會問要不要直接跑一鍵部署）。樣本影像存在 Calib/out/piece_calib/<時間>/。

    python piece_calib.py                          # 互動（會先停掉對弈伺服器、直接開相機）
    python piece_calib.py --reset                  # 刪掉 piece_profile.json，回到原本那批棋子的設定
    python piece_calib.py --files 空.jpg 正.jpg [背.jpg] [--layout=OXOXOXOXO]   # 用存好的影像算（沒有深度就不量厚度）
"""
import json
import os
import shutil
import subprocess
import sys
import time

if "--verify" not in sys.argv:
    os.environ["PIECE_PROFILE"] = "0"      # 這支程式自己用預設值量；量好的設定在另一個程序（--verify）裡驗證
import cv2                                  # noqa: E402
import numpy as np                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import stocks                               # noqa: E402
from world import load as load_cal          # noqa: E402
import piece_profile                        # noqa: E402

LAYOUT = "OXOXOXOXO"                        # 格 1～9（四個角＋中間＝圈）：轉 90 度、左右翻都一樣，不怕看反
TPL_NAME = "piece_templates.npz"
OLD_R_MM = 41.0                             # 原本那批的外緣半徑、厚度（比對用）
OLD_MM = 3.9


def say(t=""):
    print(t, flush=True)


# ======================= 相機 =======================
class Cam:
    """直接開相機（對弈伺服器、cam_worker 要先停）。grab() → (對齊後彩色 3 張中位數, 對齊後深度 5 張中位數)。"""

    def __init__(self):
        from common import OrbbecCamera
        self.cam = OrbbecCamera(depth=True)

    def grab(self):
        import align
        M = align.load()
        imgs, ds = [], []
        for i in range(5):
            img, d = self.cam.grab(depth=True, skip=3 if i == 0 else 0, timeout=15)
            if i < 3:
                imgs.append(img)
            ds.append(d)
        img = np.median(np.stack(imgs), axis=0).astype(np.uint8)
        d = np.median(np.stack(ds), axis=0).astype(np.float32)
        return align.apply(M, img), align.apply(M, d, depth=True)

    def close(self):
        self.cam.close()


# ======================= 量 =======================
def cells_px():
    plan = json.load(open(os.path.join(HERE, "cell_plan.json"), encoding="utf-8"))
    return plan, np.array([plan[str(i)]["px"] for i in range(9)], float)


def live_cells(img):
    """空棋盤畫面 → 9 格中心：用當下的格線找（場地搬過、cell_plan.json 還沒重做也能校正棋子）；找不到才用 cell_plan.json。
    回傳 (9×2, 跟 cell_plan 平均差幾 px 或 None＝用的是 cell_plan)。"""
    _, ref = cells_px()
    try:
        from board import find_grid
        g = find_grid(img)
        # 木板比棋盤長時（2026-11-16 新場地：木板一路延伸到兩台底座），橫線會抓到畫面黑邊或底座螢幕：
        # 改成只在直膠帶的上下範圍內找（空棋盤才看得到整條直膠帶）
        x0, _, x1, _ = g["roi"]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        tape = np.zeros(gray.shape[0], bool)
        for x in g["lines_x"]:
            strip = gray[:, x - 4:x + 5].mean(axis=1)
            side = np.minimum(gray[:, x - 34:x - 25].mean(axis=1), gray[:, x + 25:x + 34].mean(axis=1))
            tape |= strip < side - 30                      # 比兩旁（木頭）暗很多＝膠帶
        tape = cv2.morphologyEx(tape.astype(np.uint8)[:, None], cv2.MORPH_CLOSE, np.ones((41, 1), np.uint8))[:, 0]
        n, lab, st, _ = cv2.connectedComponentsWithStats(tape[:, None])
        if n > 1:
            i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_HEIGHT]))
            ya, yb = st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT]
            if yb - ya > 300:                              # 整條直膠帶（約 3 格高）才採用
                g = find_grid(img, roi=(x0, max(0, ya - 10), x1, min(gray.shape[0], yb + 10)))
        g = np.array(g["cells"], float)
        if g.shape == (9, 2):
            return g, float(np.hypot(*(g - ref).T).mean())
    except Exception:
        pass
    return ref, None


def _ring_mean(img, u, v, r0, r1):
    H, W = img.shape[:2]
    R = int(r1) + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy)
    ys, xs = yy + int(round(v)), xx + int(round(u))
    ok = (rr >= r0) & (rr < r1) & (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    return img[ys[ok], xs[ok]].mean(axis=0) if ok.any() else None


def locate(E, F, dE, dF, cells, r0):
    """每格中央附近找一顆棋子：[(u, v, r) 或 None]。r0＝原本那批的預期半徑（px）；新棋子大小不知道，半徑放寬到 0.55～1.45 倍。
    霍夫圓要「圓內跟空棋盤差很多、圓外一圈跟空棋盤差不多」才算（整張亮度變了也不怕）；同一格好幾個同心圓（圈的符號環）取最外圈。"""
    lab = lambda im: cv2.cvtColor(cv2.GaussianBlur(im, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    D = lab(F) - lab(E)
    old = stocks.HOUGH_R_TOL_PX
    stocks.HOUGH_R_TOL_PX = 0.45 * r0
    out = []
    try:
        for c in cells:
            best = []
            for u, v, r, cover in stocks.hough_discs(F, c[0], c[1], r0, search=0.9 * r0):
                if cover < 0.35:
                    continue
                inner = _ring_mean(D, u, v, 0, 0.85 * r)
                outer = _ring_mean(D, u, v, 1.15 * r, 1.4 * r)
                if inner is None or outer is None:
                    continue
                lift = 0.0
                if dE is not None and dF is not None:               # 深度：圓內比空棋盤高
                    a, b = ring(dE, u, v, 0.5 * r), ring(dF, u, v, 0.5 * r)
                    lift = (a - b) if a and b else 0.0
                contrast = float(np.linalg.norm(inner - outer))
                if contrast > 12 or lift > 1.2:
                    best.append((u, v, r, cover))
            if not best:
                out.append(None)
                continue
            top = max(q[3] for q in best)
            u, v, r, _ = max((q for q in best if q[3] >= 0.8 * top), key=lambda q: q[2])
            out.append((u, v, r))
        # 半徑跟其他顆的中位數差超過 12%（抓到圈符號的內環、或外緣貼著膠帶看不清楚）：用中位數半徑在原位置附近重找
        rs = [f[2] for f in out if f is not None]
        med = float(np.median(rs)) if rs else r0
        stocks.HOUGH_R_TOL_PX = max(3.5, 0.08 * med)
        for i, f in enumerate(out):
            if f is not None and abs(f[2] - med) > 0.12 * med:
                cs = stocks.hough_discs(F, f[0], f[1], med, search=0.5 * med)
                out[i] = max(cs, key=lambda q: q[3])[:3] if cs else (f[0], f[1], med)
    finally:
        stocks.HOUGH_R_TOL_PX = old
    return out


def ring(d, u, v, r1, r0=0):
    return None if d is None else stocks._ring_median(d, u, v, int(r0), int(round(r1)))


def pct(a, q):
    return float(np.percentile(a, q))


def hsv_range(px, pad=(8, 30, 35), other=None):
    """一群像素（N×3 HSV）→ [[lo, hi], …]：2～98 百分位再加寬。沒什麼顏色（黑白灰）不看色相；色相跨 0（紅）拆兩段。
    other：要排除的另一群（符號範圍排除本體）。飽和度、亮度哪一個能把兩群分開（5～95 百分位不重疊），邊界就放在兩群中間。"""
    h, s, v = px[:, 0].astype(int), px[:, 1].astype(int), px[:, 2].astype(int)
    slo, shi = max(0, pct(s, 2) - pad[1]), 255                # 飽和度上限不設：擋不了木頭，暗色的飽和度又是雜訊
    vlo, vhi = max(0, pct(v, 2) - pad[2]), min(255, pct(v, 98) + pad[2])
    if other is not None:
        lims = [[slo, shi], [vlo, vhi]]
        for j, lim in zip((1, 2), lims):
            a, b = pct(px[:, j], 5), pct(px[:, j], 95)
            oa, ob = pct(other[:, j], 5), pct(other[:, j], 95)
            if ob < a:
                lim[0] = max(lim[0], (ob + a) / 2)
            elif oa > b:
                lim[1] = min(lim[1], (oa + b) / 2)
        (slo, shi), (vlo, vhi) = lims
    if pct(s, 70) < 50 or pct(v, 70) < 90:                     # 沒什麼顏色或很暗（黑）：色相是雜訊
        return [[[0, int(slo), int(vlo)], [180, int(shi), int(vhi)]]]
    slo = max(slo, pct(s, 2) - 15)                             # 有顏色的群：灰白反光（低飽和）不算
    wrap = ((h < 15) | (h > 165)).mean() > 0.2
    hh = (h + 90) % 180 if wrap else h
    lo, hi = pct(hh, 2) - pad[0], pct(hh, 98) + pad[0]
    if hi - lo >= 170:
        return [[[0, int(slo), int(vlo)], [180, int(shi), int(vhi)]]]
    if wrap:
        lo, hi = lo - 90, hi - 90
        if hi < 0:
            lo, hi = lo + 180, hi + 180
    rs = []
    for a, b in ([(lo, hi)] if lo >= 0 and hi <= 180 else
                 [(0, hi), (lo + 180, 180)] if lo < 0 else [(lo, 180), (0, hi - 180)]):
        rs.append([[int(max(0, a)), int(slo), int(vlo)], [int(min(180, b)), int(shi), int(vhi)]])
    return rs


def disc_pixels(hsv, u, v, r, k0, k1):
    H, W = hsv.shape[:2]
    R = int(r) + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    rr = np.hypot(xx, yy) / r
    ys, xs = yy + int(round(v)), xx + int(round(u))
    ok = (rr >= k0) & (rr < k1) & (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    return hsv[ys[ok], xs[ok]]


def widen(rs, board, r_px, others=()):
    """光線會變（白天／晚上、反光）：亮度上限盡量放寬（+60、+40、+20），只要空棋盤上「粗到像棋子」的區塊
    （距離轉換超過 0.4 個半徑；膠帶線很細不算）不增加超過棋盤的 0.5%。board＝空棋盤那塊的 HSV 影像。"""
    def frac(x):
        m = stocks.in_ranges(board, stocks._ranges(list(others) + x))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return float((cv2.distanceTransform(m, cv2.DIST_L2, 5) >= 0.4 * r_px).mean())
    base = frac(rs)
    for extra in (60, 40, 20):
        w = [[lo, [hi[0], hi[1], min(255, hi[2] + extra)]] for lo, hi in rs]
        if frac(w) <= base + 0.005:
            return w
    return rs


def colors(up, found, layout, back=None, found_b=None, empty=None, cells=None):
    """正面：本體／符號分兩群（k-means），本體＝圈中心那群。回傳 (piece_hsv, symbol_hsv, 說明, 問題)。
    empty、cells：空棋盤和九格中心，用來把顏色範圍放寬到不會把棋盤當成棋子為止。"""
    hsv = cv2.cvtColor(up, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    X = np.concatenate([disc_pixels(lab, *f, 0, 0.85) for f in found]).astype(np.float32)
    _, lbl, ctr = cv2.kmeans(X, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 5, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel()
    # 圈的中心（0.18 半徑內）是本體顏色；叉的中心是符號
    def center_lbl(kind):
        ps = [disc_pixels(lab, *f, 0, 0.18) for f, k in zip(found, layout) if k == kind]
        P = np.concatenate(ps).astype(np.float32)
        d = np.linalg.norm(P[:, None, :] - ctr[None], axis=2).argmin(axis=1)
        return np.bincount(d, minlength=2) / len(d)
    co, cx = center_lbl("O"), center_lbl("X")
    body = int(np.argmax(co))
    probs = []
    if cx[1 - body] < co[1 - body] + 0.2:
        probs.append("圈和叉的中心看起來差不多（圈中心該是底色、叉中心該是符號色）：圈叉是不是放反了，或符號不在中心？")
    hs = np.concatenate([disc_pixels(hsv, *f, 0, 0.85) for f in found])
    body_r = hsv_range(hs[lbl == body])
    sym_r = hsv_range(hs[lbl != body], other=hs[lbl == body])
    back_r = []
    if back is not None and found_b:
        hb = cv2.cvtColor(back, cv2.COLOR_BGR2HSV)
        back_r = hsv_range(np.concatenate([disc_pixels(hb, *f, 0, 0.85) for f in found_b if f is not None]))
    if empty is not None and cells is not None:
        x0, y0 = np.min(cells, axis=0) - 60
        x1, y1 = np.max(cells, axis=0) + 60                                # 九格中心往外約半格＝整個棋盤
        bh = cv2.cvtColor(empty, cv2.COLOR_BGR2HSV)[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)]
        rp = float(np.median([f[2] for f in found]))
        body_r = widen(body_r, bh, rp)
        sym_r = widen(sym_r, bh, rp, body_r)
        back_r = widen(back_r, bh, rp, body_r + sym_r) if back_r else back_r
    note = "本體 Lab %s、符號 Lab %s" % (np.round(ctr[body]).astype(int).tolist(), np.round(ctr[1 - body]).astype(int).tolist())
    return body_r + sym_r + back_r, sym_r, note, probs


def apply_cfg(cfg):
    """把量到的設定套到這個程序的 stocks（評估各種認法用）。"""
    stocks.PROFILE = cfg
    stocks.PIECE_HSV = stocks._ranges(cfg["piece_hsv"])
    stocks.SYMBOL_HSV = stocks._ranges(cfg["symbol_hsv"])


def evaluate(up, found, layout):
    """三種認法各認對幾顆：模型（原本那批訓練的）、樣板（每顆用其他八顆做的樣板認，留一法）、顏色規則。"""
    res = {}
    try:
        import ox_model
        res["model"] = [ox_model.kind_at(up, u, v) if ox_model.available() else None for u, v, _ in found]
    except Exception:
        res["model"] = [None] * len(found)
    crops = [stocks.sym_crop(up, *f) for f in found]
    tk = []
    n = len(found)
    for i, (u, v, r) in enumerate(found):
        o = [crops[j] for j in range(n) if j != i and layout[j] == "O"]
        x = [crops[j] for j in range(n) if j != i and layout[j] == "X"]
        tpl = stocks.make_tpl(np.mean(o, axis=0), np.mean(x, axis=0))
        ks = {stocks.tpl_kind(up, u + du, v + dv, r * s, tpl) for du, dv, s in
              ((0, 0, 1.0), (4, 0, 1.0), (0, -4, 1.0), (0, 0, 0.92), (0, 0, 1.08))}   # 中心偏一點、半徑差一點也要對
        tk.append(ks.pop() if len(ks) == 1 else None)
    res["template"] = tk
    res["color"] = [stocks._kind_at(up, u, v, r) for u, v, r in found]
    tpl = (np.mean([c for c, k in zip(crops, layout) if k == "O"], axis=0),
           np.mean([c for c, k in zip(crops, layout) if k == "X"], axis=0))
    return res, tpl


def score(ks, layout):
    return sum(1 for k, t in zip(ks, layout) if k == t)


# ======================= 驗證（另一個程序，用量好的設定檔） =======================
def verify(d, layout):
    """用 PIECE_PROFILE_FILE 指的設定檔重新載入讀盤程式，讀 d 資料夾裡的影像：正面每格的圈叉、背面有沒有九顆、空棋盤要沒有。"""
    import find_pieces
    plan, _ = cells_px()
    cells, _ = live_cells(cv2.imread(os.path.join(d, "empty.jpg")))
    out = {}
    for name in ("empty", "up", "back"):
        p = os.path.join(d, name + ".jpg")
        if not os.path.exists(p):
            continue
        b = [" "] * 9
        for q in find_pieces.pieces(cv2.imread(p), plan, cells=cells, blank_ok=True):
            b[q["cell"]] = q["kind"]
        out[name] = "".join(b)
    print("RESULT " + json.dumps(out))


# ======================= 互動 =======================
def ask(msg):
    try:
        return input(msg).strip().lower()
    except EOFError:
        return "q"


def stop_server():
    """對弈伺服器、相機程式會搶相機，也可能把擺上來的棋子當成一步：先停掉。"""
    sys.path.insert(0, ROOT)
    import deploy
    s = deploy.state()
    if s is not None and s.get("phase") not in ("idle", "over", None):
        say("對弈伺服器正在「%s」（%s）。" % (s.get("phase"), s.get("status")))
        if ask("要中斷它、開始棋子校正嗎？(y/N) ") != "y":
            sys.exit(0)
    say("停掉對弈伺服器和相機程式（手臂不會動）…")
    if not deploy.stop_all():
        sys.exit(1)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    layout = next((a.split("=", 1)[1].upper() for a in sys.argv if a.startswith("--layout=")), LAYOUT)
    if "--reset" in sys.argv:
        if os.path.exists(piece_profile.FILE):
            shutil.move(piece_profile.FILE, piece_profile.FILE + ".bak")
            say("已刪掉 piece_profile.json（備份成 .bak）。重啟對弈伺服器後用原本那批棋子的設定。")
        else:
            say("沒有 piece_profile.json，本來就是原本那批棋子的設定。")
        return
    if "--verify" in sys.argv:
        verify(args[0], layout)
        return
    out = os.path.join(HERE, "out", "piece_calib", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out)
    fx = load_cal("B")["fx"]
    r0 = OLD_R_MM * fx / stocks.STD_TOP_DEPTH
    ids = [i for i in range(9) if layout[i] in "OX"]          # 有放棋子的格（--layout 可以用「.」表示空格）
    kinds = "".join(layout[i] for i in ids)

    def grid(E):
        c, shift = live_cells(E)
        say("   格子中心：" + ("當下畫面找不到格線，用 cell_plan.json" if shift is None else
                            "用當下畫面的格線（跟 cell_plan.json 平均差 %.0f px）" % shift))
        return c, c[ids]
    want = layout.replace(".", " ")
    say("=" * 60)
    say(" 棋子校正：讓系統認得現場這批棋子（手臂不會動）")
    say("=" * 60)
    cam = None
    if "--files" in sys.argv:
        ims = [cv2.imread(p) for p in args]
        dps = [os.path.splitext(p)[0] + "_depth.png" for p in args]          # 同名的 *_depth.png（有就用）
        dep = [cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) if os.path.exists(p) else None for p in dps]
        dep += [None] * (3 - len(dep))
        E, F = ims[0], ims[1]
        B = ims[2] if len(ims) > 2 else None
        cells_all, cells = grid(E)
    else:
        stop_server()
        say("開相機…")
        cam = Cam()
        try:
            ask("\n[1/3] 把棋盤清空（料區不用動），手離開畫面，按 Enter 拍照 ")
            E, dE = cam.grab()
            cells_all, cells = grid(E)
            while True:
                ask("\n[2/3] 九格中央各放一顆、正面朝上：\n"
                    "        圈  叉  圈\n        叉  圈  叉\n        圈  叉  圈\n"
                    "      （四個角和正中間放圈，其他四格放叉），手離開畫面，按 Enter 拍照 ")
                F, dF = cam.grab()
                found = locate(E, F, dE, dF, cells, r0)
                miss = [ids[i] + 1 for i, f in enumerate(found) if f is None]
                if not miss:
                    break
                say("   格 %s 找不到棋子（放在格子中央、手離開畫面）。" % "、".join(map(str, miss)))
                if ask("   調好按 Enter 重拍，輸入 q 離開 ") == "q":
                    return
            B, dB, found_b = None, None, None
            if ask("\n[3/3] 九顆都翻面（背面朝上，位置不用很準），按 Enter 拍照；不做輸入 s ") != "s":
                B, dB = cam.grab()
            dep = [dE, dF, dB]
        finally:
            cam.close()
    dE, dF, dB = dep
    found = locate(E, F, dE, dF, cells, r0)
    miss = [ids[i] + 1 for i, f in enumerate(found) if f is None]
    if miss:
        say("格 %s 找不到棋子，沒辦法校正。" % "、".join(map(str, miss)))
        return
    found_b = locate(E, B, dE, dB, cells, r0) if B is not None else None
    for n, im in (("empty", E), ("up", F), ("back", B)):
        if im is not None:
            cv2.imwrite(os.path.join(out, n + ".jpg"), im)
    for n, d in (("empty", dE), ("up", dF), ("back", dB)):
        if d is not None:
            cv2.imwrite(os.path.join(out, n + "_depth.png"), np.clip(d, 0, 65535).astype(np.uint16))
    for i, (u, v, r) in enumerate(found):                         # 每顆的裁切（之後重新訓練模型可以用）
        R = int(1.3 * r)
        cv2.imwrite(os.path.join(out, "crop_%d_%s.jpg" % (ids[i] + 1, kinds[i])),
                    F[max(0, int(v) - R):int(v) + R, max(0, int(u) - R):int(u) + R])

    # ---- 大小、厚度 ----
    tops = [ring(dF, u, v, 14) for u, v, _ in found]
    depth_at = [t if t else stocks.STD_TOP_DEPTH for t in tops]
    r_mm = [r * d / fx for (u, v, r), d in zip(found, depth_at)]
    disc_r = float(np.median(r_mm))
    th = [ring(dE, u, v, 14) - t for (u, v, _), t in zip(found, tops) if t and ring(dE, u, v, 14)]
    piece_mm = float(np.median(th)) if len(th) >= 5 else None
    warn = []
    say("\n量到：")
    say("   半徑 %.1f mm（每顆 %s；原本那批 %.0f mm）" % (disc_r, " ".join("%.1f" % x for x in r_mm), OLD_R_MM))
    if piece_mm is None:
        say("   厚度：沒有深度，量不到（沿用 %.1f mm）" % OLD_MM)
    else:
        say("   厚度 %.1f mm（每顆 %s；原本那批 %.1f mm）" % (piece_mm, " ".join("%.1f" % x for x in th), OLD_MM))
        if piece_mm < 2.5:
            warn.append("棋子很薄（%.1f mm）：吸取時「有沒有碰到棋子」的判斷（carrier 的 h_min 5.5）可能分不出棋子和棋盤，先實機試吸幾次" % piece_mm)
        if piece_mm > 8:
            warn.append("棋子很厚（%.1f mm）：料疊顆數、放下高度會跟著變，先實機試放幾次" % piece_mm)
    if abs(disc_r - OLD_R_MM) > 6:
        warn.append("棋子大小跟原本差很多（半徑 %.0f → %.0f mm）：吸取偏移上限、轉送點和支架的間距可能要重看" % (OLD_R_MM, disc_r))
    if np.ptp(r_mm) > 4:
        warn.append("九顆量到的半徑差到 %.1f mm：可能有一顆沒放平或被手遮到" % np.ptp(r_mm))

    # ---- 顏色、認法 ----
    piece_hsv, sym_hsv, note, probs = colors(F, found, kinds, B, found_b, E, cells_all)
    say("   顏色：%s" % note)
    say("   符號顏色範圍 %s" % sym_hsv)
    for p in probs:
        say("   ⚠ " + p)
    cfg = dict(disc_r_mm=round(disc_r, 2), piece_r_mm=round(disc_r + 1.0, 2), piece_hsv=piece_hsv, symbol_hsv=sym_hsv)
    if piece_mm is not None:
        cfg["piece_mm"] = round(piece_mm, 2)
    apply_cfg(cfg)
    res, tpl = evaluate(F, found, kinds)
    sc = {k: score(v, kinds) for k, v in res.items()}
    N = len(found)
    name = {"model": "原本的辨識模型", "template": "校正樣板（留一法）", "color": "顏色規則"}
    say("\n三種認法（正面 %d 顆，正確答案 %s）：" % (N, kinds))
    for k in ("model", "template", "color"):
        say("   %-10s %d/%d  %s" % (name[k], sc[k], N, "".join(x or "-" for x in res[k])))
    cfg["use_model"] = sc["model"] == N
    cfg["templates"] = TPL_NAME if sc["template"] >= N - 1 else None
    if sc["template"] < N - 1 and sc["color"] < N - 1 and not cfg["use_model"]:
        warn.append("三種認法都認不好這批棋子：符號跟底色可能太像，或光線反光太強。人機對弈時人下的棋子可能會被認錯（機器自己下不受影響：翻面後只靠紀錄）")
    cfg["note"] = "piece_calib %s：%s；模型 %d、樣板 %d、顏色 %d（共 %d 顆）" % (time.strftime("%Y-%m-%d %H:%M"), note, sc["model"], sc["template"], sc["color"], N)
    stocks.PROFILE = None

    # ---- 用設定檔重新載入讀盤程式驗證 ----
    try_json = os.path.join(out, "piece_profile.json")
    json.dump(cfg, open(try_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez(os.path.join(out, TPL_NAME), O=tpl[0].astype(np.float32), X=tpl[1].astype(np.float32))
    env = dict(os.environ, PIECE_PROFILE="1", PIECE_PROFILE_FILE=try_json, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--verify", out, "--layout=" + layout], env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HERE)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    ok = False
    if line is None:
        say("\n驗證失敗：\n" + (r.stderr or r.stdout)[-1500:])
    else:
        v = json.loads(line[7:])
        up = v.get("up", "?" * 9)
        say("\n用新設定讀盤（跟對弈時同一套程式）：")
        say("   空棋盤：%s" % ("沒有棋子 ✓" if v.get("empty", " " * 9).strip() == "" else "⚠ 讀到 %s（木頭或膠帶被當成棋子）" % v["empty"].replace(" ", "-")))
        say("   正面  ：%s  %s" % (up.replace(" ", "-"), "✓" if up == want else "⚠ 應該是 %s" % want.replace(" ", "-")))
        nb = sum(1 for c in v.get("back", "") if c != " ")
        if "back" in v:
            say("   背面  ：找到 %d/%d 顆 %s" % (nb, N, "✓" if nb == N else "⚠（翻面的棋子找不到，料疊、機器下的棋子會看不到）"))
        ok = up == want and v.get("empty", " " * 9).strip() == "" and ("back" not in v or nb == N)
    for w in warn:
        say("   ⚠ " + w)
    say("\n樣本影像：%s" % out)
    if "--files" in sys.argv:
        return
    if ask("\n%s要存成新的棋子設定嗎？(y/N) " % ("" if ok else "驗證沒有全部通過，還是")) != "y":
        say("沒有存。現在的設定不變。")
        return
    if os.path.exists(piece_profile.FILE):
        shutil.copy(piece_profile.FILE, piece_profile.FILE + ".bak")
    shutil.copy(try_json, piece_profile.FILE)
    shutil.copy(os.path.join(out, TPL_NAME), os.path.join(HERE, TPL_NAME))
    say("已存：%s" % piece_profile.FILE)
    say("要回到原本那批棋子的設定：python Calib\\piece_calib.py --reset")
    if ask("\n要現在重新啟動 OOXX（一鍵部署）讓新設定生效嗎？(y/N) ") == "y":
        subprocess.call([sys.executable, os.path.join(ROOT, "deploy.py")], cwd=ROOT)
    else:
        say("之後跑桌面的「OOXX一鍵部署.bat」就會用新設定。")


if __name__ == "__main__":
    main()
