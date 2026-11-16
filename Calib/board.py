"""井字棋盤偵測（相機正上方俯視）：格線交點、9 格中心、每格內容（空／O／X）。

    python board.py            # 擷取一張圖並分析，存 out/board_debug.png
座標一律是彩色影像的像素 (x, y)；格子編號 0..8 = 影像由上到下、由左到右。
"""
import cv2
import numpy as np


def _dark_mask(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (g < 70).astype(np.uint8) * 255


def board_roi(img):
    """用木紋色找棋盤外框，回傳 (x0,y0,x1,y1)。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    wood = cv2.inRange(hsv, (10, 15, 100), (35, 140, 255))
    wood = cv2.morphologyEx(wood, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    h_img, w_img = wood.shape
    wood[:, : int(w_img * 0.25)] = 0          # 左側有米色椅子，同色相
    wood[:, int(w_img * 0.75):] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(wood)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = st[i]
    return int(x), int(y), int(x + w), int(y + h)


def find_grid(img, roi=None):
    """回傳 dict：lines_x(2 條直膠帶 x)、lines_y(2 條橫膠帶 y)、corners(4 個內部交點 TL,TR,BL,BR)、
    cells(9 個格中心 px，由上到下由左到右)、cell(平均格距 px)、roi。
    roi：(x0,y0,x1,y1) 只在這個範圍找；沒給就用木紋外框（board_roi）。

    作法：木板外框三等分當初值 → 在預期位置 ±30 px 內找最暗的欄／列（膠帶）微調。"""
    x0, y0, x1, y1 = roi or board_roi(img)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 只取不含圓片的帶狀區域來找膠帶：直線用「兩圓片之間」的列、橫線用「兩圓片之間」的欄不容易定位，
    # 所以改用全高／全寬的暗度總和，圓片邊緣離膠帶夠遠（>30 px）不會搶走極值。
    def refine(profile, lo, hi, expect, win=30):
        a, b = max(int(expect - win), lo), min(int(expect + win), hi)
        seg = profile[a:b]
        k = np.ones(7) / 7
        seg = np.convolve(seg, k, mode="same")
        return a + int(np.argmax(seg))

    dark = (255.0 - g)
    col = dark[y0:y1, :].sum(axis=0)
    row = dark[:, x0:x1].sum(axis=1)
    w, h = x1 - x0, y1 - y0
    def best_pair(profile, lo, hi, gap_lo, gap_hi):
        """在 [lo,hi) 內找兩條最暗、間距在 [gap_lo,gap_hi] 的膠帶線（不依賴木紋框三等分：
        木紋框可能被同色木板延伸而變大，2026-11-10 實測）。"""
        p = np.convolve(profile, np.ones(9) / 9, mode="same")
        seg = p[lo:hi]
        peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
        peaks = sorted(peaks, key=lambda i: -seg[i])[:12]
        best = None
        for i in peaks:
            for j in peaks:
                if gap_lo <= j - i <= gap_hi and (best is None or seg[i] + seg[j] > best[0]):
                    best = (seg[i] + seg[j], i, j)
        if best is None:
            raise RuntimeError("找不到間距合理的兩條膠帶線")
        return [lo + best[1], lo + best[2]]

    lx = best_pair(col, x0, x1, 100, 150)
    # 橫線只看兩條直線之間的欄（避開木紋框外的同色木板與底座）
    row = dark[:, lx[0] - 5:lx[1] + 5].sum(axis=1)
    ly = best_pair(row, y0, y1, 120, 180)
    if not (100 <= lx[1] - lx[0] <= 150 and 120 <= ly[1] - ly[0] <= 180):
        raise RuntimeError("膠帶間距不合理 x=%s y=%s，偵測失敗" % (lx, ly))
    p = lx[1] - lx[0]                         # 兩條直膠帶的間距 = 一格寬；外圍格以半格寬外推
    cx = [lx[0] - p / 2, (lx[0] + lx[1]) / 2, lx[1] + p / 2]
    cy = [ly[0] - p / 2, (ly[0] + ly[1]) / 2, ly[1] + p / 2]
    xs = [cx[0] - p / 2, lx[0], lx[1], cx[2] + p / 2]
    ys = [cy[0] - p / 2, ly[0], ly[1], cy[2] + p / 2]
    cells = np.array([[cx[c], cy[r]] for r in range(3) for c in range(3)])
    corners = np.array([[lx[0], ly[0]], [lx[1], ly[0]], [lx[0], ly[1]], [lx[1], ly[1]]], dtype=float)
    cell = float(np.mean([xs[1] - xs[0], xs[2] - xs[1], xs[3] - xs[2], ys[1] - ys[0], ys[2] - ys[1], ys[3] - ys[2]]))
    return dict(corners=corners, cells=cells, cell=cell, roi=(x0, y0, x1, y1), lines_x=lx, lines_y=ly)


def find_grid_robust(cam, tries=7):
    """多幀偵測，丟掉不合理的，取中位數；回傳 (img, grid, cls)。"""
    ok = []
    for _ in range(tries):
        img = cam.grab(n=2, timeout=20)          # 相機降為 USB 2 時幀率低，放寬
        try:
            ok.append((img, find_grid(img)))
        except RuntimeError:
            continue
    if len(ok) < 3:
        raise RuntimeError("棋盤偵測不穩（合格 %d/%d 幀）" % (len(ok), tries))
    cells = np.median([g["cells"] for _, g in ok], axis=0)
    img, g = ok[-1]
    g = dict(g, cells=cells)
    return img, g, classify_cells(img, g)


def classify_cells(img, grid, r=None):
    """每格取中心圓內：深色圓片 + 綠色符號 → 判斷 O（綠色環）／X（綠色叉）／空。"""
    r = r or int(grid["cell"] * 0.36)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (60, 60, 60), (100, 255, 255))
    dark = _dark_mask(img)
    res = []
    for c in grid["cells"]:
        m = np.zeros(green.shape, np.uint8)
        cv2.circle(m, tuple(int(v) for v in c), r, 255, -1)
        area = float(m.sum() / 255)
        if area <= 0:                       # 格子落在影像外（格線被手臂擋住、偵測錯）：當成空，不要除以零
            res.append(("empty", 0.0, 0.0))
            continue
        g = float(cv2.bitwise_and(green, m).sum() / 255) / area
        d = float(cv2.bitwise_and(dark, m).sum() / 255) / area
        if g < 0.03 or d < 0.25:
            res.append(("empty", g, d))
            continue
        # 環 vs 叉：O 的圓心是黑色（中心小圓內無綠），X 的中心是綠色
        ctr = np.zeros(green.shape, np.uint8)
        cv2.circle(ctr, tuple(int(v) for v in c), max(r // 5, 4), 255, -1)
        centre_green = float(cv2.bitwise_and(green, ctr).sum() / 255) / max(float(ctr.sum() / 255), 1.0)
        res.append(("X" if centre_green > 0.5 else "O", g, d))
    return res


def debug_draw(img, grid, cls):
    out = img.copy()
    for p in grid["corners"]:
        cv2.circle(out, tuple(int(v) for v in p), 6, (0, 0, 255), -1)
    for i, (c, (k, g, d)) in enumerate(zip(grid["cells"], cls)):
        cv2.putText(out, "%d:%s" % (i, k), (int(c[0]) - 30, int(c[1]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


if __name__ == "__main__":
    from common import OUT, Camera
    cam = Camera()
    try:
        img = cam.grab(n=3)
        print("內參：", cam.intr)
    finally:
        cam.close()
    grid = find_grid(img)
    cls = classify_cells(img, grid)
    print("交點 TL TR BL BR：\n", grid["corners"].round(1))
    print("ROI", grid["roi"], "直線 x", grid["lines_x"], "橫線 y", grid["lines_y"])
    print("格距 %.1f px" % grid["cell"])
    for i, (k, g, d) in enumerate(cls):
        print("格 %d 中心 %s → %s (綠 %.2f 暗 %.2f)" % (i, grid["cells"][i].round(0), k, g, d))
    cv2.imwrite(OUT + "/board_debug.png", debug_draw(img, grid, cls))
