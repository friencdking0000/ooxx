"""自動手眼校正 v2：追蹤裝在前臂末端的真空方盒（不需標記、不需深度、不需示教）。

為何不用 J4：方盒裝在末端固定板上，不隨 J4 轉；隨 J4 轉的吸盤被方盒擋住看不到（實測）。
模型：手臂固定高度 Z0、末端板保持水平，方盒上一個參考點在手臂座標 = TCP_xy + R(J1)·o（o 為固定偏移）。
  相機近似正下視：參考點像素 p 與其 XY 的關係為  XY = s·Rφ·F·(p − c) + t     （F = 可能的鏡像）
  → TCP_xy = s·Rφ·F·(p − c) + t − R(J1)·o ，對 (a=s·cosφ, b=s·sinφ, t, o) 是線性，最小平方解。
  s = D_h / fx（D_h = 參考點到相機距離）；換算到棋盤面用 s_board = D_board / fx，t 不變（光軸位置）。
用法：
    python calib_box.py B                 # 乾跑：IK 檢查、印計畫
    python calib_box.py B --go            # 實際運動（Z=0、10% 速度）
結果：calib_<arm>.json，含 board_px_to_arm()所需參數；並印出每點殘差。
"""
import json
import sys
import time

import cv2
import numpy as np

from common import OUT, Camera, connect_arm, enable, joint_angles, sync_move

CX, CY, FX = 639.5, 359.6, 687.8
D_BOARD = 540.0           # mm，棋盤面實測深度（2026-11-14 重新量測；先前 533）
POSES = {   # X 上限 310：再遠方盒會靠近對面手臂的底座
    "B": [(230, -100), (230, 0), (230, 100), (270, -100), (270, 0), (270, 100), (310, -70), (310, 0), (310, 70)],
    # A 只用 Y ≥ 0（影像左半邊）：A 往右側（Y<0）移動時兩次讓相機中斷、曾撞偏相機（右側有相機線與支架）
    "A": [(230, 0), (230, 60), (230, 120), (270, 0), (270, 60), (270, 120), (310, 0), (310, 60), (310, 120)],
}
BASE_PX = {"A": (575.0, 680.0), "B": (600.0, 15.0)}


def opt(name, default):
    for a in sys.argv[1:]:
        if a.startswith("--%s=" % name):
            return type(default)(a.split("=", 1)[1])
    return default


_FACE = None


FACE_D0 = 244.0      # mm：模板拍攝時（B，Z=0）方盒頂面到相機的距離（4 點校正得到）


def face_scale(z):
    """手臂 Z（mm）時方盒頂面相對模板的影像縮放（頂面離相機 FACE_D0 − z）。"""
    return FACE_D0 / (FACE_D0 - z)


def _face_gray():
    global _FACE
    if _FACE is None:
        import os
        _FACE = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "box_face.png"))
    return cv2.cvtColor(_FACE, cv2.COLOR_BGR2GRAY)


def _match(g, t, ang):
    h, w = t.shape
    mask = np.zeros_like(t)
    cv2.circle(mask, (w // 2, h // 2), w // 2 - 2, 255, -1)
    tr = cv2.warpAffine(t, cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0), (w, h), borderMode=cv2.BORDER_REPLICATE)
    res = cv2.matchTemplate(g, tr, cv2.TM_CCOEFF_NORMED, mask=mask)
    res[~np.isfinite(res)] = 0
    _, mx, _, loc = cv2.minMaxLoc(res)
    return mx, np.array([loc[0] + w / 2, loc[1] + h / 2])


def find_face_scaled(img, j1, scales, coarse=0.5):
    """尺寸未知時找頂面：半解析度粗掃（比例 × 角度 J1／J1+180 ±8°，步 4°）→ 全解析度局部細化（±1 比例步、±3°、步 0.5°）。
    回傳 (中心, 角度, 分數, 比例)。"""
    t0 = _face_gray()
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gs = cv2.resize(g, None, fx=coarse, fy=coarse, interpolation=cv2.INTER_AREA)
    best = None
    for sc in scales:
        t = cv2.resize(t0, None, fx=sc * coarse, fy=sc * coarse, interpolation=cv2.INTER_AREA)
        for base in (j1, j1 + 180):
            for ang in np.arange(base - 8, base + 8.1, 4):
                mx, c = _match(gs, t, ang)
                if best is None or mx > best[0]:
                    best = (mx, c / coarse, ang, sc)
    _, c, ang0, sc0 = best
    step = (scales[1] - scales[0]) if len(scales) > 1 else 0.05
    r = int(t0.shape[0] * sc0) + 60
    x0, y0 = max(0, int(c[0]) - r), max(0, int(c[1]) - r)
    sub = g[y0:int(c[1]) + r, x0:int(c[0]) + r]
    fine = None
    for sc in np.arange(sc0 - step, sc0 + step + 1e-6, step / 4):
        t = cv2.resize(t0, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= sub.shape[0] or t.shape[1] >= sub.shape[1]:
            continue
        for ang in np.arange(ang0 - 3, ang0 + 3.01, 0.5):
            mx, cc = _match(sub, t, ang)
            if fine is None or mx > fine[2]:
                fine = (cc + [x0, y0], float(((ang + 180) % 360) - 180), float(mx), float(sc))
    return fine


def fit_robust(px, tcp, j1, tol=2.0):
    """反覆擬合，每次剔除殘差最大且 > tol mm 的點，至少保留 5 點。回傳 (fit, 保留索引)。"""
    idx = list(range(len(px)))
    while True:
        f = fit([px[i] for i in idx], [tcp[i] for i in idx], [j1[i] for i in idx])
        worst = int(np.argmax(f["err"]))
        if f["err"][worst] <= tol or len(idx) <= 5:
            return f, idx
        idx.pop(worst)


_FACES = {}


def find_face(img, j1=None, search=None, step=2.0, scale=1.0, arm="B"):
    """在影像中找方盒頂面（中心 = R 字）。旋轉掃描模板 + 圓形遮罩 NCC。
    j1：手臂 J1 角度（給了就只在 ±j1 附近 ±12° 掃描，再細掃 0.5°）。search：(x0,y0,x1,y1) 搜尋範圍。
    回傳 (中心像素, 旋轉角°, 分數)。"""
    import os
    if arm not in _FACES:      # 各臂自己的模板：J1=0° 時拍的頂面（B：Z=0；A：Z=-40）
        fn = "box_face.png" if arm == "B" else "box_face_%s.png" % arm
        _FACES[arm] = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), fn))
    t = cv2.cvtColor(_FACES[arm], cv2.COLOR_BGR2GRAY)
    if abs(scale - 1) > 1e-3:
        t = cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = t.shape
    mask = np.zeros_like(t)
    cv2.circle(mask, (w // 2, h // 2), w // 2 - 2, 255, -1)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ox = oy = 0
    if search:
        x0, y0, x1, y1 = search
        x0, y0 = max(0, x0), max(0, y0)
        g, ox, oy = g[y0:y1, x0:x1], x0, y0

    def score(ang):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        tr = cv2.warpAffine(t, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        res = cv2.matchTemplate(g, tr, cv2.TM_CCOEFF_NORMED, mask=mask)
        res[~np.isfinite(res)] = 0
        _, mx, _, loc = cv2.minMaxLoc(res)
        return mx, loc

    if j1 is None:
        angs = np.arange(-180, 180, step)
    else:
        angs = np.concatenate([np.arange(a - 12, a + 12.1, step) for a in (j1, -j1, j1 + 180, -j1 + 180)])
    best = max(((*score(a), a) for a in angs), key=lambda r: r[0])
    for a in np.arange(best[2] - step, best[2] + step + 0.01, 0.5):
        r = (*score(a), a)
        if r[0] > best[0]:
            best = r
    mx, loc, ang = best
    return np.array([loc[0] + ox + w / 2, loc[1] + oy + h / 2]), float(ang), float(mx)


def home_ghost(home, imgs, thr=40, frac=0.7):
    """停放殘影：和停放影像相比、在大多數姿態都不同的像素（手臂離開後露出的背景）。"""
    votes = sum(((cv2.absdiff(home, im).max(axis=2) > thr).astype(np.uint8)) for im in imgs)
    g = (votes >= max(2, int(np.ceil(frac * len(imgs))))).astype(np.uint8) * 255
    return cv2.dilate(g, np.ones((15, 15), np.uint8))


def arm_mask(img, home, ghost, thr=40):
    m = (cv2.absdiff(img, home).max(axis=2) > thr).astype(np.uint8) * 255
    m = cv2.bitwise_and(m, cv2.bitwise_not(ghost))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def box_template(img, home, base_px, ghost=None):
    """該姿態的手臂區域（扣除停放殘影）中離底座最遠的一端 → 方盒區域（bbox）。"""
    m = arm_mask(img, home, ghost if ghost is not None else np.zeros(img.shape[:2], np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))            # 最大連通區 = 該姿態的手臂
    ys, xs = np.nonzero(lab == k)
    dist = np.hypot(xs - base_px[0], ys - base_px[1])
    far = np.array([xs[dist.argmax()], ys[dist.argmax()]], float)
    sel = np.hypot(xs - far[0], ys - far[1]) < 170               # 方盒約 150 px 大
    return int(xs[sel].min()), int(ys[sel].min()), int(xs[sel].max()), int(ys[sel].max())


def track(tmpl_img, tmpl_box, img, home, ghost=None):
    """把模板方盒的中心點對應到 img 中；回傳 (點, 旋轉角°, 內點)。"""
    x0, y0, x1, y1 = tmpl_box
    g0 = cv2.cvtColor(tmpl_img, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m0 = np.zeros_like(g0)
    m0[y0:y1, x0:x1] = 255
    m1 = arm_mask(img, home, ghost if ghost is not None else np.zeros(img.shape[:2], np.uint8), thr=30)
    m1 = cv2.dilate(m1, np.ones((31, 31), np.uint8))
    orb = cv2.ORB_create(3000)
    k0, d0 = orb.detectAndCompute(g0, m0)
    k1, d1 = orb.detectAndCompute(g1, m1)
    if d0 is None or d1 is None or len(k0) < 15 or len(k1) < 15:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in bf.knnMatch(d0, d1, k=2):
        if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 10:
        return None
    p0 = np.float32([k0[g.queryIdx].pt for g in good])
    p1 = np.float32([k1[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inl.sum() < 10:
        return None
    c0 = np.array([(x0 + x1) / 2, (y0 + y1) / 2, 1.0])
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    return M @ c0, float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), int(inl.sum()), scale


def fit(px, tcp, j1):
    """線性最小平方；試兩種鏡像，取殘差小者。"""
    best = None
    for flip in (1, -1):
        rows, rhs = [], []
        for (u, v), (x, y), th in zip(px, tcp, np.radians(j1)):
            u, v = u - CX, (v - CY) * flip
            c, s = np.cos(th), np.sin(th)
            #           a    b   tx  ty   ox   oy
            rows.append([u, -v, 1, 0, -c, s]); rhs.append(x)
            rows.append([v, u, 0, 1, -s, -c]); rhs.append(y)
        A, b = np.array(rows), np.array(rhs)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = (A @ sol - b).reshape(-1, 2)
        err = np.linalg.norm(res, axis=1)
        if best is None or err.mean() < best["err"].mean():
            best = dict(flip=flip, sol=sol, err=err)
    return best


def main():
    name = sys.argv[1]
    go = "--go" in sys.argv
    z0 = opt("z", -40.0)
    arm = connect_arm(name)
    cam = Camera()
    try:
        arm.wait_feed()
        r0 = float(arm.pose()[3])
        x_h, y_h = arm.pose()[:2]
        plan = [(x, y) for x, y in POSES[name] if arm.dash.InverseSolution(x, y, z0, r0, 0, 0).startswith("0,")]
        print("%s 目前 %s；Z0=%.0f；可達姿態 %d/%d" % (name, arm.pose().round(1), z0, len(plan), len(POSES[name])))
        if not go:
            return
        if len(plan) < 6:
            raise RuntimeError("可達姿態不足")
        enable(arm, speed=10)
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (x_h, y_h, max(z0, arm.pose()[2]), r0))   # 先到 Z0（只升不降）
        px, tcp, j1s, log = [], [], [], []
        face_scale_found = [None]
        for x, y in plan:
            sync_move(arm, "MovJ(%f,%f,%f,%f)" % (x, y, z0, r0))
            time.sleep(0.4)
            img = cam.grab(skip=3, timeout=15)
            j1 = joint_angles(arm)[0]
            cv2.imwrite("%s/cb_%s_%d_%d.png" % (OUT, name, x, y), img)
            # 各臂用自己的模板：B 模板於 Z=0 拍（頂面距相機 244 mm）、A 模板於 Z=-40 拍（281 mm）
            tz, td = (0.0, 244.0) if name == "B" else (-40.0, 281.0)
            nominal = td / (td + (tz - z0))
            # 相機高度可能變了（2026-11-14 格線間距 115→123 px）：第一張先在 0.8～1.25 倍掃比例，之後在最佳比例 ±3% 內找
            if face_scale_found[0] is None:
                cands = np.arange(0.8, 1.26, 0.04) * nominal
            else:
                cands = np.arange(0.97, 1.031, 0.03) * face_scale_found[0]
            best = None
            for k in cands:
                r = find_face(img, j1=j1, scale=k, arm=name)
                if best is None or r[2] > best[0][2]:
                    best = (r, k)
            (c, ang, sc), kbest = best
            if face_scale_found[0] is None and sc > 0.7:
                face_scale_found[0] = kbest
                print("  （頂面比例 %.3f，為名義值的 %.2f 倍）" % (kbest, kbest / nominal))
            print("  (%d,%d) J1 %+.1f° → 頂面 (%.1f,%.1f) 旋轉 %+.1f° 分數 %.3f" % (x, y, j1, c[0], c[1], ang, sc))
            if sc > 0.7:
                px.append(c), tcp.append((x, y)), j1s.append(j1)
                log.append(dict(xy=[x, y], j1=j1, px=c.tolist(), angle=ang, score=sc))
        arm.dash.DisableRobot()
        # 旋轉一致性：頂面旋轉 − J1（模 180）應為常數（相機相對手臂的偏轉＋模板角度），以中位數為準 ±3°
        d = np.array([((p["angle"] - p["j1"] + 90) % 180) - 90 for p in log])
        if len(d):
            ok = np.abs(d - np.median(d)) < 3
            print("旋轉偏差中位數 %+.1f°，一致 %d/%d 點" % (np.median(d), ok.sum(), len(d)))
            px = [p for p, k in zip(px, ok) if k]
            tcp = [p for p, k in zip(tcp, ok) if k]
            j1s = [p for p, k in zip(j1s, ok) if k]
            log = [p for p, k in zip(log, ok) if k]
        if len(px) < 6:
            raise RuntimeError("有效點不足（%d）" % len(px))
        f, keep = fit_robust(px, tcp, j1s, tol=1.5)        # 剔除殘差離群點（錯配）
        log = [log[i] for i in keep]
        px = [px[i] for i in keep]
        a, b, tx, ty, ox, oy = f["sol"]
        s = float(np.hypot(a, b))
        d_h = s * FX
        print("\n擬合 %d 點：殘差 平均 %.2f mm、最大 %.2f mm；鏡像 %s" % (len(px), f["err"].mean(), f["err"].max(), f["flip"] == -1))
        print("相機旋轉 φ=%.1f°；頂面距相機 %.0f mm（Z=%.0f 時高於棋盤面 %.0f mm）；方盒偏移 o=(%.1f, %.1f) mm"
              % (np.degrees(np.arctan2(b, a)), d_h, z0, D_BOARD - d_h, ox, oy))
        out = dict(arm=name, z0=z0, a=a, b=b, tx=tx, ty=ty, flip=f["flip"], o=[ox, oy], s=s,
                   D_ref=d_h, D_board=D_BOARD, cx=CX, cy=CY, fx=FX, face_above_board_at_z0=D_BOARD - d_h,
                   residual_mean=float(f["err"].mean()), residual_max=float(f["err"].max()),
                   n_points=len(px), points=log)
        json.dump(out, open("%s/../calib_%s.json" % (OUT, name), "w"), indent=1)
        print("已存 calib_%s.json" % name)
    except Exception as e:
        print("停止：", e)
        try:
            arm.dash.ResetRobot()
        except Exception:
            pass
    finally:
        if go:
            try:
                arm.dash.DisableRobot()
            except Exception:
                pass
        cam.close()
        arm.close()


def board_px_to_arm(cal, px, depth=None):
    """棋盤面（或指定深度）上一個像素 → 手臂 TCP 的 XY（mm）。"""
    k = (depth or cal["D_board"]) / cal["fx"] / cal["s"]          # 相對參考點高度的比例
    u, v = px[0] - cal["cx"], (px[1] - cal["cy"]) * cal["flip"]
    a, b = cal["a"] * k, cal["b"] * k
    return np.array([a * u - b * v + cal["tx"], b * u + a * v + cal["ty"]])


if __name__ == "__main__":
    main()
