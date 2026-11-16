"""自動手眼校正（不需標記、不需深度）：用 J4 旋轉求夾爪軸心的像素位置。

原理：夾爪繞 J4 軸旋轉時，影像上軸心是不動點。在若干高處姿態各拍兩張（J4 與 J4+θ），
  用特徵匹配估相似轉換 → 解出旋轉中心 c（像素）= 該姿態 TCP 軸的 XY。
  再用 (c_i, 手臂 X_i Y_i) 擬合：  xy_i = R(φ) · [(u-cx), (v-cy)] · k + t     （k = D_h / fx，mm/px）
  其中 D_h 是軸心到相機的距離；棋盤面深度 D_b=533 mm 已量測，所以
  「軸心高於棋盤面」 = D_b − D_h，並可把像素換算到棋盤面： k_board = D_b / fx。
用法：
    python calib_auto.py B              # 乾跑：只檢查可達性、印計畫
    python calib_auto.py B --go         # 實際運動（高處、低速；周圍淨空、急停在手邊）
    python calib_auto.py B --go --z=60 --theta=30
結果：calib_<arm>_auto.json（φ、t、k、D_h、殘差）。
"""
import json
import sys
import time

import cv2
import numpy as np

from common import OUT, Camera, connect_arm, enable, joint_angles, sync_move

CX, CY, FX = 639.5, 359.6, 687.8
D_BOARD = 533.0                          # mm，棋盤面實測深度
# 各臂校正姿態（手臂座標 X,Y；Z 由 --z 給）。假設座標：B 的 +X 朝影像下、+Y 朝右，A 相反（見 CLAUDE.md）。
POSES = {
    "B": [(270, -90), (270, 0), (270, 90), (330, -90), (330, 0), (330, 90), (300, 45)],
    "A": [(270, -90), (270, 0), (270, 90), (330, -90), (330, 0), (330, 90), (300, 45)],
}


def opt(name, default):
    for a in sys.argv[1:]:
        if a.startswith("--%s=" % name):
            return type(default)(a.split("=", 1)[1])
    return default


def pivot(img0, img1, theta_deg):
    """回傳 (pivot_px, 實測角度, 內點數)；失敗回傳 None。"""
    g0, g1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g0, g1)
    mask = (diff > 25).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((25, 25), np.uint8))
    orb = cv2.ORB_create(1500)
    k0, d0 = orb.detectAndCompute(g0, mask)
    k1, d1 = orb.detectAndCompute(g1, mask)
    if d0 is None or d1 is None or len(k0) < 12 or len(k1) < 12:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
    if len(m) < 10:
        return None
    p0 = np.float32([k0[x.queryIdx].pt for x in m])
    p1 = np.float32([k1[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.5)
    if M is None or inl.sum() < 8:
        return None
    ang = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    R, t = M[:, :2], M[:, 2]
    A = np.eye(2) - R
    if abs(np.linalg.det(A)) < 1e-3:
        return None
    c = np.linalg.solve(A, t)
    return c, float(ang), int(inl.sum())


def fit(px, xy):
    """xy = R(φ) @ ((px - c0) * k) + t，最小平方；回傳 dict。以 Procrustes（含尺度）求解。"""
    P = (np.asarray(px, float) - [CX, CY])
    Q = np.asarray(xy, float)
    pm, qm = P.mean(0), Q.mean(0)
    P0, Q0 = P - pm, Q - qm
    # 允許鏡像（B 的座標軸手性不確定）：兩種情形都試，取殘差小者
    best = None
    for flip in (1, -1):
        Pf = P0 * [1, flip]
        U, S, Vt = np.linalg.svd(Pf.T @ Q0)
        Rm = (U @ Vt).T
        if np.linalg.det(Rm) < 0:
            continue
        k = S.sum() / (Pf ** 2).sum()
        pred = (Rm @ (Pf * k).T).T + qm
        err = np.linalg.norm(pred - Q, axis=1)
        if best is None or err.mean() < best["err"].mean():
            best = dict(flip=flip, R=Rm, k=float(k), err=err, qm=qm, pm=pm)
    return best


def main():
    name = sys.argv[1]
    go = "--go" in sys.argv
    z0, theta = opt("z", 60.0), opt("theta", 30.0)
    arm = connect_arm(name)
    cam = Camera()
    try:
        arm.wait_feed()
        j_home = joint_angles(arm)
        r0 = float(arm.pose()[3])
        print("%s 目前姿態 %s 關節 %s" % (name, arm.pose().round(1), np.round(j_home, 1)))
        plan = []
        for x, y in POSES[name]:
            rep = arm.dash.InverseSolution(x, y, z0, r0, 0, 0)
            ok = rep.startswith("0,")
            print("  姿態 (%d,%d,%.0f,R%.1f) IK %s" % (x, y, z0, r0, "OK" if ok else "不可達"))
            if ok:
                plan.append((x, y))
        lift = arm.dash.InverseSolution(arm.pose()[0], arm.pose()[1], z0, r0, 0, 0).startswith("0,")
        print("先垂直抬升到 Z=%.0f：%s" % (z0, "OK" if lift else "不可達"))
        if not go:
            print("（乾跑，不動）可用姿態 %d 個" % len(plan))
            return
        if not lift or len(plan) < 4:
            raise RuntimeError("可達姿態不足")

        enable(arm, speed=10)
        x_now, y_now = arm.pose()[:2]
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (x_now, y_now, z0, r0))        # 垂直抬升
        pts_px, pts_xy, log = [], [], []
        j4 = None
        for x, y in plan:
            sync_move(arm, "MovJ(%f,%f,%f,%f)" % (x, y, z0, r0))
            time.sleep(0.8)
            img0 = cam.grab(n=20)
            j = joint_angles(arm)
            sync_move(arm, "JointMovJ(%f,%f,%f,%f)" % (j[0], j[1], j[2], j[3] + theta))
            time.sleep(0.8)
            img1 = cam.grab(n=20)
            sync_move(arm, "JointMovJ(%f,%f,%f,%f)" % tuple(j))
            tag = "%s_%d_%d" % (name, x, y)
            cv2.imwrite("%s/ca_%s_0.png" % (OUT, tag), img0)
            cv2.imwrite("%s/ca_%s_1.png" % (OUT, tag), img1)
            res = pivot(img0, img1, theta)
            if res is None:
                print("  (%d,%d) 特徵匹配失敗，略過" % (x, y))
                continue
            c, ang, n = res
            good = abs(abs(ang) - theta) < 8
            print("  (%d,%d) 軸心像素 (%.1f,%.1f) 測得旋轉 %+.1f°（預期 ±%.0f）內點 %d %s"
                  % (x, y, c[0], c[1], ang, theta, n, "" if good else "← 角度不符，捨棄"))
            if good:
                pts_px.append(c)
                pts_xy.append([x, y])
                log.append(dict(xy=[x, y], px=c.tolist(), angle=ang, inliers=n))
        if len(pts_px) < 4:
            raise RuntimeError("有效點不足 4 個（%d）" % len(pts_px))
        f = fit(pts_px, pts_xy)
        k = f["k"]
        d_h = k * FX
        print("\n擬合：%d 點 殘差 平均 %.2f mm 最大 %.2f mm；鏡像=%s" % (len(pts_px), f["err"].mean(), f["err"].max(), f["flip"] == -1))
        print("k = %.4f mm/px → 軸心距相機 D_h ≈ %.0f mm；棋盤面 %.0f mm → 軸心高於棋盤面約 %.0f mm"
              % (k, d_h, D_BOARD, D_BOARD - d_h))
        json.dump(dict(arm=name, z0=z0, R=f["R"].tolist(), k=k, flip=f["flip"], pm=f["pm"].tolist(),
                       qm=f["qm"].tolist(), D_h=d_h, D_board=D_BOARD, axis_above_board=D_BOARD - d_h,
                       residual_mean=float(f["err"].mean()), residual_max=float(f["err"].max()), points=log),
                  open("%s/../calib_%s_auto.json" % (OUT, name), "w"), indent=1)
        print("已存 calib_%s_auto.json" % name)
    except Exception as e:
        print("停止：", e)
        arm.dash.ResetRobot()
    finally:
        if go:
            try:
                sync_move(arm, "JointMovJ(%f,%f,%f,%f)" % tuple(j_home))   # 回到原停放姿態
            except Exception:
                pass
            arm.dash.DisableRobot()
        cam.close()
        arm.close()


if __name__ == "__main__":
    main()
