"""校正快速自我檢查（手臂不動）：讀兩台目前姿態 → 用校正檔預測方盒頂面在影像哪裡 → 在相機畫面找方盒 → 比較。

    python drift_check.py          # 印每台的偏差（像素、約略 mm），存標註圖 out/drift.jpg

相機或手臂被移動過，偏差就會變大：
- < 6 px（約 2 mm）：正常
- 6～15 px：還能用，但吸取可能偏，建議重新校正
- > 15 px 或找不到方盒：校正失效，要重跑 calib_box.py（或網頁上的「重新校正」）
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "MG400"))
from calib_box import face_scale, find_face   # noqa: E402
from plan_cells import face_px                # noqa: E402
from world import load                        # noqa: E402

OK_PX, WARN_PX = 6.0, 15.0


def check(img, poses):
    """poses：{"A": (x,y,z,j1), "B": …}。回傳 {arm: dict(pred, found, score, err_px, err_mm, level)}。"""
    out = {}
    for name, (x, y, z, j1) in poses.items():
        cal = load(name)
        pred = np.asarray(face_px(cal, (x, y), z, j1_deg=j1), float)
        scale = face_scale(z) if name == "B" else face_scale(z) / face_scale(cal["z0"])
        r = 170
        box = (int(pred[0]) - r, int(pred[1]) - r, int(pred[0]) + r, int(pred[1]) + r)
        try:
            found, ang, score = find_face(img, j1=None, search=box, scale=scale, arm=name)
        except Exception:
            found, score = None, 0.0
        if found is None or score < 0.6:
            out[name] = dict(pred=pred, found=None, score=score, err_px=None, err_mm=None, level="找不到方盒")
            continue
        e = float(np.hypot(*(found - pred)))
        d_face = cal["D_ref"] + (cal["z0"] - z)                  # 方盒頂面到相機
        out[name] = dict(pred=pred, found=found, score=score, err_px=e, err_mm=e * d_face / cal["fx"],
                         level="正常" if e < OK_PX else "建議重新校正" if e < WARN_PX else "校正失效")
    return out


def draw(img, res):
    vis = img.copy()
    for name, r in res.items():
        p = tuple(int(v) for v in r["pred"])
        cv2.drawMarker(vis, p, (255, 0, 0), cv2.MARKER_CROSS, 40, 3)
        if r["found"] is not None:
            f = tuple(int(v) for v in r["found"])
            cv2.circle(vis, f, 12, (0, 200, 0), 3)
            cv2.line(vis, p, f, (0, 0, 255), 2)
        cv2.putText(vis, "%s %s" % (name, "%.1fpx" % r["err_px"] if r["err_px"] is not None else "?"),
                    (p[0] + 20, p[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return vis


def main():
    from common import Camera, connect_arm, joint_angles
    poses = {}
    for name in ("A", "B"):
        a = connect_arm(name)
        try:
            a.wait_feed()
            p = a.pose()
            poses[name] = (float(p[0]), float(p[1]), float(p[2]), float(joint_angles(a)[0]))
        finally:
            a.close()
    cam = Camera()
    try:
        img = cam.grab(skip=3, timeout=20)
    finally:
        cam.close()
    res = check(img, poses)
    for name, r in res.items():
        if r["err_px"] is None:
            print("%s：%s（分數 %.2f）" % (name, r["level"], r["score"]))
        else:
            print("%s：偏差 %.1f px（方盒高度約 %.1f mm）分數 %.2f → %s" % (name, r["err_px"], r["err_mm"], r["score"], r["level"]))
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    cv2.imwrite(os.path.join(HERE, "out", "drift.jpg"), draw(img, res))
    return res


if __name__ == "__main__":
    main()


# ---------- 不用動手臂的檢查：底座（跟手臂固定在一起）在畫面上的位置 vs 基準照 ----------
REF_SCENE = os.path.join(HERE, "calib_ref_scene.png")      # 校正後、兩台都在停放點時拍的（2026-11-14 下午）
BASES = {"B 底座（上）": (480, 0, 780, 110), "A 底座（下）": (510, 605, 810, 720)}
BASE_OK_PX = 8.0                 # 約 3 mm；超過就要重新校正


def base_shift(img, ref=None):
    """回傳 (最大位移 px, {底座: (dx, dy, 相似度)})。找不到（被擋住、相似度 < 0.6）的底座不算。"""
    ref = cv2.imread(REF_SCENE) if ref is None else ref
    if ref is None:
        return None, {}
    out = {}
    for name, (x0, y0, x1, y1) in BASES.items():
        t = cv2.cvtColor(ref[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        m = 60
        sx0, sy0 = max(0, x0 - m), max(0, y0 - m)
        g = cv2.cvtColor(img[sy0:min(img.shape[0], y1 + m), sx0:min(img.shape[1], x1 + m)], cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        out[name] = (loc[0] + sx0 - x0, loc[1] + sy0 - y0, float(mx))
    good = [np.hypot(dx, dy) for dx, dy, s in out.values() if s >= 0.6]
    return (max(good) if good else None), out
