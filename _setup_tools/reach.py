"""只算不動：每台手臂到九格格子中心到不到得了（IK、支架、路線）；到不了就找最近、到得了的落點（偏多少）。"""
import os, sys, json
R = os.path.join(os.environ["USERPROFILE"], "Desktop", "OOXX-robot-v1.1")
sys.path.insert(0, os.path.join(R, "Arena")); sys.path.insert(0, os.path.join(R, "Calib"))
os.chdir(os.path.join(R, "Arena"))
import numpy as np
import grip_test as G
import mover
from common import OrbbecCamera

def why_at(arm, p, z):
    w = []
    if not arm.ik(p, z): w.append("IK(放子高度)")
    if not arm.ik(p, G.Z_HOVER): w.append("IK(移動高度)")
    if arm.post(p, G.Z_HOVER): w.append("支架")
    if not (arm.leg_ok(arm.park, p) and arm.leg_ok(p, arm.park)): w.append("路線")
    return w

fit = json.load(open(G.FIT, encoding="utf-8")); plan = json.load(open(G.PLAN, encoding="utf-8"))
cam = OrbbecCamera(depth=True); M = mover.Mover()
try:
    sc = G.Scene(cam, plan, fit)
    _, piles = sc.measure(M.arms["B"].cal)
    sc.use_cam_posts()
    pile = {s["kind"]: s for s in sorted(piles, key=lambda s: s["n"]) if s["kind"] in "OX"}
    out = {}
    for name in ("A", "B"):
        arm = M.arms[name]; zb = sc.plane[name]
        kind = "X" if name == "A" else "O"
        pp = pile[kind]["arm_xy"][name] if kind in pile else None
        print("\n== %s 臂（%s 疊在 %s）" % (name, kind, None if pp is None else pp.round(1)), flush=True)
        for cell in range(9):
            tgt = np.array(fit[name]["cells"][str(cell)][:2], float)
            z = zb(tgt) + G.PIECE - 6
            post, mg = sc.post_margin(name, tgt)
            w = why_at(arm, tgt, z)
            if pp is not None and not w and not arm.leg_ok(pp, tgt):
                w.append("料疊→格子路線")
            row = dict(tgt=tgt.round(1).tolist(), r=round(float(np.hypot(*tgt)), 1), post=post, margin=round(float(mg), 1), why=w)
            if w:
                best = None
                for rr in np.arange(2.0, 46.0, 2.0):
                    for a in np.radians(np.arange(0, 360, 15)):
                        d = rr * np.array([np.cos(a), np.sin(a)])
                        p = tgt + d
                        if not why_at(arm, p, z) and (pp is None or arm.leg_ok(pp, p)):
                            best = d; break
                    if best is not None: break
                row["shift"] = None if best is None else best.round(1).tolist()
            out["%s%d" % (name, cell)] = row
            print("格 %d（你的 %d）：中心 %s 半徑 %.0f，離%s %.0f mm → %s" % (
                cell, cell + 1, tgt.round(1), row["r"], post, mg,
                "可以" if not w else "到不了（%s）；最近可到的落點要偏 %s mm（%s）" % (
                    "、".join(w), "45 以上" if row.get("shift") is None else "%.0f" % np.hypot(*row["shift"]), row.get("shift"))), flush=True)
    json.dump(out, open(os.path.join(G.OUT, "reach.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
finally:
    M.close(); cam.close()
