"""偏心吸取的範圍：每台手臂、每一格，棋子要放在格子正中央時，吸盤得偏多少才到得了（只問控制器 IK，手臂不會動）。

    python grip_reach.py            # 連兩台手臂問 IK；不要跟其他連手臂的程式（jog.py、server.py）同時開
輸出 Calib/grip_reach.json，並印出表格。

- 格子位置用手動教點的結果（teach_fit.json，各臂自己的座標）。
- 放子高度：該臂四個交叉點記錄的 Z 擬合成棋盤平面 ＋ 棋子厚 3.7 ＋ 放開前留的 1.5 mm（跟 server 一樣），
  另外也查往下 SEARCH mm（接觸偵測會往下探）。
- 沿「底座 → 格子中心」那條線二分找 IK 的可達半徑 [r_lo, r_hi]；格子中心在範圍外就要偏：
  遠排吸棋子靠底座那側、近排吸遠離底座那側，偏移量＝超出的距離＋MARGIN。
- 料區（吸取）那一側也要用同一個偏移到得了，所以另外印出各疊高時的可達半徑。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "MG400"))

from common import connect_arm             # noqa: E402
from play import Z_HOVER                   # noqa: E402

PIECE = 3.7                                # mm：棋子厚
RELEASE = 1.5                              # mm：放開時吸盤離棋盤上棋子頂面的高度（server：Z_TOUCH_B + 1.5）
SEARCH = 6.0                               # mm：接觸偵測往下探的餘裕
MARGIN = 3.0                               # mm：離 IK 邊界留的距離（推到關節限位 0.1° 就報警）
ROW_NAMES = {"B": {6: "點1", 7: "點2", 8: "點3", 3: "點4", 4: "點5", 5: "點6", 0: "點7", 1: "點8", 2: "點9"},
             "A": {i: "點%d" % (i + 1) for i in range(9)}}


def board_plane(fit_arm):
    P = np.array([(*fit_arm["cells"][k][:2], fit_arm["board_z"][k]) for k in ("c0", "c1", "c2", "c3")], float)
    coef, *_ = np.linalg.lstsq(np.c_[P[:, :2], np.ones(len(P))], P[:, 2], rcond=None)
    return lambda xy: float(coef @ [xy[0], xy[1], 1.0])


def radial_window(ik, ang, z, lo=120.0, hi=470.0):
    """沿角度 ang 的射線，IK 可達的半徑範圍（假設是一段連續區間）。找不到回 None。"""
    u = np.array([np.cos(ang), np.sin(ang)])
    rs = np.arange(lo, hi, 5.0)
    ok = [ik(r * u, z) for r in rs]
    if not any(ok):
        return None
    i0 = ok.index(True)
    i1 = len(ok) - 1 - ok[::-1].index(True)

    def edge(a, b):                         # a 可達、b 不可達
        for _ in range(12):
            m = (a + b) / 2
            a, b = (m, b) if ik(m * u, z) else (a, m)
        return a
    r_lo = rs[i0] if i0 == 0 else edge(rs[i0], rs[i0 - 1])
    r_hi = rs[i1] if i1 == len(rs) - 1 else edge(rs[i1], rs[i1 + 1])
    return float(r_lo), float(r_hi)


def main():
    fit = json.load(open(os.path.join(HERE, "teach_fit.json"), encoding="utf-8"))
    out = {"note": "r_lo/r_hi：IK 可達半徑（mm，該臂座標）；need：吸盤要往外(+)/往底座(-)偏多少 mm（0＝吸中心）",
           "PIECE": PIECE, "RELEASE": RELEASE, "SEARCH": SEARCH, "MARGIN": MARGIN}
    for name in ("B", "A"):
        arm = connect_arm(name)
        try:
            arm.wait_feed()
            r4 = float(arm.pose()[3])

            def ik(p, z):
                s = arm.dash.InverseSolution(float(p[0]), float(p[1]), float(z), r4, 0, 0)
                if not s:
                    raise ConnectionError("%s 沒有回應" % name)
                return s.startswith("0,")
            zb = board_plane(fit[name])
            cells = {}
            print("\n== %s 臂（格號＝畫面格；括號是 jog 頁上這台的點號）" % name)
            print("  格      半徑   放子Z    可達半徑(放子Z)   可達半徑(再低%.0f)  需要偏移" % SEARCH)
            for c in range(9):
                p = np.array(fit[name]["cells"][str(c)][:2], float)
                r, ang = float(np.hypot(*p)), float(np.arctan2(p[1], p[0]))
                zp = zb(p) + PIECE + RELEASE
                w1 = radial_window(ik, ang, zp)
                w2 = radial_window(ik, ang, zp - SEARCH)
                lo = max(w1[0], w2[0]) + MARGIN
                hi = min(w1[1], w2[1]) - MARGIN
                need = 0.0 if lo <= r <= hi else (lo - r if r < lo else hi - r)
                hover_ok = ik(p + need * np.array([np.cos(ang), np.sin(ang)]), Z_HOVER)
                cells[str(c)] = dict(xy=p.round(2).tolist(), r=round(r, 1), z_place=round(zp, 1),
                                     window=[round(w1[0], 1), round(w1[1], 1)], window_low=[round(w2[0], 1), round(w2[1], 1)],
                                     need=round(need, 1), hover_ok=bool(hover_ok))
                print("  %d(%s)  %5.1f  %6.1f   %5.1f～%5.1f      %5.1f～%5.1f       %s" % (
                    c, ROW_NAMES[name][c], r, zp, w1[0], w1[1], w2[0], w2[1],
                    "吸中心" if need == 0 else ("%+.1f mm（%s）" % (need, "往外" if need > 0 else "往底座"))))
            # 料區：疊高 1～6 顆時（頂面＝桌面附近＋n×厚），各方向的可達半徑；料區大致在棋盤左右兩側
            z4 = zb(np.array(fit[name]["cells"]["4"][:2]))
            stock = {}
            for n in (1, 3, 6):
                zs = z4 + n * PIECE - 9.0                         # 吸取時往下探 9 mm 的餘裕（mover.plan）
                stock[n] = {}
                for deg in (-60, -40, -20, 20, 40, 60):
                    w = radial_window(ik, np.radians(deg), zs)
                    stock[n][deg] = None if w is None else [round(w[0], 1), round(w[1], 1)]
            print("  料區可達半徑（疊 n 顆、各方向角度）：")
            for n, d in stock.items():
                print("   %d 顆：" % n + "  ".join("%+d°:%s" % (k, "—" if v is None else "%.0f～%.0f" % tuple(v)) for k, v in d.items()))
            out[name] = dict(cells=cells, stock=stock, r_used=r4)
        finally:
            arm.close()
    with open(os.path.join(HERE, "grip_reach.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已存 Calib/grip_reach.json")


if __name__ == "__main__":
    main()
