"""棋子貼了膠帶（頂面平）時，吸盤不必對正中央：在中心周圍 MAX_OFF mm 內找最小偏移的吸取點，
讓吸取、放到料區（同偏移）的 IK、路線、支架都通過。只查詢，不動。

    python find_offsets.py <料區 O|X> <格:中心x,中心y:dz> ...
"""
import sys

import numpy as np

from common import connect_arm
from pickplace import STOCK, PIECE, Z_TOUCH_BOARD, counts, seg_clear, travel_points
from plan_cells import hits_post
from play import PARK, Z_HOVER
from world import load

MAX_OFF = 22.0
PRESS = 0.5


def main():
    to = sys.argv[1] + "疊"
    cal = load("B")
    dst0 = np.array(STOCK["stocks"][to]["xy"])
    n = counts()[to]
    park = np.array(PARK["B"][:2])
    B = connect_arm("B")
    try:
        B.wait_feed()
        r = float(B.pose()[3])
        ik = lambda xy, z: B.dash.InverseSolution(xy[0], xy[1], z, r, 0, 0).startswith("0,")   # noqa: E731
        for spec in sys.argv[2:]:
            cell, xy, dz = spec.split(":")
            c = np.array([float(v) for v in xy.split(",")])
            z_pick = Z_TOUCH_BOARD + float(dz) - PRESS
            z_place = STOCK["stocks"][to]["z_first"] + n * PIECE
            cands = sorted({(round(dx), round(dy)) for dx in np.arange(-MAX_OFF, MAX_OFF + 0.1, 2)
                            for dy in np.arange(-MAX_OFF, MAX_OFF + 0.1, 2) if np.hypot(dx, dy) <= MAX_OFF},
                           key=lambda o: np.hypot(*o))
            found = None
            for off in cands:
                s, d = c + off, dst0 + off
                if hits_post(cal, s, z_pick) or hits_post(cal, s, Z_HOVER) or hits_post(cal, d, z_place):
                    continue
                if not (seg_clear(cal, travel_points(park, s), Z_HOVER) and seg_clear(cal, travel_points(s, d), Z_HOVER)
                        and seg_clear(cal, travel_points(d, park), Z_HOVER)):
                    continue
                if ik(s, z_pick) and ik(s, Z_HOVER) and ik(d, z_place) and ik(d, Z_HOVER):
                    found = np.array(off, float)
                    break
            if found is None:
                print("格 %s：中心 %s，%d mm 內找不到可吸點" % (cell, c, MAX_OFF))
            else:
                print("格 %s：中心 %s → 偏移 %s（%.0f mm），吸取 Z=%.1f，半徑 %.0f" % (
                    cell, c, found, np.hypot(*found), z_pick, np.hypot(*(c + found))))
    finally:
        B.close()


if __name__ == "__main__":
    main()
