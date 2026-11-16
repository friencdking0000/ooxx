"""現場校正（手動教點）：用 jog.py 記錄的「格線四個交叉點＋中心點」算出每台手臂的世界座標。

    python fit_teach.py [teach_cells.json]      # 只計算、印出結果並存 teach_fit.json，不改任何設定

做法：
- 板子的標準形狀用實際尺寸（板寬 270 mm＝點 1→2→3 方向、板長 295 mm＝點 1→4→7 方向）配合畫面上膠帶的位置算出，
  單位 mm、以中間格中心為原點、第一軸＝畫面上下、第二軸＝畫面左右（BOARD_* 常數）。
- 對每台手臂，只用「四個交叉點」擬合「旋轉＋平移」（不縮放，尺寸是量的）；4 點時殘差最大且 > OUTLIER_MM 的那點剔除。
  中心點不參加擬合，只拿來驗算（它的殘差才是獨立的檢查）。舊的「格子中心」記錄只在交叉點不到 3 個時才拿來補，而且會標出來。
- 記錄時間比本次交叉點早超過 STALE_H 小時的點不用（teach_cells.json 會跨次保留）。
- 記錄的是手臂 TCP；吸盤就在 TCP 正下方，所以直接當吸盤位置用（方盒的 4 mm 偏移只跟相機看方盒有關）。
- 格子中心＝外圈格「膠帶中心線到板邊」的中點（實際的格子中心）。舊的 cell_plan.json 是把外圈格當成跟中間格一樣大往外推，
  所以外圈兩排會跟舊值差約 3.5 mm，這是定義不同，不是誤差。
- 輸出：九格中心與四個交叉點在該手臂座標的位置、每點殘差、中心點驗算誤差、各點高度（棋盤面）、跟舊校正的差。
"""
import json
import os
import sys

import numpy as np

from world import rigid2d

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 板子形狀（mm）。2026-11-15 實測：板子 270（點1→2→3 方向）×295（點1→4→7 方向）mm；
# 膠帶中心線間距：前後 116.2、左右 91.8 mm（空棋盤畫面逐像素量 148／117 px，兩台手臂教點量到 115～117／92～94，一致）；
# 膠帶寬約 16 mm；兩條膠帶相對板子中心對稱（畫面上兩側各差 1 px 內）。
# 格子中心＝「看得到的格子範圍」的中心：中間格在四條膠帶內側之間（＝四個交叉點的中心）；
# 外圈格在膠帶外緣與板邊之間 → 離中間格 ±107.0（前後）、±94.2（左右）mm。
# （之前用 board.find_grid 的格線位置算出 110.4 mm，那個偵測差了約 7 px，已不用。）
BOARD_ROWS = np.array([-107.0, 0.0, 107.0])      # 第一軸：畫面上（遠離 A、靠近 B）→ 畫面下
BOARD_COLS = np.array([-94.2, 0.0, 94.2])        # 第二軸：畫面左 → 右
CROSS_ROW = 58.1                                 # 交叉點離中心：前後 ±58.1、左右 ±45.9
CROSS_COL = 45.9
OUTLIER_MM = 4.0
STALE_H = 2.0
TEACH_FILE = os.path.join(ROOT, "Arena", "teach_cells.json")   # jog.py 記錄、這裡讀（jog.py 從這裡 import，路徑只有一份）
# 格線四個交叉點的鍵 → 夾在哪四格之間（畫面格號）；jog.py 也用這份
CROSS = {"c0": (0, 1, 3, 4), "c1": (1, 2, 4, 5), "c2": (3, 4, 6, 7), "c3": (4, 5, 7, 8)}


def template():
    t = {str(r * 3 + c): np.array([BOARD_ROWS[r], BOARD_COLS[c]]) for r in range(3) for c in range(3)}
    t.update({"c0": np.array([-CROSS_ROW, -CROSS_COL]), "c1": np.array([-CROSS_ROW, CROSS_COL]),
              "c2": np.array([CROSS_ROW, -CROSS_COL]), "c3": np.array([CROSS_ROW, CROSS_COL])})
    return t



def fresh_points(points, detail):
    """丟掉比最新一個交叉點早 STALE_H 小時以上記錄的點（沒有時間紀錄的舊點也丟）。"""
    import datetime
    def ts(k):
        try:
            return datetime.datetime.strptime(detail[k]["time"], "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            return None
    cross_t = [ts(k) for k in CROSS if k in points and ts(k)]
    if not cross_t:
        return dict(points), []
    newest = max(cross_t)
    keep = {k: v for k, v in points.items() if ts(k) is not None and newest - ts(k) <= STALE_H * 3600}
    return keep, sorted(set(points) - set(keep))


def cells_from_crosses(points):
    """四個交叉點都有時：中間格中心＝四點平均，前後／左右方向沿著膠帶（這台手臂實際量到的方向），
    外圈格沿這兩個方向各推 BOARD_ROWS／BOARD_COLS。交叉點本身就用量到的值。"""
    if not all(k in points for k in ("c0", "c1", "c2", "c3")):
        return None
    c = {k: np.array(points[k][:2], float) for k in ("c0", "c1", "c2", "c3")}
    m = (c["c0"] + c["c1"] + c["c2"] + c["c3"]) / 4
    u = (c["c2"] + c["c3"] - c["c0"] - c["c1"]) / 2
    v = (c["c1"] + c["c3"] - c["c0"] - c["c2"]) / 2
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    out = {str(r * 3 + k): (m + BOARD_ROWS[r] * u + BOARD_COLS[k] * v).round(2).tolist() for r in range(3) for k in range(3)}
    out.update({k: c[k].round(2).tolist() for k in c})
    return out


def fit_arm(points, detail=None):
    """points：{key: [x, y, z]}。用交叉點擬合；中心點（'4'）只驗算；交叉點不到 3 個才拿其他格子中心補。"""
    T = template()
    points, stale = fresh_points(points, detail or {})
    keys = [k for k in CROSS if k in points]
    filled = []
    if len(keys) < 3:
        filled = [k for k in points if k in T and k not in keys and k != "4"]
        keys += filled
    if len(keys) < 3:
        return None
    P = np.array([T[k] for k in keys])
    Q = np.array([points[k][:2] for k in keys], float)
    keep = list(range(len(keys)))
    while True:
        R, t, _ = rigid2d(P[keep], Q[keep])
        res = np.hypot(*(Q - (P @ R.T + t)).T)
        worst = max(keep, key=lambda i: res[i])
        if res[worst] > OUTLIER_MM and len(keep) > 3:
            keep.remove(worst)
            continue
        break
    z = {k: float(points[k][2]) for k in keys if len(points[k]) > 2}
    check = None
    if "4" in points and "4" not in keys:                # 中心點：沒參加擬合，是獨立的驗算
        check = round(float(np.hypot(*(np.array(points["4"][:2]) - (R @ T["4"] + t)))), 2)
    return dict(R=R.tolist(), t=t.tolist(), rot_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
                center_check_mm=check, stale=stale, filled_from_cells=filled,
                used=[keys[i] for i in keep], rejected=[keys[i] for i in range(len(keys)) if i not in keep],
                residual={keys[i]: round(float(res[i]), 2) for i in range(len(keys))},
                rms=float(np.sqrt(np.mean(res[keep] ** 2))),
                cells=cells_from_crosses(points) or {k: (R @ T[k] + t).round(2).tolist() for k in T}, board_z=z)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else TEACH_FILE
    teach = json.load(open(path, encoding="utf-8"))
    plan = json.load(open(os.path.join(HERE, "cell_plan.json"), encoding="utf-8"))
    out = {}
    for arm in ("B", "A"):
        f = fit_arm(teach.get(arm, {}), teach.get("_detail", {}).get(arm, {}))
        if f is None:
            print("%s：記錄的點不到 3 個，不能算" % arm)
            continue
        out[arm] = f
        print("\n== %s 臂：用 %s 擬合（剔除 %s），均方根殘差 %.1f mm，旋轉 %.1f°" % (
            arm, f["used"], f["rejected"] or "無", f["rms"], f["rot_deg"]))
        print("   每點殘差 mm：", f["residual"])
        print("   中心點驗算：%s" % ("沒記錄" if f["center_check_mm"] is None else "差 %.1f mm" % f["center_check_mm"]))
        if f["filled_from_cells"]:
            print("   ⚠ 交叉點不到 3 個，拿舊的格子中心補：", f["filled_from_cells"])
        if f["stale"]:
            print("   （太舊、沒用的記錄：%s）" % f["stale"])
        if f["board_z"]:
            print("   記錄時的高度 Z：", {k: round(v, 1) for k, v in f["board_z"].items()})
        d = [np.hypot(*(np.array(f["cells"][str(i)]) - np.array(plan[str(i)][arm]))) for i in range(9)]
        print("   九格中心跟舊校正（cell_plan）差：", [round(x, 1) for x in d], "mm")
    if "A" in out and "B" in out:                      # 兩台各自算出的同一個點，用來求 A→B 轉換、互相驗證
        T = template()
        PA = np.array([out["A"]["cells"][k] for k in T]); PB = np.array([out["B"]["cells"][k] for k in T])
        R, t, _ = rigid2d(PA, PB)
        out["A_to_B"] = dict(R=R.tolist(), t=t.tolist(), rot_deg=float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))))
        print("\nA→B：旋轉 %.1f°、A 底座在 B 座標 %s（兩底座相距 %.0f mm）" % (
            out["A_to_B"]["rot_deg"], np.round(t, 1), np.hypot(*t)))
    from common import atomic_write
    atomic_write(os.path.join(HERE, "teach_fit.json"), json.dumps(out, indent=1))
    print("\n已存 Calib/teach_fit.json（還沒套用到對弈程式）")


if __name__ == "__main__":
    main()
