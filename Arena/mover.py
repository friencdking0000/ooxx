"""雙臂的基本動作：連線、移動（縮回→轉向→伸出）、檢查（IK、支架、路線、跟停著那台的距離）、接觸偵測、抬起。
搬運的規劃（哪一台、吸哪裡、放哪裡、轉送）在 carrier.py。一次只動一台，另一台在停放點。

吸取／放下都用「接觸偵測」：慢慢往下，J2 電流偏離正常值＝碰到（2026-11-14 實測，B 在 O 疊三次都在 -157.3～-157.5 碰到）。
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Calib"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "MG400"))

from common import connect_arm, sync_move                                           # noqa: E402
from plan_cells import hits_post                                       # noqa: E402
from play import PARK, Z_HOVER, a_to_b_transform, CLEAR_MM            # noqa: E402
from world import load                                                 # noqa: E402

Z_TOUCH_B = -154.82          # B 碰到格 4 標準棋子頂面
DZ_A = 3.0                   # A 的 Z 比 B 高約 3 mm（兩臂方盒頂面高度差）
DI_J2 = 0.12                 # 接觸：J2 電流偏離剛剛的正常值（A）
TRAVEL_SPEED = 40           # %：搬運途中的速度（原 15，使用者嫌慢；往下探、起抬仍是慢速）
PRESS_B = 1.0               # mm：B 碰到後再往下壓這麼多才開真空
VAC_DWELL = 2.5             # 秒：開真空後壓在棋子上停留的時間（原 0.7 太短）


# ---- 現場調過的參數（tuning.json；舊的自動調整寫的，沿用）----
TUNING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning.json")


def _load_tuning():
    global TRAVEL_SPEED, VAC_DWELL, PRESS_B
    try:
        t = json.load(open(TUNING_FILE, encoding="utf-8"))
        TRAVEL_SPEED = int(min(60, max(15, t.get("TRAVEL_SPEED", TRAVEL_SPEED))))
        VAC_DWELL = float(min(4.0, max(1.0, t.get("VAC_DWELL", VAC_DWELL))))
        PRESS_B = float(min(2.5, max(0.5, t.get("PRESS_B", PRESS_B))))
    except (OSError, ValueError):
        pass


_load_tuning()


def board_dz(cell):
    try:
        return float(json.load(open(os.path.join(os.path.dirname(HERE), "Calib", "board_dz.json"),
                                    encoding="utf-8")).get(str(cell), 0.0))
    except Exception:
        return 0.0


class Arm:
    def __init__(self, name, T):
        self.name = name
        self.halted = lambda: False               # 伺服器的「停止」（carrier.Scene.attach 接上）：touch 不會再把關掉的手臂上電
        self.arm = connect_arm(name)
        self.arm.wait_feed()
        self.cal = load(name)
        self.r = float(self.arm.pose()[3])
        self.T = T                                   # A→B：B = R·A + t
        self.dz = DZ_A if name == "A" else 0.0
        # A 的舊停放點 (60,340) 伸出 345 mm（使用者：為什麼不能像 B 那樣收緊）→ 同方向收到 240 mm（IK、支架已查）
        self.park = np.array(PARK[name][:2]) if name == "B" else np.array([42.0, 236.0])

    # ---- 座標 ----
    def loc(self, pB):
        pB = np.asarray(pB, float)
        if self.name == "B":
            return pB
        R, t = self.T
        return R.T @ (pB - t)

    def base_B(self):
        return np.zeros(2) if self.name == "B" else self.T[1]

    # ---- 檢查 ----
    def ik(self, p, z):
        r = self.arm.dash.InverseSolution(p[0], p[1], z, self.r, 0, 0)
        if not r:                                    # 手臂被重開電源後舊連線會一直回空字串：這不是「搆不到」，是斷線
            raise ConnectionError("手臂 %s 沒有回應（連線斷了）" % self.name)
        return r.startswith("0,")

    def post(self, p, z):
        return hits_post(self.cal, p, z)

    R_INS = (215.0, 200.0, 185.0, 170.0)     # 轉向半徑候選：先縮回到這麼小再轉（使用者：手臂伸那麼遠轉當然會撞）

    @staticmethod
    def route_pts(p0, p1, rin, n=25):
        """縮回（沿 p0 方向到 rin）→ 在 rin 轉到 p1 方向 → 伸出到 p1。"""
        a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
        r0, r1 = np.hypot(*p0), np.hypot(*p1)
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi              # 走短的那一邊
        pts = [((1 - t) * r0 + t * rin) * np.array([np.cos(a0), np.sin(a0)]) for t in np.linspace(0, 1, n)]
        pts += [rin * np.array([np.cos(a0 + t * da), np.sin(a0 + t * da)]) for t in np.linspace(0, 1, n)]
        pts += [((1 - t) * rin + t * r1) * np.array([np.cos(a1), np.sin(a1)]) for t in np.linspace(0, 1, n)]
        return pts

    def route_rin(self, p0, p1):
        """可用的轉向半徑：整條路線（懸停高度）不碰支架、轉向圈 IK 可達。沒有回 None。"""
        for rin in self.R_INS:
            if rin > min(np.hypot(*p0), np.hypot(*p1)) + 1:
                rin = min(np.hypot(*p0), np.hypot(*p1))
            pts = self.route_pts(np.asarray(p0, float), np.asarray(p1, float), rin)
            if any(hits_post(self.cal, q, Z_HOVER) for q in pts):
                continue
            if all(self.ik(q, Z_HOVER) for q in pts[25:50:6]):
                return rin
        return None

    def leg_ok(self, p0, p1):
        return self.route_rin(p0, p1) is not None

    # ---- 動作 ----
    def goto(self, p):
        """到本機座標 p 的懸停高度：先升高 → 縮回到轉向半徑 → 轉 → 伸出（兩台同一種走法）。"""
        a = self.arm
        q = a.pose()
        if q[2] < Z_HOVER - 1:
            sync_move(a, "MovL(%f,%f,%f,%f)" % (q[0], q[1], Z_HOVER, self.r))
        p0, p = a.pose()[:2], np.asarray(p, float)
        rin = self.route_rin(p0, p)
        if rin is None and np.hypot(*p0) < self.R_INS[0]:
            # 手臂被人推到離底座很近（2026-11-15 A 停在半徑 191 mm）：這麼近在移動高度轉不過去，先沿原方向直線伸到 215 mm 再走正常路線
            q = a.pose()
            out = p0 / max(np.hypot(*p0), 1e-6) * self.R_INS[0]
            if self.ik(out, q[2]) and self.ik(out, Z_HOVER) and not self.post(out, Z_HOVER) and self.route_rin(out, p) is not None:
                sync_move(a, "MovL(%f,%f,%f,%f)" % (out[0], out[1], max(q[2], Z_HOVER), self.r))
                p0 = a.pose()[:2]
                rin = self.route_rin(p0, p)
        if rin is None:
            raise RuntimeError("%s 到 %s 沒有不碰支架的路線" % (self.name, np.round(p, 0)))
        a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p[1], p[0])
        if np.hypot(*p0) > rin + 2:
            sync_move(a, "MovL(%f,%f,%f,%f)" % (rin * np.cos(a0), rin * np.sin(a0), Z_HOVER, self.r))
        da = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
        for k in (1, 2, 3):                                        # 轉向分三段，確保走短邊、保持小半徑
            ak = a0 + da * k / 3
            sync_move(a, "MovJ(%f,%f,%f,%f)" % (rin * np.cos(ak), rin * np.sin(ak), Z_HOVER, self.r))
        sync_move(a, "MovL(%f,%f,%f,%f)" % (p[0], p[1], Z_HOVER, self.r))

    def go_park(self):
        self.goto(self.park)

    def reenable(self):
        """碰撞保護觸發（模式 9）後：關碰撞偵測 → 關電、清報警、上電 → 開回碰撞偵測（壓著時會一直重觸發，要先關）。"""
        d = self.arm.dash
        d.sendRecvMsg("SetCollisionLevel(0)")
        try:
            d.DisableRobot()
            time.sleep(0.3)
            d.ClearError()
            time.sleep(0.3)
            d.EnableRobot()
            time.sleep(1.2)
        finally:
            d.sendRecvMsg("SetCollisionLevel(3)")

    COMPRESS = 4.5           # 伸縮吸盤碰到後先壓縮這麼多才偵測到（實測：5 顆疊幾何頂面 -152.4，接觸 -156.9）

    def touch(self, p, z_est, vacuum_on, over=6.0, below=9.0, speed=5, z_surf=None, h_min=2.0):
        """在 p 從 z_est+over 慢慢往下，碰到就停，回傳接觸 Z 或 None（到底沒碰到）。vacuum_on＝吸取：碰到、而且接觸點
        比底面 z_surf 高出 h_min 以上才開真空（下面真的是棋子）。"""
        import dual_sync
        a = self.arm
        if getattr(a, "feed_slow", False):          # 回饋只有 5 Hz：往下探放更慢，才不會壓太深才停
            speed = min(speed, 2)
        sync_move(a, "MovL(%f,%f,%f,%f)" % (p[0], p[1], z_est + over, self.r))
        a.dash.SpeedFactor(speed)                    # 真空先不開：碰到、而且高度對，才開（曾經開著真空把整塊棋盤吸起來）
        z_floor = z_est - below
        dual_sync.check(a.move.sendRecvMsg("MovL(%f,%f,%f,%f)" % (p[0], p[1], z_floor, self.r)), "MovL down")
        t0, hist, started, zc = time.perf_counter(), [], False, None
        while time.perf_counter() - t0 < 15:
            rec = a.last_rec
            z, mode, i2 = float(rec["tool_vector_actual"][2]), int(rec["robot_mode"]), float(rec["i_actual"][1])
            if mode == 7 or z < z_est + over - 0.5:
                started = True
            if mode == 9:
                zc = z
                break
            if started and time.perf_counter() - t0 > 0.25:
                if len(hist) >= 15 and abs(i2 - float(np.median(hist[-25:-5]))) > DI_J2:
                    zc = z
                    break
                hist.append(i2)
            if started and abs(z - z_floor) < 0.2 and mode == 5:
                break
            time.sleep(0.005)
        a.dash.ResetRobot()
        time.sleep(0.2)
        if self.halted():                       # 往下探時按了停止（伺服器已關電）：不要再上電、開真空、繼續動
            a.dash.sendRecvMsg("DOExecute(1,0)")
            raise RuntimeError("已停止")
        alarmed = a.mode() == 9
        if alarmed:                             # 碰撞偵測先觸發：暫關、清、再開（壓著會一直重觸發）
            self.reenable()
        if a.mode() != 5:
            a.dash.EnableRobot()
            time.sleep(1.0)
        a.dash.SpeedFactor(TRAVEL_SPEED)
        if vacuum_on:
            # 下面有沒有棋子：接觸點（補回吸盤壓縮量）要比底下的桌面／棋盤面高出至少半顆棋子
            h = None if (zc is None or z_surf is None) else zc + self.COMPRESS - z_surf
            if zc is None or (h is not None and h < h_min):
                print("[接觸] %s：%s → 不開真空" % (self.name, "沒碰到" if zc is None else
                      "碰到 Z=%.1f，只比底面高 %.1f mm，下面不是棋子" % (zc, h)), flush=True)
                return None
            print("[接觸] %s：Z=%.1f，比底面高 %s mm → 開真空" % (self.name, zc, "?" if h is None else "%.1f" % h), flush=True)
            if not alarmed:                              # 兩台都多壓一點再開真空（使用者 2026-11-15：再壓下去 1 mm、吸久一點，不然吸到一半掉）         # A 吸不住（碰到就停，吸盤沒貼緊）：再多壓 2 mm 才開真空（已觸發碰撞保護就不再壓）
                try:
                    a.dash.SpeedFactor(5)
                    sync_move(a, "MovL(%f,%f,%f,%f)" % (p[0], p[1], zc - (2.0 if self.name == "A" else PRESS_B), self.r))
                except Exception as e:
                    print("[接觸] %s 多壓時：%s" % (self.name, str(e)[:60]), flush=True)
                    self.reenable()
                a.dash.SpeedFactor(TRAVEL_SPEED)
            a.dash.sendRecvMsg("DOExecute(1,1)")
            time.sleep(VAC_DWELL)                       # 持壓：開真空後壓著等負壓建立再抬（使用者：沒有持壓時間會吸不起來）
        return zc

    def lift(self):
        q = self.arm.pose()
        if q[2] < Z_HOVER - 12:                      # 剛吸起來：前 8 mm 慢慢抬，棋子才不會被甩掉
            try:
                self.arm.dash.SpeedFactor(4)
                sync_move(self.arm, "MovL(%f,%f,%f,%f)" % (q[0], q[1], q[2] + 8.0, self.r))
            finally:
                self.arm.dash.SpeedFactor(TRAVEL_SPEED)
        sync_move(self.arm, "MovL(%f,%f,%f,%f)" % (q[0], q[1], Z_HOVER, self.r))


class Mover:
    def __init__(self):
        self.T = a_to_b_transform()
        self.arms = {n: Arm(n, self.T) for n in ("B", "A")}

    def clear_of_other(self, active, ptsB):
        """active 那台伸到 ptsB 時，跟停在停放點的另一台（底座→停放點線段）距離要 ≥ CLEAR_MM。"""
        from play import seg_dist
        other = self.arms["A" if active.name == "B" else "B"]
        R, t = self.T
        o0 = other.base_B()
        o1 = other.park if other.name == "B" else R @ other.park + t
        b0 = active.base_B()
        return all(seg_dist(b0, np.asarray(p, float), o0, o1) >= CLEAR_MM for p in ptsB)

    def park_all(self):
        """兩台回停放點（還上電的才動），一律關真空、關電；回停放點失敗也照樣關。"""
        for a in self.arms.values():
            try:
                if a.arm.mode() == 5:
                    a.go_park()
            except Exception:
                pass
            finally:
                try:
                    a.arm.dash.sendRecvMsg("DOExecute(1,0)")
                    a.arm.dash.DisableRobot()
                except Exception:
                    pass

    def close(self):
        for a in self.arms.values():
            try:
                a.arm.dash.sendRecvMsg("DOExecute(1,0)")
                a.arm.dash.DisableRobot()
            except Exception:
                pass
            a.arm.close()
