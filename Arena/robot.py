"""B 臂下子：從某一疊（O疊／X疊）頂端吸一顆 → 放到棋盤某格 → 回停放。沿用 Calib 的校正、路線與安全檢查。

高度（2026-11-14 實測）：B 碰棋盤上棋子頂面 Z=-154.82；棋子厚 3.7 mm；疊上吸取多壓 5 mm；
放到棋盤時棋子底離板 1.5 mm 放開。路線：直線縮回 R_IN → 小半徑轉向 → 直線伸出（pickplace.travel）。
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Calib"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "MG400"))

from common import connect_arm, enable, sync_move           # noqa: E402
from pickplace import STOCK, Z_TOUCH_BOARD, seg_clear, travel, travel_points   # noqa: E402
from plan_cells import hits_post                             # noqa: E402
from play import PARK, R_IN, Z_HOVER                         # noqa: E402
from world import load                                       # noqa: E402

PIECE = STOCK["piece_mm"]
STACK_PRESS = 3.0      # 疊頂高度改用深度實測（STACK_Z）後只需多壓一點；舊的推算法要壓 5 mm（碰撞偵測 3 級會報警）
STD_TOP_DEPTH = 540.0 - 3.7   # 格 4 標準棋子頂面離相機的距離：B 碰到它時 Z=Z_TOUCH_BOARD
STACK_Z = {}            # 料區實測：{疊: (量測時頂端吸盤接觸 Z, 量測時顆數, 每顆厚度)}，server 開新局時更新


def stack_top_z(stack, n_in_stack):
    """疊頂（第 n 顆頂面）吸盤接觸的 Z：有實測就用實測推，沒有就用舊的推算。"""
    if stack in STACK_Z:
        z0, n0, th = STACK_Z[stack]
        return z0 - (n0 - n_in_stack) * th
    return STOCK["stocks"][stack]["z_first"] - 1.5 + (n_in_stack - 1) * PIECE
Z_RELEASE = Z_TOUCH_BOARD + 1.5
# 棋盤不平（2026-11-14 深度實測：格 5、7、8 那側比格 4 高 1～3 mm）：每格棋盤面比格 4 高多少（mm），由 board_height.py 量
BOARD_DZ_FILE = os.path.join(os.path.dirname(HERE), "Calib", "board_dz.json")


def board_dz(cell):
    try:
        import json
        return float(json.load(open(BOARD_DZ_FILE, encoding="utf-8")).get(str(cell), 0.0))
    except Exception:
        return 0.0
DROP_MM = 25.0          # 交接：棋子底部停在掌心上方這麼高放開（手臂不碰手）
TRAVEL_ABOVE = 35.0     # 交接：移動時比放開點再高這麼多
Z_TRAVEL_MAX = 30.0     # 交接：移動高度上限（再高方盒會太靠近相機）
STAGE_MM = 110.0        # 交接：先停在手掌內側這麼遠（不擋住手，相機再瞄準一次）
HAND_R_MIN, HAND_R_MAX = 285.0, 400.0   # 交接：手掌離 B 底座的距離（旁邊點最近 R_IN+5，與手掌至少差 60 mm）
# 快速交接（使用者：手伸出去等不能超過 3 秒）：先吸好棋子在待命點等，看到手就直接過去
READY_XY = np.array([240.0, 0.0])      # 待命點：棋盤靠 B 那一側上方，手掌區（285～400 mm）內側
Z_READY = -20.0                        # 待命高度：手掌離棋盤 ≤ 75 mm 時不用再升高
FAST_SPEED = 40                        # 平移到掌心上方（離掌心至少 60 mm）
DESCENT_SPEED = 15                     # 最後下降 35 mm


class Tripped(RuntimeError):
    """交接最後一段，監看發現手動了／不見了：手臂已被停住。"""


class ArmB:
    def __init__(self):
        self.arm = connect_arm("B")
        self.arm.wait_feed()
        self.cal = load("B")
        self.r = float(self.arm.pose()[3])

    def ik(self, p, z):
        return self.arm.dash.InverseSolution(p[0], p[1], z, self.r, 0, 0).startswith("0,")

    OFFSET_MAX = 28.0       # 偏移吸取上限（mm）：棋子半徑約 42、吸盤半徑約 10 → 吸盤整個還在棋子上（棋子貼了膠帶，頂面平）
    LAND_MAX = 10.0         # 還不行時，允許棋子落點離格中心這麼多（格寬約 95 mm）

    def _offset_ok(self, c, cell, off, stacks=("O疊",)):
        """吸盤相對棋子中心偏 off：料區吸取點（疊中心＋off）、放下點（格中心＋off）的 IK、支架、路線都要過。"""
        p = c + off
        zr = Z_RELEASE + board_dz(cell)
        park = np.array(PARK["B"][:2])
        if hits_post(self.cal, p, zr) or hits_post(self.cal, p, Z_HOVER):
            return False
        if not seg_clear(self.cal, travel_points(p, park), Z_HOVER):
            return False
        for name in stacks:
            sp = np.array(STOCK["stocks"][name]["xy"]) + off
            if hits_post(self.cal, sp, Z_HOVER) or not seg_clear(self.cal, travel_points(sp, p), Z_HOVER):
                return False
        if not (self.ik(p, zr) and self.ik(p, Z_HOVER)):
            return False
        for name in stacks:
            sp = np.array(STOCK["stocks"][name]["xy"]) + off
            z_lo = STOCK["stocks"][name]["z_first"] - 1.5 - STACK_PRESS - 3.0     # 疊只剩 1 顆、多壓重試
            if not (self.ik(sp, z_lo) and self.ik(sp, z_lo + 6 * PIECE) and self.ik(sp, Z_HOVER)):
                return False
        return True

    def target(self, plan, cell, stacks=("O疊",)):
        """回傳 (吸盤放下點, 偏移)：先試正中央；不行就「偏移吸取」——在料區偏 off 吸，放下時吸盤到格中心＋off，
        棋子仍落在格中心，但手臂可以離支架遠一點或少伸一點（使用者：搆不到要用偏移夾取解決）。都不行回 None。"""
        c0 = np.array(plan[str(cell)]["B"])
        k = int(self.OFFSET_MAX // 3)
        offs = sorted({(3.0 * i, 3.0 * j) for i in range(-k, k + 1) for j in range(-k, k + 1)
                       if np.hypot(3 * i, 3 * j) <= self.OFFSET_MAX}, key=lambda o: np.hypot(*o))
        u = c0 / np.hypot(*c0)
        lands = [c0] + [c0 + s * d * u for d in np.arange(2, self.LAND_MAX + 0.1, 2) for s in (1, -1)]
        for c in lands:                                     # 先要求棋子落在格中心，不行才讓它偏一點
            for off in offs:
                off = np.array(off)
                if self._offset_ok(c, cell, off, stacks):
                    return c + off, off
        return None

    def x_target(self, plan, cell, bar_deg):
        """X 棋子沒貼膠帶：凸起的綠叉讓大部分表面不平，只有「正中央」或「叉與叉之間黑區的中段」吸得住。
        bar_deg＝料區最上面那顆 X 的叉在影像上的角度（度）。回傳 (吸盤放下點, 偏移) 或 None。"""
        c = np.array(plan[str(cell)]["B"])
        cands = [np.zeros(2)]
        for k in range(4):
            a = np.radians(bar_deg + 45 + 90 * k)            # 兩條叉之間的角平分線（影像方向）
            v = np.array([np.sin(a), np.cos(a)])             # 影像 (x,y) → B 座標：B +y＝影像 +x、B +x＝影像 +y
            cands += [v * d for d in (29.0, 27.0, 31.0)]   # 黑區中段：離叉邊、離棋子邊緣都約 3 mm 餘裕
        for off in cands:                                    # 正中央優先，其次黑區中段
            if self._offset_ok(c, cell, off, ("X疊",)):
                return c + off, off
        return None

    def reachable(self, plan):
        """回傳 {格: 吸盤放下點}；偏移另存 self.offsets（place 依格取用）。"""
        out, self.offsets, self.x_cells = {}, {}, set()
        for i in range(9):
            t = self.target(plan, i, stacks=("O疊", "X疊"))          # 兩疊都能用同一個偏移最好（也能幫人代放 X）
            if t is not None:
                self.x_cells.add(i)
            else:
                t = self.target(plan, i, stacks=("O疊",))
            if t is not None:
                out[i], self.offsets[i] = t
        return out

    def clear_piece(self, center, cell, stack, n_in_stack, stop_flag=lambda: False, extra_press=0.0):
        """收棋盤（不靠人手）：棋盤上中心在 center（B 座標）的棋子 → 偏移吸取（必要時）→ 疊到 stack 第 n_in_stack+1 顆。
        回傳 None＝做完（是否真的吸起由呼叫端看相機確認）；字串＝做不到的原因（沒有動）。"""
        a, r = self.arm, self.r
        c = np.asarray(center, float)
        dz = board_dz(cell) if cell is not None else 0.0
        z_pick = Z_TOUCH_BOARD + dz + 0.3 - 1.0 - extra_press       # 棋子貼膠帶約厚 0.3 mm；多壓 1 mm
        dst0 = np.array(STOCK["stocks"][stack]["xy"])
        th = STACK_Z[stack][2] if stack in STACK_Z else 4.0
        z_rel = stack_top_z(stack, n_in_stack) + th + 1.5 if n_in_stack > 0 else STOCK["stocks"][stack]["z_first"] + 1.5
        park = np.array(PARK["B"][:2])
        k = int(self.OFFSET_MAX // 3)
        offs = sorted({(3.0 * i, 3.0 * j) for i in range(-k, k + 1) for j in range(-k, k + 1)
                       if np.hypot(3 * i, 3 * j) <= self.OFFSET_MAX}, key=lambda o: np.hypot(*o))
        best = None
        for off in offs:
            off = np.array(off)
            s, d = c + off, dst0 + off
            if hits_post(self.cal, s, z_pick) or hits_post(self.cal, s, Z_HOVER) or hits_post(self.cal, d, Z_HOVER):
                continue
            if not (seg_clear(self.cal, travel_points(park, s), Z_HOVER) and seg_clear(self.cal, travel_points(s, d), Z_HOVER)
                    and seg_clear(self.cal, travel_points(d, park), Z_HOVER)):
                continue
            if self.ik(s, z_pick) and self.ik(s, Z_HOVER) and self.ik(d, z_rel) and self.ik(d, Z_HOVER):
                best = off
                break
        if best is None:
            return "格 %s 的棋子 %d mm 內找不到 B 吸得到、送得到料區的點" % (cell, self.OFFSET_MAX)
        s, d = c + best, dst0 + best

        def step(cmd):
            if stop_flag():
                raise RuntimeError("已停止")
            sync_move(a, cmd)

        enable(a, speed=15)
        p = a.pose()
        if p[2] < Z_HOVER - 1:
            step("MovL(%f,%f,%f,%f)" % (p[0], p[1], Z_HOVER, r))
        travel(a, s, r)
        step("MovL(%f,%f,%f,%f)" % (s[0], s[1], z_pick, r))
        a.dash.sendRecvMsg("DOExecute(1,1)")
        time.sleep(0.8)
        step("MovL(%f,%f,%f,%f)" % (s[0], s[1], Z_HOVER, r))
        travel(a, d, r)
        step("MovL(%f,%f,%f,%f)" % (d[0], d[1], z_rel, r))
        a.dash.sendRecvMsg("DOExecute(1,0)")
        time.sleep(0.6)
        step("MovL(%f,%f,%f,%f)" % (d[0], d[1], Z_HOVER, r))
        travel(a, park, r)
        a.dash.DisableRobot()
        return None

    def place(self, stack, n_in_stack, target, stop_flag=lambda: False, cell=None, extra_press=0.0, offset=None,
              piece_shift=None):
        """stack＝"O疊"／"X疊"；n_in_stack＝目前疊上顆數（頂端那顆就是第 n 顆）；cell 給了就依該格棋盤高度修正放開 Z。"""
        a, r = self.arm, self.r
        off = (np.asarray(offset, float) if offset is not None
               else getattr(self, "offsets", {}).get(cell, np.zeros(2)) if cell is not None else np.zeros(2))
        stk = np.array(STOCK["stocks"][stack]["xy"]) + off           # 偏移吸取：吸盤在疊中心＋off
        if piece_shift is not None:                                   # 最上面那顆沒疊正：吸取點跟著棋子實際中心
            stk = stk + np.asarray(piece_shift, float)
        z_top = stack_top_z(stack, n_in_stack)

        def step(cmd):
            if stop_flag():
                raise RuntimeError("已停止")
            sync_move(a, cmd)

        enable(a, speed=15)
        p = a.pose()
        if p[2] < Z_HOVER - 1:
            step("MovL(%f,%f,%f,%f)" % (p[0], p[1], Z_HOVER, r))
        travel(a, stk, r)
        step("MovL(%f,%f,%f,%f)" % (stk[0], stk[1], z_top - STACK_PRESS - extra_press, r))
        a.dash.sendRecvMsg("DOExecute(1,1)")
        time.sleep(0.8)
        step("MovL(%f,%f,%f,%f)" % (stk[0], stk[1], Z_HOVER, r))
        travel(a, target, r)
        step("MovL(%f,%f,%f,%f)" % (target[0], target[1], Z_RELEASE + (board_dz(cell) if cell is not None else 0.0), r))
        a.dash.sendRecvMsg("DOExecute(1,0)")
        time.sleep(0.6)
        step("MovL(%f,%f,%f,%f)" % (target[0], target[1], Z_HOVER, r))
        travel(a, np.array(PARK["B"][:2]), r)
        a.dash.DisableRobot()

    # ---------- 花式交接：把棋子放到使用者攤開的手掌上 ----------
    def handover_plan(self, palm):
        """手掌（hand.detect 的結果）→ 放開點與高度；不行時回 (None, 原因)。手臂不碰手：棋子底部停在掌心上方 DROP_MM 放開。"""
        p = np.asarray(palm["xy"], float)
        z_rel = Z_TOUCH_BOARD + palm["h"] + DROP_MM          # 吸盤在棋盤棋子頂面＝Z_TOUCH_BOARD；再往上 手掌高＋落差
        z_trv = max(Z_HOVER, z_rel + TRAVEL_ABOVE)          # 移動高度：比手掌上方再高，前臂也掃不到
        r = float(np.hypot(*p))
        if r < HAND_R_MIN:          # 要能先停在手掌內側 STAGE_MM 的「旁邊點」（不擋住手）再靠近
            return None, "手離我太近了，請往你自己那邊移一點"
        if r > HAND_R_MAX:
            return None, "手離我太遠了，請往我這邊靠一點"
        if z_trv > Z_TRAVEL_MAX:
            return None, "手太高了，請放低一點"
        for z in (z_rel, z_trv):
            if not self.ik(p, z):
                return None, "那個位置我的手臂到不了，請換個地方"
            if hits_post(self.cal, p, z):
                return None, "手太靠近相機支架了，請往棋盤中間移一點"
        return dict(xy=p, z_rel=z_rel, z_trv=z_trv), None

    def _travel_at(self, p1, z, step):
        """在高度 z 從目前位置移到 p1：直線縮回 R_IN → 小半徑轉向 → 直線伸出（跟 pickplace.travel 同一條路線）。"""
        a, r = self.arm, self.r
        p0 = a.pose()[:2]
        a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
        if np.hypot(*p0) > R_IN + 2:
            step("MovL(%f,%f,%f,%f)" % (R_IN * np.cos(a0), R_IN * np.sin(a0), z, r))
        step("MovJ(%f,%f,%f,%f)" % (R_IN * np.cos(a1), R_IN * np.sin(a1), z, r))
        step("MovL(%f,%f,%f,%f)" % (p1[0], p1[1], z, r))

    def handover(self, stack, n_in_stack, get_palm, stop_flag=lambda: False, say=lambda t: None, guard=None):
        """從 stack 吸一顆 → 再確認一次手掌（手臂不在手上方，相機看得到）→ 高處移到掌心上方 → 慢慢降 → 放開 → 抬起回停放。
        get_palm(timeout) → 穩定的手掌或 None。回傳 "done"／"no_hand"／"bad:<原因>"（後兩者會把棋子放回疊上）。"""
        a, r = self.arm, self.r
        stk = np.array(STOCK["stocks"][stack]["xy"])
        z_top = stack_top_z(stack, n_in_stack)

        def step(cmd):
            if stop_flag():
                raise RuntimeError("已停止")
            if guard is not None and guard.tripped():
                raise Tripped(guard.tripped())
            sync_move(a, cmd)
            if guard is not None and guard.tripped():      # 監看在動作中按了停（ResetRobot），sync_move 會正常回來
                raise Tripped(guard.tripped())

        enable(a, speed=15)
        try:
            a.dash.sendRecvMsg("SetCollisionLevel(3)")      # 有支援就開碰撞偵測（不支援會回錯誤碼，不影響）
        except Exception:
            pass
        p = a.pose()
        if p[2] < Z_HOVER - 1:
            step("MovL(%f,%f,%f,%f)" % (p[0], p[1], Z_HOVER, r))
        travel(a, stk, r)
        step("MovL(%f,%f,%f,%f)" % (stk[0], stk[1], z_top - STACK_PRESS, r))
        a.dash.sendRecvMsg("DOExecute(1,1)")
        time.sleep(0.8)
        step("MovL(%f,%f,%f,%f)" % (stk[0], stk[1], Z_HOVER, r))

        def put_back(result):
            step("MovL(%f,%f,%f,%f)" % (stk[0], stk[1], z_top + 3.0, r))
            a.dash.sendRecvMsg("DOExecute(1,0)")
            time.sleep(0.6)
            step("MovL(%f,%f,%f,%f)" % (stk[0], stk[1], Z_HOVER, r))
            travel(a, np.array(PARK["B"][:2]), r)
            a.dash.DisableRobot()
            return result

        palm = get_palm(15.0)
        if palm is None:
            return put_back("no_hand")
        plan, why = self.handover_plan(palm)
        if plan is None:
            return put_back("bad:" + why)
        z_cur = plan["z_trv"]
        stage = self._stage(plan["xy"])
        if stage is None:
            return put_back("bad:手離我太近了，請往你自己那邊移一點")
        if not seg_clear(self.cal, travel_points(stk, stage), z_cur):
            return put_back("bad:路線會碰到相機支架，請換個位置")
        step("MovL(%f,%f,%f,%f)" % (stk[0], stk[1], z_cur, r))
        self._travel_at(stage, z_cur, step)                # 先停在手掌內側 STAGE_MM：手臂不擋住手，相機看得到

        def retreat(result):
            self._travel_at(stk, z_cur, step)
            step("MovL(%f,%f,%f,%f)" % (stk[0], stk[1], Z_HOVER, r))
            return put_back(result)

        released = False
        for attempt in range(3):                            # 手動了就退回來重新瞄準，最多追 3 次
            palm = get_palm(8.0)                            # 最新的手掌位置（高頻追蹤，連兩次同位置）
            if palm is None:
                return retreat("no_hand")
            plan, why = self.handover_plan(palm)
            if plan is None:
                return retreat("bad:" + why)
            st2 = self._stage(plan["xy"])
            if st2 is None:
                return retreat("bad:手的位置我搆不到")
            if plan["z_trv"] > z_cur + 1:                   # 手抬高了：先在原地升高
                z_cur = plan["z_trv"]
                step("MovL(%f,%f,%f,%f)" % (a.pose()[0], a.pose()[1], z_cur, r))
            if np.hypot(*(st2 - a.pose()[:2])) > 15:        # 換到新手掌對應的旁邊點
                if not seg_clear(self.cal, [a.pose()[:2] + t * (st2 - a.pose()[:2]) for t in np.linspace(0, 1, 15)], z_cur):
                    return retreat("bad:路線會碰到相機支架，請換個位置")
                step("MovL(%f,%f,%f,%f)" % (st2[0], st2[1], z_cur, r))
                stage = st2
            say("手不要動喔。")
            if guard is not None:
                err = guard.start(stage, plan["xy"], z_cur, on_trip=lambda: a.dash.ResetRobot())
                if err:
                    return retreat("bad:" + err)
            tripped = None
            try:
                step("MovL(%f,%f,%f,%f)" % (plan["xy"][0], plan["xy"][1], z_cur, r))
                a.dash.SpeedFactor(5)                      # 靠近手：慢速
                step("MovL(%f,%f,%f,%f)" % (plan["xy"][0], plan["xy"][1], plan["z_rel"], r))
                a.dash.sendRecvMsg("DOExecute(1,0)")
                released = True
                time.sleep(0.6)
            except Tripped as e:
                tripped = str(e)
            finally:
                if guard is not None:
                    guard.stop()
                if a.mode() != 5:
                    enable(a, speed=15)
                a.dash.SpeedFactor(15)
            p = a.pose()
            step("MovL(%f,%f,%f,%f)" % (p[0], p[1], z_cur, r))          # 先往上離開手
            if released:
                break
            say("%s，我重新對準。" % (tripped or "你的手動了"))
            back = self._stage(p[:2])                                     # 退回目前位置的內側，讓相機重新看到手
            if back is not None:
                step("MovL(%f,%f,%f,%f)" % (back[0], back[1], z_cur, r))
                stage = back
        if not released:
            return retreat("bad:你的手一直在動，這次先不給了")
        park = np.array(PARK["B"][:2])
        self._travel_at(park, z_cur, step)
        step("MovL(%f,%f,%f,%f)" % (park[0], park[1], Z_HOVER, r))
        a.dash.DisableRobot()
        return "done"

    def _stage(self, xy):
        """手掌點沿徑向往內 STAGE_MM 的「旁邊點」：手臂停在這裡不會擋住手掌。"""
        xy = np.asarray(xy, float)
        rr = float(np.hypot(*xy))
        rs = max(rr - STAGE_MM, R_IN + 5)
        if rr - rs < 60:                                   # 手太靠近我，旁邊點跟手重疊
            return None
        return xy / rr * rs

    # ---------- 快速交接 ----------
    def _fast(self, cmd, target, tol=1.0, timeout=8.0, guard=None, stop_flag=lambda: False):
        """送出運動指令，一到目標附近（tol mm）就回來（不等 0.3 s 靜止）；途中看監看與停止。"""
        import dual_sync
        a = self.arm
        dual_sync.check(a.move.sendRecvMsg(cmd), cmd)
        target = np.asarray(target, float)
        t0 = time.time()
        while True:
            if guard is not None and guard.tripped():
                raise Tripped(guard.tripped())
            if stop_flag():
                a.dash.ResetRobot()
                raise RuntimeError("已停止")
            if a.mode() == 9:
                raise RuntimeError("B 報警 %s（%s）" % (a.errors(), cmd))
            if np.abs(a.pose()[:3] - target).max() < tol:
                return
            if time.time() - t0 > timeout:
                raise RuntimeError("動作逾時：%s" % cmd)
            time.sleep(0.008)

    def hand_prepare(self, stack, n_in_stack, stop_flag=lambda: False):
        """吸一顆 → 帶著棋子到待命點等（之後 hand_deliver／hand_put_back）。"""
        a, r = self.arm, self.r
        stk = np.array(STOCK["stocks"][stack]["xy"])
        z_top = stack_top_z(stack, n_in_stack)
        if not seg_clear(self.cal, travel_points(stk, READY_XY), Z_READY):
            raise RuntimeError("料區到待命點的路線會碰支架")

        def step(cmd):
            if stop_flag():
                raise RuntimeError("已停止")
            sync_move(a, cmd)

        enable(a, speed=15)
        try:
            a.dash.sendRecvMsg("SetCollisionLevel(3)")
        except Exception:
            pass
        p = a.pose()
        if p[2] < Z_HOVER - 1:
            step("MovL(%f,%f,%f,%f)" % (p[0], p[1], Z_HOVER, r))
        travel(a, stk, r)
        step("MovL(%f,%f,%f,%f)" % (stk[0], stk[1], z_top - STACK_PRESS, r))
        a.dash.sendRecvMsg("DOExecute(1,1)")
        time.sleep(0.8)
        step("MovL(%f,%f,%f,%f)" % (stk[0], stk[1], Z_READY, r))
        self._travel_at(READY_XY, Z_READY, step)
        self.hand_ctx = dict(stack=stack, stk=stk, z_top=z_top, z_cur=Z_READY)

    def hand_deliver(self, palm, guard=None, stop_flag=lambda: False):
        """從待命點直接到掌心上方 → 下降 → 放開 → 抬起。回傳 dict(ok, why, t_release＝出發到放開秒數)。"""
        a, r = self.arm, self.r
        plan, why = self.handover_plan(palm)
        if plan is None:
            return dict(ok=False, why=why)
        x, y = plan["xy"]
        p = a.pose()
        z = max(self.hand_ctx["z_cur"], plan["z_trv"])
        if not seg_clear(self.cal, [p[:2] + t * (plan["xy"] - p[:2]) for t in np.linspace(0, 1, 15)], z):
            return dict(ok=False, why="路線會碰到相機支架，請換個位置")
        if z > p[2] + 1:
            self._fast("MovL(%f,%f,%f,%f)" % (p[0], p[1], z, r), (p[0], p[1], z), stop_flag=stop_flag)
            self.hand_ctx["z_cur"] = z
        if guard is not None:
            err = guard.start(p[:2], plan["xy"], z, on_trip=lambda: a.dash.ResetRobot())
            if err:
                return dict(ok=False, why=err)
        t0, released, why, t_rel = time.time(), False, None, None
        try:
            a.dash.SpeedFactor(FAST_SPEED)
            self._fast("MovL(%f,%f,%f,%f)" % (x, y, z, r), (x, y, z), tol=2.0, guard=guard, stop_flag=stop_flag)
            a.dash.SpeedFactor(DESCENT_SPEED)
            self._fast("MovL(%f,%f,%f,%f)" % (x, y, plan["z_rel"], r), (x, y, plan["z_rel"]), tol=1.0,
                       guard=guard, stop_flag=stop_flag)
            a.dash.sendRecvMsg("DOExecute(1,0)")
            released, t_rel = True, time.time() - t0
            time.sleep(0.25)                                   # 讓棋子離開吸盤再抬
        except Tripped as e:
            why = str(e)
        finally:
            if guard is not None:
                guard.stop()
            if a.mode() != 5:
                enable(a, speed=15)
            a.dash.SpeedFactor(15)
        q = a.pose()
        sync_move(a, "MovL(%f,%f,%f,%f)" % (q[0], q[1], z, r))   # 先往上離開手
        return dict(ok=released, why=why, t_release=t_rel)

    def hand_back_ready(self, stop_flag=lambda: False):
        a, r = self.arm, self.r
        z = self.hand_ctx["z_cur"]
        if stop_flag():
            raise RuntimeError("已停止")
        sync_move(a, "MovL(%f,%f,%f,%f)" % (READY_XY[0], READY_XY[1], z, r))

    def hand_finish(self):
        """交完了：回停放、下使能。"""
        a, r = self.arm, self.r
        park = np.array(PARK["B"][:2])
        self._travel_at(park, self.hand_ctx["z_cur"], lambda c: sync_move(a, c))
        sync_move(a, "MovL(%f,%f,%f,%f)" % (park[0], park[1], Z_HOVER, r))
        a.dash.DisableRobot()

    def hand_put_back(self):
        """沒交出去：把棋子放回疊上、回停放。"""
        a, r, c = self.arm, self.r, self.hand_ctx
        stk = c["stk"]
        self._travel_at(stk, c["z_cur"], lambda cmd: sync_move(a, cmd))
        sync_move(a, "MovL(%f,%f,%f,%f)" % (stk[0], stk[1], c["z_top"] + 3.0, r))
        a.dash.sendRecvMsg("DOExecute(1,0)")
        time.sleep(0.6)
        sync_move(a, "MovL(%f,%f,%f,%f)" % (stk[0], stk[1], Z_HOVER, r))
        travel(a, np.array(PARK["B"][:2]), r)
        a.dash.DisableRobot()

    def stop(self):
        try:
            self.arm.dash.sendRecvMsg("DOExecute(1,0)")
            self.arm.dash.ResetRobot()
            self.arm.dash.DisableRobot()
        except Exception:
            pass

    def close(self):
        self.stop()
        self.arm.close()
