"""乾跑用的模擬手臂：介面與 robot.ArmB 相同，不連線、不動，只照真實步驟花時間並檢查停止旗標。"""
import time

STEPS = [("升到懸停高度", 0.4), ("移到料區上方", 1.2), ("下壓吸取", 0.6), ("開真空", 0.8), ("抬起", 0.5),
         ("移到目標格上方", 1.2), ("下降到放開高度", 0.6), ("關真空", 0.6), ("抬起", 0.5), ("回停放點", 1.2)]


class SimArm:
    def __init__(self):
        self.log = []

    def clear_piece(self, center, cell, stack, n_in_stack, stop_flag=lambda: False, extra_press=0.0):
        for name, dt in STEPS:
            t_end = time.time() + dt
            while time.time() < t_end:
                if stop_flag():
                    raise RuntimeError("已停止")
                time.sleep(0.05)
        self.log.append("收 格%s → %s" % (cell, stack))
        return None

    def place(self, stack, n_in_stack, target, stop_flag=lambda: False, cell=None, extra_press=0.0, offset=None,
              piece_shift=None):
        if n_in_stack < 1:
            raise RuntimeError("%s 沒有棋子" % stack)
        for name, dt in STEPS:
            t_end = time.time() + dt
            while time.time() < t_end:
                if stop_flag():
                    self.log.append("中斷於：" + name)
                    raise RuntimeError("已停止")
                time.sleep(0.05)
        self.log.append("%s → %s 完成" % (stack, target))

    def handover(self, stack, n_in_stack, get_palm, stop_flag=lambda: False, say=lambda t: None, guard=None):
        def wait(dt):
            t_end = time.time() + dt
            while time.time() < t_end:
                if stop_flag():
                    raise RuntimeError("已停止")
                time.sleep(0.05)
        wait(3.5)                                    # 去料區吸一顆、抬起
        palm = get_palm(15.0)
        if palm is None:
            wait(2.5)
            return "no_hand"
        say("手不要動喔。")
        wait(4.0)                                    # 高處移過去、慢慢降、放開、回停放
        self.log.append("%s → 手掌 %s" % (stack, palm.get("xy")))
        return "done"

    def stop(self):
        self.log.append("stop")

    def close(self):
        pass


# ---------- 快速交接的模擬（--dry-arm：相機、手掌偵測、監看都是真的，只有手臂是模擬的） ----------
import numpy as np                                    # noqa: E402

# 手臂速度的估計值（尚未實測）：SpeedFactor 每 1% 約 5 mm/s，每段動作另加 0.25 s 加減速
MM_PER_S_PER_PCT = 5.0
ACCEL_S = 0.25


class SimHandArm(SimArm):
    """介面同 robot.ArmB 的交接部分；幾何（校正、支架、可達範圍）用真的，動作只照估計時間等。"""

    def __init__(self):
        super().__init__()
        import robot
        from world import load
        self.R = robot
        self.cal = load("B")
        self.r = 10.0
        self.p = np.array([0.0, -220.0, -40.0, 10.0])
        self.arm = self                                # server 用 self.arm.arm.pose() 算手臂在畫面上的位置
        self.hand_ctx = None
        self._reset = False
        self.dash = self

    # server／監看會呼叫的
    def pose(self):
        return self.p.copy()

    def ResetRobot(self):
        self._reset = True

    def ik(self, p, z):
        return 220.0 <= float(np.hypot(*p)) <= 420.0      # 沒有控制器可問：用實測半徑範圍代替

    def handover_plan(self, palm):
        return self.R.ArmB.handover_plan(self, palm)

    def _move(self, target, speed, guard=None, stop_flag=lambda: False):
        target = np.asarray(target, float)
        dist = float(np.linalg.norm(target - self.p[:3]))
        dt = ACCEL_S + dist / (MM_PER_S_PER_PCT * speed)
        t0, start = time.time(), self.p[:3].copy()
        while True:
            k = min(1.0, (time.time() - t0) / dt)
            self.p[:3] = start + k * (target - start)
            if guard is not None and guard.tripped():
                raise self.R.Tripped(guard.tripped())
            if stop_flag():
                raise RuntimeError("已停止")
            if k >= 1.0:
                return
            time.sleep(0.01)

    def hand_prepare(self, stack, n_in_stack, stop_flag=lambda: False):
        R = self.R
        stk = np.array(R.STOCK["stocks"][stack]["xy"])
        for tgt in ((*stk, R.Z_HOVER), (*stk, -160.0), (*stk, R.Z_READY), (*R.READY_XY, R.Z_READY)):
            self._move(tgt, 15, stop_flag=stop_flag)
        self.hand_ctx = dict(stack=stack, z_cur=R.Z_READY)
        self.log.append("待命點 %s" % R.READY_XY)

    def hand_deliver(self, palm, guard=None, stop_flag=lambda: False):
        R = self.R
        plan, why = self.handover_plan(palm)
        if plan is None:
            return dict(ok=False, why=why)
        x, y = plan["xy"]
        z = max(self.hand_ctx["z_cur"], plan["z_trv"])
        if z > self.p[2] + 1:
            self._move((self.p[0], self.p[1], z), 15)
            self.hand_ctx["z_cur"] = z
        if guard is not None:
            err = guard.start(self.p[:2].copy(), plan["xy"], z, on_trip=self.ResetRobot)
            if err:
                return dict(ok=False, why=err)
        t0, ok, why, t_rel = time.time(), False, None, None
        try:
            self._move((x, y, z), R.FAST_SPEED, guard, stop_flag)
            self._move((x, y, plan["z_rel"]), R.DESCENT_SPEED, guard, stop_flag)
            ok, t_rel = True, time.time() - t0
            time.sleep(0.25)
        except R.Tripped as e:
            why = str(e)
        finally:
            if guard is not None:
                guard.stop()
        self._move((self.p[0], self.p[1], z), 15)
        self.log.append("交接 %s 秒" % (None if t_rel is None else round(t_rel, 2)))
        return dict(ok=ok, why=why, t_release=t_rel)

    def hand_back_ready(self, stop_flag=lambda: False):
        self._move((*self.R.READY_XY, self.hand_ctx["z_cur"]), 15, stop_flag=stop_flag)

    def hand_finish(self):
        self._move((0.0, -220.0, -40.0), 15)

    def hand_put_back(self):
        self._move((0.0, -220.0, -40.0), 15)
