"""雙臂同步運動測試，並量測兩台的同步誤差（毫秒）。

和「兩個行程各跑一次 safe_test.py」的差別：
  - 同一個行程、每台一條執行緒，用 Barrier 讓兩台的指令在同一瞬間送出
  - 用 Sync() 阻塞等待到位，不輪詢 Dashboard（原本每 0.1 s 輪詢一次）
  - 兩台各連 30004 即時回饋埠（約 125 Hz），用回饋封包的實際位置算「起動時間差」與「到位時間差」
  - 連線失敗自動重試（實測偶發 29999 逾時）

用法（在本目錄下）：
    python dual_sync.py                          # 只讀兩台狀態並印計畫，不會動
    python dual_sync.py --go                     # 實際運動（兩台周圍淨空、急停在手邊）
    python dual_sync.py --go --speed=50 --cycles=5 --speedl=50 --accl=50   # 兩台統一的內部速度／加速度（%）
任一台出錯或 Ctrl+C，兩台都會 ResetRobot() 停止，結束一律 DisableRobot()。
"""
import collections
import re
import sys
import threading
import time

import numpy as np

from dobot_api import DobotApi, DobotApiDashboard, DobotApiMove, MyType

DobotApi.log = lambda self, text: None      # 關掉 SDK 每筆收送都 print（會拖慢、干擾時序）

ARMS = {"A": "192.168.1.6", "B": "192.168.2.7"}
try:                                     # 專案根目錄的 site.json 可覆蓋 IP（跟 Calib/common.py 同一份）
    import json as _json
    import os as _os
    ARMS.update(_json.load(open(_os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "site.json"),
                                encoding="utf-8")).get("arm_ips", {}))
except (OSError, ValueError):
    pass
SAFE_X, SAFE_Z, SWING_Y = 270.0, 100.0, 40.0
ONSET_TOL = 0.05        # mm：位置偏離起點超過此值視為「開始動」
ARRIVE_TOL = 0.05       # mm：離目標小於此值視為「到位」
SOCKET_TIMEOUT = 60.0   # 秒：Sync() 最長等待


def opt(name, default):
    for a in sys.argv[1:]:
        if a.startswith("--%s=" % name):
            return type(default)(a.split("=", 1)[1])
    return default


def code(reply):
    m = re.match(r"\s*(-?\d+)", reply or "")
    return int(m.group(1)) if m else None


def values(reply):
    m = re.search(r"\{([^}]*)\}", reply or "")
    return [float(v) for v in m.group(1).split(",") if v.strip()] if m else []


def check(reply, what):
    if code(reply) != 0:
        raise RuntimeError("%s 失敗，回傳 %r" % (what, (reply or "").strip()))


class _RawFeed:
    """跟 DobotApi 一樣有 socket_dobot／close()，但不限制埠號。"""

    def __init__(self, ip, port):
        import socket
        self.socket_dobot = socket.create_connection((ip, port), timeout=5)

    def close(self):
        try:
            self.socket_dobot.close()
        except OSError:
            pass


class Arm:
    def __init__(self, name, ip):
        self.name, self.ip = name, ip
        for attempt in range(3):
            try:
                self.dash = DobotApiDashboard(ip, 29999)
                self.move = DobotApiMove(ip, 30003)
                self.feed = DobotApi(ip, 30004)
                self.feed_slow = False
                # 30004（8 ms）偶爾會卡死：連得上但完全不送資料（2026-11-15，USB 網卡被重插之後兩台都這樣，要重開手臂電源才恢復）。
                # 2 秒內沒資料就改用 30005（同樣的封包，200 ms 一筆）：到位判斷與接觸偵測變粗，但還能動。
                try:
                    self.feed.socket_dobot.settimeout(2.0)
                    first = self.feed.socket_dobot.recv(1440)
                except OSError:
                    first = b""
                if not first:
                    try:
                        self.feed.close()
                    except Exception:
                        pass
                    self.feed = _RawFeed(ip, 30005)          # 官方 DobotApi 只認 30004，這裡直接開 socket
                    self.feed_slow = True
                    first = b""
                    print("%s：30004 沒有資料，改用 30005（200 ms）回饋；重開手臂電源可恢復" % name, flush=True)
                self.feed.socket_dobot.settimeout(None)
                self._first = first
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.0)
        self.move.socket_dobot.settimeout(SOCKET_TIMEOUT)
        self.movl_params = ""                   # 附在每個 MovL 後的參數，例如 "AccL=35"
        self.samples = collections.deque(maxlen=20000)   # (perf_counter, [x,y,z,r], robot_mode)；只留最近約 160 s
        self.stop = False
        self.thread = threading.Thread(target=self._feed_loop, daemon=True)
        self.thread.start()

    def _feed_loop(self):
        buf = getattr(self, "_first", b"")
        if len(buf) >= 1440:
            buf = b""
        while not self.stop:
            try:
                chunk = self.feed.socket_dobot.recv(1440 - len(buf))
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            if len(buf) < 1440:
                continue
            t = time.perf_counter()
            rec = np.frombuffer(buf, dtype=MyType)[0]
            buf = b""
            if int(rec["test_value"]) != 0x123456789abcdef:
                continue
            self.samples.append((t, np.array(rec["tool_vector_actual"][:4]), int(rec["robot_mode"])))
            self.last_rec = rec.copy()                 # 完整回饋（電流、目標位置…）：接觸偵測用

    def wait_feed(self):
        for _ in range(50):
            if self.samples:
                return
            time.sleep(0.1)
        raise RuntimeError("%s：收不到 30004 即時回饋" % self.name)

    def pose(self):
        return self.samples[-1][1]

    def mode(self):
        return self.samples[-1][2]

    def errors(self):
        return [int(n) for n in re.findall(r"-?\d+", self.dash.GetErrorID())[1:]]

    def close(self):
        self.stop = True
        for c in (self.dash, self.move, self.feed):
            c.close()


def fmt(cmd, p, extra=""):
    return "%s(%f,%f,%f,%f%s)" % (cmd, *p, ("," + extra) if extra else "")


def run_leg(arms, targets, cmd):
    """兩台同時送出 cmd 並等到位；回傳每台的 (送出時間, 起點) 供事後分析。"""
    barrier = threading.Barrier(len(arms), timeout=10)
    out, errs = {}, []

    def worker(a):
        try:
            start = a.pose().copy()
            barrier.wait()
            t_send = time.perf_counter()
            extra = a.movl_params if cmd == "MovL" else ""
            check(a.move.sendRecvMsg(fmt(cmd, targets[a.name], extra)), a.name + " " + cmd)
            check(a.move.sendRecvMsg("Sync()"), a.name + " Sync")
            out[a.name] = (t_send, start)
            if a.mode() == 9:
                raise RuntimeError("%s 運動中報警 %s" % (a.name, a.errors()))
        except Exception as e:
            errs.append(e)
            barrier.abort()

    ts = [threading.Thread(target=worker, args=(a,)) for a in arms]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    if errs:
        raise errs[0]
    time.sleep(0.1)                              # 讓最後幾筆回饋進來
    return out


def analyze(a, t_send, start, target):
    for _ in range(5):                                   # 先快照：回饋執行緒仍在 append，偶爾會撞到「迭代中被改動」
        try:
            snap = list(a.samples)
            break
        except RuntimeError:
            time.sleep(0.001)
    s = [x for x in snap if x[0] >= t_send - 0.02]
    onset = next((t for t, p, m in s if np.abs(p[:3] - start[:3]).max() > ONSET_TOL), None)
    arrive = None
    if onset is not None:
        arrive = next((t for t, p, m in s if t >= onset and np.abs(p[:3] - target[:3]).max() <= ARRIVE_TOL), None)
    return onset, arrive


def report(label, arms, targets, res):
    a, b = arms
    (ta, sa), (tb, sb) = res[a.name], res[b.name]
    oa, ra = analyze(a, ta, sa, targets[a.name])
    ob, rb = analyze(b, tb, sb, targets[b.name])
    if None in (oa, ra, ob, rb):
        print("%-10s 回饋資料不足，無法量測" % label)
        return None
    row = ((tb - ta) * 1e3, (ob - oa) * 1e3, (rb - ra) * 1e3, (ra - oa), (rb - ob))
    print("%-10s 送出差 %+6.1f ms | 起動差 %+6.1f ms | 到位差 %+6.1f ms | 行程 A %.3fs B %.3fs"
          % ((label,) + row))
    return row


def main(go):
    speed, cycles = opt("speed", 20), opt("cycles", 3)
    arms = [Arm(n, ip) for n, ip in ARMS.items()]
    try:
        for a in arms:
            a.wait_feed()
            print("%s %s：模式 %d 報警 %s 姿態 %s" % (a.name, a.ip, a.mode(), a.errors(), np.round(a.pose(), 1)))
        for a in arms:
            a.movl_params = opt("movl-" + a.name.lower(), "")   # 例：--movl-b=AccL=35
        safe = {a.name: np.array([SAFE_X, 0.0, SAFE_Z, a.pose()[3]]) for a in arms}
        left = {a.name: np.array([SAFE_X, -SWING_Y, SAFE_Z, a.pose()[3]]) for a in arms}
        right = {a.name: np.array([SAFE_X, SWING_Y, SAFE_Z, a.pose()[3]]) for a in arms}
        print("計畫：速度 %d%%，兩台同時 MovJ 安全點，再同時 MovL Y=±%.0f mm 來回 %d 次，回安全點並下使能"
              % (speed, SWING_Y, cycles))
        if not go:
            print("（未加 --go，只讀狀態，不會動）")
            return

        for a in arms:
            if a.mode() == 9:
                check(a.dash.ClearError(), a.name + " ClearError")
                time.sleep(0.5)
            check(a.dash.SpeedFactor(speed), a.name + " SpeedFactor")
            # 兩台控制器內部預設的 SpeedL/AccL 不同，會讓同一段軌跡時間差約 0.1 s，這裡統一
            check(a.dash.SpeedL(opt("speedl", 50)), a.name + " SpeedL")
            check(a.dash.AccL(opt("accl", 50)), a.name + " AccL")
            check(a.dash.SpeedJ(opt("speedj", 50)), a.name + " SpeedJ")
            check(a.dash.AccJ(opt("accj", 50)), a.name + " AccJ")
            check(a.dash.EnableRobot(), a.name + " EnableRobot")
        time.sleep(1.0)
        for a in arms:
            if a.mode() != 5:
                raise RuntimeError("%s 使能後模式為 %s，不是 5" % (a.name, a.mode()))

        rows = []
        res = run_leg(arms, safe, "MovJ")
        report("安全點", arms, safe, res)
        for i in range(cycles):
            for tgt, nm in ((left, "左"), (right, "右")):
                res = run_leg(arms, tgt, "MovL")
                rows.append(report("#%d %s" % (i + 1, nm), arms, tgt, res))
        res = run_leg(arms, safe, "MovL")
        rows.append(report("回安全點", arms, safe, res))

        rows = [r for r in rows if r]
        if rows:
            m = np.abs(np.array(rows))
            print("\n統計（%d 段 MovL，絕對值）：" % len(rows))
            for i, nm in enumerate(("送出差", "起動差", "到位差")):
                print("  %s：平均 %.1f ms，最大 %.1f ms" % (nm, m[:, i].mean(), m[:, i].max()))
        print("\n完成 ✓")
    except KeyboardInterrupt:
        print("\n中斷：停止兩台")
        for a in arms:
            a.dash.ResetRobot()
    except Exception as e:
        print("\n停止：%s" % e)
        for a in arms:
            try:
                a.dash.ResetRobot()
            except Exception:
                pass
    finally:
        if go:
            for a in arms:
                try:
                    a.dash.DisableRobot()
                except Exception:
                    pass
        for a in arms:
            a.close()


if __name__ == "__main__":
    main("--go" in sys.argv)
