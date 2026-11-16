"""MG400 雙臂控制台（本機網頁）：監控 + 控制兩台手臂的所有介面。

    python server.py                # 開 http://127.0.0.1:8800
    python server.py --no-camera    # 不開相機
    python server.py --watch        # 只看不控：對弈伺服器（Arena/server.py）在跑時用。只讀 30004 回饋，不送任何指令
                                    # （29999 會跟對弈伺服器搶）、不開相機（畫面用 Arena/cam_full.jpg）
只綁 127.0.0.1（只有這台電腦能開）。手臂通訊沿用 MG400/dobot_api.py。

安全設計：
- 點動（MoveJog）採「按住才動」：網頁每 100 ms 送心跳，超過 JOG_TIMEOUT 沒收到心跳，後端自動送停止。
- 座標移動前先用控制器 InverseSolution 檢查可達，並檢查 Z 下限與相機支架禁入區（影像 x > KEEPOUT_X_PX）。
- DO 用 DOExecute（立即生效；DO() 會排進運動佇列，未使能時不執行）。
"""
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "MG400"))
sys.path.insert(0, os.path.join(ROOT, "Calib"))
from dobot_api import DobotApi, DobotApiDashboard, DobotApiMove, MyType  # noqa: E402

DobotApi.log = lambda self, text: None      # SDK 每筆收送都 print，關掉

from dual_sync import ARMS                  # noqa: E402  手臂 IP（專案根目錄 site.json 可覆蓋）

WATCH = "--watch" in sys.argv               # 只看不控
PORT = 8800
JOG_TIMEOUT = 0.4            # 秒：點動心跳逾時就停
Z_MIN = -110.0               # mm：座標移動的 Z 下限（B 曾在 Z=-114.5 於棋子上方未碰觸，留餘裕）
KEEPOUT_X_PX = 700.0         # 相機支架側禁入區（與 Calib/play.py 一致）
PARK = {"A": (60.0, 340.0, -40.0), "B": (0.0, -220.0, -40.0)}
MODE_NAMES = {1: "初始化", 2: "抱閘鬆開", 3: "本體下電", 4: "未使能", 5: "使能空閒", 6: "拖曳",
              7: "運行中", 8: "軌跡錄製", 9: "報警", 10: "暫停", 11: "點動"}
JOG_AXES = {"J1+", "J1-", "J2+", "J2-", "J3+", "J3-", "J4+", "J4-",
            "X+", "X-", "Y+", "Y-", "Z+", "Z-", "R+", "R-"}


def _alarm_tables():
    out = {}
    for fn, kind in (("alarm_controller.json", "控制器"), ("alarm_servo.json", "伺服")):
        try:
            for it in json.load(open(os.path.join(ROOT, "MG400", "files", fn), encoding="utf-8")):
                out.setdefault(it["id"], "%s：%s" % (kind, it["zh_CN"]["description"]))
        except Exception:
            pass
    return out


ALARMS = _alarm_tables()


def code(reply):
    m = re.match(r"\s*(-?\d+)", reply or "")
    return int(m.group(1)) if m else None


def _calib(arm):
    try:
        return json.load(open(os.path.join(ROOT, "Calib", "calib_%s.json" % arm), encoding="utf-8"))
    except Exception:
        return None


def in_keepout(arm, x, y):
    """TCP 投影到棋盤面的影像 x 座標是否在相機支架禁入區（需要該臂的手眼校正檔；沒有就不擋）。"""
    cal = _calib(arm)
    if not cal:
        return False
    from world import arm_to_px
    return float(arm_to_px(cal, (x, y))[0]) > KEEPOUT_X_PX


class Arm:
    """一台手臂：Dashboard（29999）+ Move（30003）+ 即時回饋（30004）背景執行緒，自動重連。"""

    def __init__(self, name, ip):
        self.name, self.ip = name, ip
        self.lock = threading.Lock()           # Dashboard／Move 指令序列化
        self.dash = self.move = None
        self.state = {"online": False}
        self.errors = []
        self.jog_axis = None
        self.jog_beat = 0.0
        self.log = []
        threading.Thread(target=self._feed_loop, daemon=True).start()
        threading.Thread(target=self._slow_loop, daemon=True).start()
        threading.Thread(target=self._jog_watchdog, daemon=True).start()

    # ---------- 連線 ----------
    def _ensure(self):
        if self.dash is None:
            self.dash = DobotApiDashboard(self.ip, 29999)
            self.dash.socket_dobot.settimeout(5)
        if self.move is None:
            self.move = DobotApiMove(self.ip, 30003)
            self.move.socket_dobot.settimeout(5)

    def _drop(self):
        for s in (self.dash, self.move):
            try:
                s.close()
            except Exception:
                pass
        self.dash = self.move = None

    def _call(self, which, cmd):
        """送指令並回傳回覆字串；斷線（例外或空回覆）就重連後重送一次。

        手臂重開機後，舊 socket 常常不會丟例外，而是一直回空字串（SDK 的 wait_reply 吞掉了錯誤），
        所以空回覆也要當成斷線處理，否則指令通道會一直卡在死連線上。"""
        with self.lock:
            for attempt in (1, 2):
                try:
                    self._ensure()
                    sock = self.dash if which == "dash" else self.move
                    r = sock.sendRecvMsg(cmd)
                    if not r:
                        raise ConnectionError("空回覆（連線已失效）")
                    return r
                except Exception:
                    self._drop()
                    if attempt == 2:
                        raise

    def dcall(self, cmd):
        return self._call("dash", cmd)

    def mcall(self, cmd):
        return self._call("move", cmd)

    # ---------- 監控 ----------
    def _feed_loop(self):
        while True:
            try:
                f = DobotApi(self.ip, 30004)
                f.socket_dobot.settimeout(3)
                buf = b""
                while True:
                    chunk = f.socket_dobot.recv(1440 - len(buf))
                    if not chunk:
                        raise ConnectionError("closed")
                    buf += chunk
                    if len(buf) < 1440:
                        continue
                    rec = np.frombuffer(buf, dtype=MyType)[0]
                    buf = b""
                    if int(rec["test_value"]) != 0x123456789abcdef:
                        continue
                    mode = int(rec["robot_mode"])
                    di, do = int(rec["digital_input_bits"]), int(rec["digital_outputs"])
                    self.state = {
                        "online": True, "t": time.time(), "mode": mode, "mode_name": MODE_NAMES.get(mode, str(mode)),
                        "pose": [round(float(v), 2) for v in rec["tool_vector_actual"][:4]],
                        "joints": [round(float(v), 2) for v in rec["q_actual"][:4]],
                        "di": [(di >> i) & 1 for i in range(16)], "do": [(do >> i) & 1 for i in range(16)],
                        "speed": round(float(rec["speed_scaling"]), 1),
                        "enabled": int(rec["EnableStatus"][0]), "error": int(rec["ErrorStatus"][0]),
                        "running": int(rec["RunningStatus"][0]),
                        "queue": int(rec["isRunQueuedCmd"][0]), "paused": int(rec["isPauseCmdFlag"][0]),
                    }
            except Exception as e:
                self.state = {"online": False, "why": str(e)[:120], "t": time.time()}
                self.jog_axis = None
                with self.lock:                      # 監控斷了：指令連線多半也失效，丟掉，恢復後用新連線
                    self._drop()
                time.sleep(2)

    def _slow_loop(self):
        """每秒讀一次報警碼（附中文說明）。"""
        while True:
            try:
                if self.state.get("online") and not WATCH:
                    r = self.dcall("GetErrorID()")
                    ids = [int(n) for n in re.findall(r"-?\d+", r)[1:]]
                    self.errors = [{"id": i, "desc": ALARMS.get(i, "未知報警")} for i in ids]
            except Exception:
                pass
            time.sleep(1.0)

    # ---------- 點動 ----------
    def _jog_watchdog(self):
        while True:
            if self.jog_axis and time.time() - self.jog_beat > JOG_TIMEOUT:
                self.jog_stop("心跳逾時")
            time.sleep(0.05)

    def jog_start(self, axis):
        if axis not in JOG_AXES:
            raise ValueError("未知的點動軸 %s" % axis)
        self.jog_beat = time.time()
        if self.jog_axis == axis:
            return "持續中"
        if self.jog_axis:
            self.jog_stop("換軸")
        coord = 0 if axis.startswith("J") else 1                 # 0 = 關節點動、1 = 使用者座標系點動
        # 這台韌體不接受 "MoveJog(Z-,CoordType=1,User=0,Tool=0)"（回 -40002 參數錯誤，2026-11-14），
        # 依序試幾種寫法，第一個被接受的就記住沿用
        variants = [self.jog_fmt] if getattr(self, "jog_fmt", None) else             ["MoveJog({a})", "MoveJog({a},CoordType={c})", "MoveJog({a},CoordType={c},User=0,Tool=0)"]
        last = ""
        for fmt in variants:
            last = self.mcall(fmt.format(a=axis, c=coord))
            if code(last) == 0:
                self.jog_fmt = fmt
                self.jog_axis = axis
                return last
        raise RuntimeError("MoveJog 失敗：%s" % last.strip())

    def jog_stop(self, why="放開"):
        self.jog_axis = None
        try:
            return self.mcall("MoveJog()")
        except Exception as e:
            return "停止失敗：%s" % e

    def snapshot(self):
        s = dict(self.state)
        s.update(name=self.name, ip=self.ip, errors=self.errors, jog=self.jog_axis,
                 age=round(time.time() - s.get("t", 0), 2), log=self.log[-12:])
        return s

    def note(self, text):
        self.log.append("%s %s" % (time.strftime("%H:%M:%S"), text))
        self.log = self.log[-50:]


ARM_OBJ = {n: Arm(n, ip) for n, ip in ARMS.items()}


# ---------- 指令 ----------
def check_target(arm, x, y, z, r):
    if z < Z_MIN:
        raise ValueError("Z=%.1f 低於下限 %.0f" % (z, Z_MIN))
    if in_keepout(arm.name, x, y):
        raise ValueError("目標在相機支架禁入區（影像 x > %.0f px）" % KEEPOUT_X_PX)
    rep = arm.dcall("InverseSolution(%f,%f,%f,%f,0,0)" % (x, y, z, r))
    if code(rep) != 0:
        raise ValueError("IK 不可達：%s" % rep.strip())


def run(arm_name, cmd, args):
    arm = ARM_OBJ[arm_name]
    st = arm.state
    if cmd == "ping":                                   # 唯讀：測試指令通道是否活著
        return arm.dcall("RobotMode()")
    if cmd == "continue":                               # 恢復運動佇列（ClearError 後佇列可能停住）
        return arm.dcall("Continue()")
    if cmd == "enable":
        return arm.dcall("EnableRobot()")
    if cmd == "disable":
        arm.jog_stop()
        return arm.dcall("DisableRobot()")
    if cmd == "clear":
        return arm.dcall("ClearError()")
    if cmd == "stop":
        arm.jog_stop("停止")
        return arm.dcall("ResetRobot()")
    if cmd == "estop":
        arm.jog_stop("急停")
        return arm.dcall("EmergencyStop()")
    if cmd == "speed":
        return arm.dcall("SpeedFactor(%d)" % max(1, min(100, int(args["value"]))))
    if cmd == "do":
        idx, val = int(args["index"]), int(args["value"])
        if not 1 <= idx <= 16 or val not in (0, 1):
            raise ValueError("DO 參數錯誤")
        return arm.dcall("DOExecute(%d,%d)" % (idx, val))
    if cmd == "jog":
        if st.get("mode") not in (5, 11):
            raise ValueError("手臂需先使能（目前模式 %s）" % st.get("mode_name"))
        return arm.jog_start(args["axis"])
    if cmd == "jog_stop":
        return arm.jog_stop()
    if cmd in ("movj", "movl", "park"):
        if st.get("mode") != 5:
            raise ValueError("手臂需為「使能空閒」才能移動（目前 %s）" % st.get("mode_name"))
        if cmd == "park":
            x, y, z = PARK[arm_name]
            r = st["pose"][3]
            cur = st["pose"]
            if cur[2] < z - 1:                                   # 先只升不降
                check_target(arm, cur[0], cur[1], z, r)
                arm.mcall("MovL(%f,%f,%f,%f)" % (cur[0], cur[1], z, r))
            check_target(arm, x, y, z, r)
            return arm.mcall("MovJ(%f,%f,%f,%f)" % (x, y, z, r))
        x, y, z, r = (float(args[k]) for k in ("x", "y", "z", "r"))
        check_target(arm, x, y, z, r)
        return arm.mcall("%s(%f,%f,%f,%f)" % ("MovJ" if cmd == "movj" else "MovL", x, y, z, r))
    if cmd == "jointmovj":
        if st.get("mode") != 5:
            raise ValueError("手臂需為「使能空閒」才能移動")
        j = [float(args[k]) for k in ("j1", "j2", "j3", "j4")]
        return arm.mcall("JointMovJ(%f,%f,%f,%f)" % tuple(j))
    raise ValueError("未知指令 %s" % cmd)


# ---------- 相機（選用） ----------
CAM = {"err": None, "proc": None}


CAM_FILE = os.path.join(ROOT, "Arena", "cam_full.jpg") if WATCH else os.path.join(HERE, "cam_latest.jpg")
CAM_ERR = os.path.join(HERE, "cam_error.txt")


def camera_loop():
    """在子程序跑 camera_worker.py（只讀 Gemini 2）。Gemini 2 斷線時 Orbbec SDK 會直接結束整個程序
    （實測 2026-11-12 把控制台一起帶掛），所以隔離到子程序；子程序結束就記錄原因、5 秒後重啟。"""
    import subprocess
    while True:
        try:
            p = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "camera_worker.py")], cwd=HERE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            CAM["proc"] = p
            p.wait()
            why = ""
            try:
                why = open(CAM_ERR, encoding="utf-8").read().strip()
            except OSError:
                pass
            CAM["err"] = "Gemini 2 斷線或無回應（相機程序結束，碼 %s）%s" % (p.returncode, "：" + why if why else "")
        except Exception as e:
            CAM["err"] = "相機程序無法啟動：%s" % e
        time.sleep(5)


def camera_frame():
    """回傳 (jpeg bytes, 秒數)；超過 5 秒沒更新視為異常。"""
    try:
        age = time.time() - os.path.getmtime(CAM_FILE)
        if age > 5:
            return None, age
        return open(CAM_FILE, "rb").read(), age
    except OSError:
        return None, None


# ---------- HTTP ----------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code_, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code_)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        try:
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass                                      # 瀏覽器在傳完前就換下一張圖，正常

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        if self.path.startswith("/api/state"):
            jpg, _ = camera_frame()
            return self._send(200, {"watch": WATCH, "arms": {n: a.snapshot() for n, a in ARM_OBJ.items()},
                                    "camera": {"ok": jpg is not None, "err": None if jpg else CAM["err"]},
                                    "limits": {"z_min": Z_MIN, "park": PARK}})
        if self.path.startswith("/api/camera.jpg"):
            jpg, _ = camera_frame()
            if jpg is None:
                return self._send(503, {"error": CAM["err"] or "相機尚未就緒"})
            return self._send(200, jpg, "image/jpeg")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/cmd":
            return self._send(404, {"error": "not found"})
        if WATCH:
            return self._send(403, {"error": "只看模式（對弈伺服器在跑）：不能控制手臂"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            arms = list(ARM_OBJ) if req.get("arm") == "ALL" else [req["arm"]]
            out = {}
            for a in arms:
                try:
                    r = run(a, req["cmd"], req.get("args", {}))
                    out[a] = {"ok": code(r) == 0 if isinstance(r, str) and code(r) is not None else True,
                              "reply": r.strip() if isinstance(r, str) else str(r)}
                except Exception as e:
                    out[a] = {"ok": False, "reply": str(e)}
                if req["cmd"] not in ("jog",):             # 點動心跳不記錄
                    ARM_OBJ[a].note("%s %s → %s" % (req["cmd"], json.dumps(req.get("args", {}), ensure_ascii=False),
                                                    out[a]["reply"][:80]))
            self._send(200, out)
        except Exception as e:
            self._send(400, {"error": str(e)})


if __name__ == "__main__":
    if "--no-camera" not in sys.argv and not WATCH:
        threading.Thread(target=camera_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("MG400 控制台%s：http://127.0.0.1:%d" % ("（只看不控）" if WATCH else "", PORT))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        for a in ARM_OBJ.values():
            a.jog_stop("結束")
    finally:
        if CAM.get("proc"):
            CAM["proc"].kill()
