"""相機 + 手臂共用工具（手眼校正 / 井字棋用）。

相機：Orbbec Gemini 2（彩色 1280x720 MJPG、深度 1280x800 Y14），從正上方拍桌面。
手臂：重用 MG400/dual_sync.py 的 Arm（Dashboard 29999 + Move 30003 + 回饋 30004）。
"""
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "MG400"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ARM_IPS = {"A": "192.168.1.6", "B": "192.168.2.7"}


class UVCCamera:
    """Orbbec SDK 失敗時的備援：用 OpenCV（Windows MSMF）直接讀 Gemini 2 的 UVC 彩色鏡頭。
    以「能偵測到井字棋盤」挑出正確的鏡頭（index 0 通常是筆電內建鏡頭，不能用）。只有彩色。"""

    def __init__(self, index=None):
        import threading
        self.cap = None
        for idx in ([index] if index is not None else [3, 1, 2, 4]):
            cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            ok, f = False, None
            for _ in range(4):
                ok, f = cap.read()
            if ok and f.shape[:2] == (720, 1280) and self._looks_like_board(f):
                self.cap, self.index = cap, idx
                break
            cap.release()
        if self.cap is None:
            raise RuntimeError("找不到 Gemini 2 的 UVC 鏡頭")
        self.intr = dict(fx=687.77, fy=687.92, cx=639.51, cy=359.63, w=1280, h=720)   # 與 SDK 相同模式的內參
        self.has_depth = False
        self.latest, self.seq = None, 0
        self.cv = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.grab(skip=2)

    @staticmethod
    def _looks_like_board(img):
        try:
            from board import find_grid
            find_grid(img)
            return True
        except Exception:
            return False

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.cv:
                self.seq += 1
                self.latest = (time.perf_counter(), self.seq, f, None)
                self.cv.notify_all()

    def grab(self, n=None, depth=False, skip=2, timeout=20.0):
        """USB 降級時只有約 3 fps，所以逾時放寬。"""
        if depth:
            raise RuntimeError("UVC 備援沒有深度")
        if n is not None:
            skip = max(skip, n)
        t0 = time.perf_counter()
        with self.cv:
            target = (self.latest[1] if self.latest else 0) + 1 + skip
            while not (self.latest and self.latest[1] >= target and self.latest[0] > t0):
                if time.perf_counter() - t0 > timeout:
                    raise RuntimeError("UVC 取像逾時")
                self.cv.wait(0.2)
            return self.latest[2]

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


def Camera(depth=False):
    """先試 Orbbec SDK（正常時 30 fps）；失敗（例如 deviceMgr 為空）且只要彩色時改用 UVC（OpenCV MSMF）。"""
    try:
        return OrbbecCamera(depth=depth)
    except Exception as e:
        if depth:
            raise
        print("（Orbbec SDK 無法使用：%s → 改用 UVC）" % e)
        return UVCCamera()


class OrbbecCamera:
    """背景執行緒持續取像，只保留最新一張（含時間戳）。

    grab() 只回傳「呼叫之後」才拍到的影格（再額外跳過 skip 張），避免手臂移動期間 SDK 佇列堆積的舊影格。
    depth=True 時回傳對齊到彩色（1280x720）的深度 mm。"""

    def __init__(self, depth=False):
        """depth=False：只開彩色串流。手臂靠近相機時深度可能停止輸出，
        若同時開深度，管線會等不到配對而連彩色都停（實測）。需要深度才開 depth=True。"""
        import threading
        from pyorbbecsdk import AlignFilter, Config, OBSensorType, OBStreamType, Pipeline
        self.pipe = Pipeline()
        cfg = Config()
        cfg.enable_stream(self.pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile())
        if depth:
            cfg.enable_stream(self.pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile())
        self.has_depth = depth
        self.pipe.start(cfg)
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.intr = None
        try:
            i = self.pipe.get_camera_param().rgb_intrinsic
            self.intr = dict(fx=i.fx, fy=i.fy, cx=i.cx, cy=i.cy, w=i.width, h=i.height)
        except Exception:
            pass
        self.want_depth = False
        self.latest = None                    # (t, seq, color, depth)
        self.seq = 0
        self.cv = threading.Condition()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.grab(skip=10)                    # 暖機（自動曝光）

    def _loop(self):
        from pyorbbecsdk import OBFormat
        while self.running:
            try:
                fr = self.pipe.wait_for_frames(200)
            except Exception:
                continue
            if fr is None:
                continue
            t = time.perf_counter()
            try:
                dep = None
                if self.want_depth:
                    al = self.align.process(fr)
                    al = al.as_frame_set() if hasattr(al, "as_frame_set") else al
                    d = al.get_depth_frame() if al is not None else None
                    if d is not None:
                        dep = np.frombuffer(d.get_data(), dtype=np.uint16).reshape(
                            d.get_height(), d.get_width()).astype(np.float32) * d.get_depth_scale()
                c = fr.get_color_frame()
                if c is None:
                    continue
                raw = np.asanyarray(c.get_data())
                w, h, fmt = c.get_width(), c.get_height(), c.get_format()
                if fmt == OBFormat.MJPG:
                    color = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                elif fmt == OBFormat.RGB:
                    color = cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
                else:
                    color = raw.reshape(h, w, 3).copy()
            except Exception:
                continue
            with self.cv:
                self.seq += 1
                self.latest = (t, self.seq, color, dep)
                self.cv.notify_all()

    def grab(self, n=None, depth=False, skip=2, timeout=5.0):
        """等「現在之後」的影格，再跳過 skip 張，回傳最新一張。n 為舊參數，視同 skip。"""
        if n is not None:
            skip = max(skip, n)
        if depth and not self.has_depth:
            raise RuntimeError("此 Camera 沒開深度：請用 Camera(depth=True)")
        self.want_depth = self.want_depth or depth
        t0 = time.perf_counter()
        with self.cv:
            start = self.latest[1] if self.latest else 0
            target = start + 1 + skip
            while True:
                ok = self.latest is not None and self.latest[1] >= target and self.latest[0] > t0                     and (not depth or self.latest[3] is not None)
                if ok:
                    break
                if time.perf_counter() - t0 > timeout:
                    age = time.perf_counter() - self.latest[0] if self.latest else -1
                    raise RuntimeError("相機取像逾時（最新影格 seq=%s、%.1f s 前；執行緒存活=%s）"
                                       % (self.latest[1] if self.latest else None, age, self.thread.is_alive()))
                self.cv.wait(0.2)
            _, _, color, dep = self.latest
        return (color, dep) if depth else color

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.pipe.stop()


def connect_arm(name):
    import dual_sync
    return dual_sync.Arm(name, ARM_IPS[name])


def joint_angles(arm):
    import dual_sync
    return dual_sync.values(arm.dash.GetAngle())[:4]


def sync_move(arm, cmd, timeout=40.0):
    """送出運動指令，用 30004 回饋判斷到位（不依賴 Sync()：實測它可能在動作完成前就回傳）。

    先等手臂開始動（模式 7 或位置變化），再等模式回到 5 且位置 0.3 s 內變化 < 0.05 mm。"""
    import dual_sync
    p_start = arm.pose().copy()
    dual_sync.check(arm.move.sendRecvMsg(cmd), cmd)
    t0 = time.perf_counter()
    started = False
    while time.perf_counter() - t0 < 1.5:                    # 等開始動
        if arm.mode() == 7 or np.abs(arm.pose() - p_start).max() > 0.05:
            started = True
            break
        if arm.mode() == 9:
            raise RuntimeError("%s 報警 %s（%s）" % (arm.name, arm.errors(), cmd))
        time.sleep(0.01)
    stable_since, last = None, arm.pose().copy()
    while time.perf_counter() - t0 < timeout:
        m, p = arm.mode(), arm.pose().copy()
        if m == 9:
            raise RuntimeError("%s 運動中報警 %s（%s）" % (arm.name, arm.errors(), cmd))
        if m == 5 and np.abs(p - last).max() < 0.05:
            stable_since = stable_since or time.perf_counter()
            if time.perf_counter() - stable_since > 0.3:
                return started
        else:
            stable_since = None
        last = p
        time.sleep(0.02)
    raise RuntimeError("%s 等待到位逾時（%s）" % (arm.name, cmd))


def enable(arm, speed=10):
    import dual_sync
    if arm.mode() == 9:
        dual_sync.check(arm.dash.ClearError(), "ClearError")
        time.sleep(0.5)
    dual_sync.check(arm.dash.SpeedFactor(speed), "SpeedFactor")
    dual_sync.check(arm.dash.EnableRobot(), "EnableRobot")
    time.sleep(1.0)
    if arm.mode() != 5:
        raise RuntimeError("%s 使能後模式為 %s" % (arm.name, arm.mode()))
