"""OOXX 對弈機器人：語音互動＋即時盤面＋雙臂。手機／平板／電腦都能開。

    python server.py                   # 雙臂：https://<本機IP>:8443/?k=<存取碼>（存取碼印在畫面上、存於 access.txt）
    python server.py --no-robot        # 不連手臂（只測介面、語音、盤面）
    python server.py --dry             # 乾跑：模擬手臂＋模擬盤面（點格子＝人擺 X），不連手臂也不開相機

- 人下 X（手擺棋子，或用語音說「幫我下右上角」由機器代放）；機器下 O。機器對下：O 由 B、X 由 A（照 ox_record.json）。
- 雙臂模式的讀盤與搬運都在 carrier.py：棋子翻面，O／X 只靠紀錄；搆不到的格經棋盤外的轉送點交給另一台。
- 盤面：Gemini 2（子程序 cam_worker.py）→ 棋盤偵測；輪到人時每 5 秒判斷一次有沒有新落子（使用者指定）。
- 語音：瀏覽器錄音 → GX10-B ASR → LLM（gemma4-31b）理解意圖 → 動作 → 千鉑 Studio TTS 回話。
- 安全：HTTPS＋存取碼；「停」（按鈕或語音）立刻 ResetRobot；手臂只在機器回合動。
"""
import base64
import json
import os
import re
import secrets
import ssl
import subprocess
import sys
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Calib"))
import hand                                                  # noqa: E402
import find_pieces                                           # noqa: E402
import drift_check                                           # noqa: E402
import services                                              # noqa: E402
from board import classify_cells, find_grid                  # noqa: E402
from game import best_move, winner, LINES                    # noqa: E402

PORT = 8443
CAM_FILE = os.path.join(HERE, "cam_full.jpg")
CAM_ERR = os.path.join(HERE, "cam_error.txt")
CAM_DEPTH_FILE = os.path.join(HERE, "cam_depth.png")
CAM_DEPTH = False                    # True：相機子程序用深度模式（只在交接時）
HAND_BG = os.path.join(HERE, "hand_bg.png")   # 沒有手時的背景深度（手臂都在停放點）；交接時和當下取「較遠」當參考
COUNT_FILE = os.path.join(ROOT, "Calib", "stock_count.json")
PLAN_FILE = os.path.join(ROOT, "Calib", "cell_plan.json")
NAMES = ["左上", "上中", "右上", "左中", "中間", "右中", "左下", "下中", "右下"]
HUMAN, ROBOT = "X", "O"
MARK = {"X": "叉", "O": "圈"}


def set_roles(human):
    """人用哪一種棋子（另一種歸機器）。人第一手放了圈，就自動改成人下圈、機器下叉（使用者現場一直拿圈下）。"""
    global HUMAN, ROBOT
    HUMAN, ROBOT = ("O", "X") if human == "O" else ("X", "O")


def _turn_text():
    return "輪到你下%s。" % MARK[HUMAN]

POLL_HUMAN = 1.2                     # 使用者：5 秒判斷一次，沒落子就再等 5 秒
DRY = "--dry" in sys.argv
DRY_ARM = "--dry-arm" in sys.argv
ALWAYS_DEPTH = not DRY and not DRY_ARM and "--b-only" not in sys.argv     # 雙臂＋自動料區（預設）
DRY_FAIL_PICKS = int(next((a[12:] for a in sys.argv if a.startswith("--dry-fails=")), "0"))   # 乾跑：模擬前幾次沒吸起來
HANDOVER_ON = "--handover" in sys.argv   # 放棋子到手掌（已放棄，2026-11-14）：要用才加這個參數    # 只有手臂是模擬的：相機、讀盤、手掌偵測、監看都是真的（交接乾跑）
REACH_CACHE = os.path.join(HERE, "reach_cache.json")   # 真手臂連上時存，乾跑時沿用
VBOARD = [" "] * 9                   # 乾跑用的「實體棋盤」


def _local_ips():
    """這台電腦目前的 IPv4（換電腦不用改程式）；手臂網段（192.168.1.x／2.x 的 .3）與 169.254 不列。"""
    import socket
    ips = ["127.0.0.1"]
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in ips and not ip.startswith("169.254.") and not ip.endswith(".3"):
                ips.append(ip)
    except OSError:
        pass
    return tuple(ips)


# ======================= 相機 =======================
def camera_supervisor():
    while True:
        try:
            args = [sys.executable, "-u", os.path.join(HERE, "cam_worker.py")]
            p = subprocess.Popen(args, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            CTRL.cam_proc = p
            p.wait()
        except Exception:
            pass
        time.sleep(0.5 if CTRL.cam_switch else 5)
        CTRL.cam_switch = False


def set_camera_depth(on, timeout=25.0):
    """切換相機子程序的深度模式（重開子程序）；on=True 時等到有新的深度影像。回傳是否成功。"""
    global CAM_DEPTH
    if DRY:
        return True
    if not on and ALWAYS_DEPTH:                      # 雙臂模式：深度一直開著（料區要隨時量）
        return True
    if CAM_DEPTH != on:
        CAM_DEPTH = on
        CTRL.cam_switch = True
        if CTRL.cam_proc:
            CTRL.cam_proc.kill()
    if not on:
        return True
    t0 = time.time()
    while time.time() - t0 < timeout:
        if latest_depth(max_age=1.0) is not None:
            return True
        time.sleep(0.3)
    return False


def latest_depth(max_age=2.0):
    try:
        if time.time() - os.path.getmtime(CAM_DEPTH_FILE) > max_age:
            return None
        d = cv2.imread(CAM_DEPTH_FILE, cv2.IMREAD_UNCHANGED)
        return None if d is None else d.astype(np.float32)
    except Exception:
        return None


def capture_depth(n=8, warmup=1.2, timeout=15.0):
    """深度模式下收 n 張「不同的」深度影像（先丟掉剛開串流的 warmup 秒）。"""
    time.sleep(warmup)
    out, last_t, t0 = [], 0.0, time.time()
    while len(out) < n and time.time() - t0 < timeout:
        try:
            t = os.path.getmtime(CAM_DEPTH_FILE)
        except OSError:
            t = 0.0
        if t != last_t:
            d = latest_depth(max_age=1.0)
            if d is not None and (d > 0).mean() > 0.8:
                out.append(d)
                last_t = t
        time.sleep(0.05)
    return out


_FRAME = {"key": None, "img": None}
_FRAME_LOCK = threading.Lock()


def camera_fresh(max_age=9.0):
    try:
        return time.time() - os.path.getmtime(CAM_FILE) <= max_age
    except OSError:
        return False


def latest_frame(max_age=9.0):
    """最新的相機畫面。同一張檔案只解碼一次（主迴圈、網頁、校正檢查會一直來讀）；回傳複本，呼叫端可以在上面畫。"""
    try:
        st = os.stat(CAM_FILE)
        if time.time() - st.st_mtime > max_age:
            return None
        key = (st.st_mtime_ns, st.st_size)
        with _FRAME_LOCK:
            if _FRAME["key"] != key:
                img = cv2.imread(CAM_FILE)
                if img is None:
                    return None
                _FRAME["key"], _FRAME["img"] = key, img
            return _FRAME["img"].copy()
    except Exception:
        return None


def read_board_once():
    if DRY:
        return list(VBOARD)
    img = latest_frame()
    if img is None:
        return None
    try:                                    # 以棋子為主讀盤（2026-11-14：舊的「看格子中心」會把擺歪的 X 認成 O）
        if CTRL.flip:                       # 雙臂（棋子翻面）：種類照紀錄，沒紀錄的是「?」
            return CTRL.flip.read_board(img, latest_depth() if CTRL.flip.maps is None else None)   # 深度只有還沒找格線時要用
        return find_pieces.read_board(img, CTRL.plan)
    except Exception:                       # 手或手臂蓋住棋盤時偵測會失敗：當成這一張讀不到
        return None


def read_board(n=3, gap=0.4):
    """多幀多數決（單張會把 X 判成 O）。"""
    reads = []
    for _ in range(n * 2):
        b = read_board_once()
        if b:
            reads.append(b)
        if len(reads) >= n:
            break
        time.sleep(gap)
    if not reads:
        return None
    return [max(set(r[i] for r in reads), key=[r[i] for r in reads].count) for i in range(9)]


class FileCam:
    """carrier.Scene 的相機：讀 cam_worker 寫的檔（已套對齊）。先要求「這一刻之後」的深度（depth_request，約 5 張）取中位數，
    再等下一張彩色。相機斷流時丟 RuntimeError（訊息含「相機沒有影像」，_stocks 會等一下重試）。"""

    def grab(self):
        import align
        req = os.path.join(HERE, "depth_request")
        t_req = time.time()
        open(req, "w").close()
        ds, last_t = [], 0.0
        while time.time() - t_req < 30:
            try:
                mt = os.path.getmtime(CAM_DEPTH_FILE)
            except OSError:
                mt = 0.0
            if mt > t_req and mt != last_t:
                d = cv2.imread(CAM_DEPTH_FILE, cv2.IMREAD_UNCHANGED)
                if d is not None:
                    ds.append(d.astype(np.float32))
                    last_t = mt
            if not os.path.exists(req) and (ds or time.time() - t_req > 3):
                break
            time.sleep(0.05)
        t_back = time.time()
        img = None
        while time.time() - t_back < 12:
            try:
                if os.path.getmtime(CAM_FILE) > t_back:
                    img = cv2.imread(CAM_FILE)
                    if img is not None:
                        break
            except OSError:
                pass
            time.sleep(0.1)
        if img is None or len(ds) < 2:
            raise RuntimeError("相機沒有影像，看不到料區")
        self.frames = ds                                    # 量支架用（posts.near_depth）
        return img, np.median(np.stack(ds), axis=0).astype(np.float32), align.load()


# ======================= 交接監看 =======================
class HandGuard:
    """交接最後一段（手臂移到手上方、慢降、放開）的監看。

    開始時把「手＋前臂」的像素記下來（基準）。之後每張深度影像只問兩件事：
    1. 基準像素裡還有多少仍是「手」（比背景近）——手移開或抽走，這個比例會掉；
       手臂伸進來只會讓畫面多出東西，不會讓基準像素變空，所以不會因為手臂自己誤判（2026-11-14 舊版用重心位移，被手臂連桿／真空管誤觸）。
    2. 離手臂很遠的那段前臂有沒有往上抬（深度變近）。
    深度影像超過 STALE_S 沒更新也停（看不到就不能靠近手）。"""
    KEEP_MIN = 0.6           # 基準像素還在的比例低於這個 → 手移開了（約 2～3 cm）
    RISE_MM = 20.0           # 前臂抬高超過這個 → 停
    STALE_S = 1.5

    def __init__(self, ctrl):
        self.ctrl = ctrl
        self._trip = None
        self.running = False
        self.th = None

    def tripped(self):
        return self._trip

    @staticmethod
    def _ipt(p):
        return int(round(p[0])), int(round(p[1]))

    def start(self, stage, target, z, on_trip):
        """開始監看；看不夠清楚、沒辦法監看時回傳原因（呼叫端就不靠近手）。"""
        self.stop()
        if DRY:
            return None
        from plan_cells import face_px
        cal, ref = self.ctrl.arm.cal, self.ctrl.hand_ref
        far = np.zeros(ref.shape, np.uint8)            # 手臂可能出現的區域（取很寬：連桿較高，透視會往外偏）
        base = face_px(cal, (0.0, 0.0), z)
        for q in (stage, target):
            qp = face_px(cal, q, z)
            cv2.line(far, self._ipt(base), self._ipt(qp), 1, 330)
            cv2.circle(far, self._ipt(qp), 200, 1, -1)
        d = latest_depth(max_age=self.STALE_S)
        if d is None:
            return "相機深度中斷"
        hand_m = ((d > 0) & (ref > 0) & (ref - d > hand.DIFF_MM)).astype(np.uint8)
        hand_m = cv2.morphologyEx(hand_m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(hand_m)
        if n < 2:
            return "看不到你的手"
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        base_m = (lab == k) & (far == 0)                # 基準：手臂不會經過的那部分手（前臂、手指）
        n0 = int(base_m.sum())
        if n0 < 1500:
            return "我看不到你的手臂，請從旁邊伸過來（不要從我這一側）"
        d0 = float(np.median(d[base_m]))
        self.running = True
        print("[監看] 開始：基準 %d 像素、前臂深度 %.0f mm" % (n0, d0), flush=True)

        def loop():
            last_t = 0.0
            while self.running:
                try:
                    t = os.path.getmtime(CAM_DEPTH_FILE)
                except OSError:
                    t = 0.0
                if time.time() - t > self.STALE_S:
                    return self._fire("相機深度中斷", on_trip)
                if t != last_t:
                    last_t = t
                    dd = latest_depth(max_age=self.STALE_S)
                    if dd is None:
                        continue
                    present = base_m & (dd > 0) & (ref - dd > hand.DIFF_MM)
                    keep = present.sum() / n0
                    rise = d0 - float(np.median(dd[present])) if present.sum() > 300 else 0.0
                    if keep < self.KEEP_MIN:
                        print("[監看] 手移開：基準像素剩 %.0f%%" % (keep * 100), flush=True)
                        return self._fire("你的手移開了", on_trip)
                    if rise > self.RISE_MM:
                        print("[監看] 手抬高 %.0f mm" % rise, flush=True)
                        return self._fire("你的手抬高了", on_trip)
                time.sleep(0.03)

        self.th = threading.Thread(target=loop, daemon=True)
        self.th.start()
        return None

    def _fire(self, reason, on_trip):
        self._trip = reason
        self.running = False
        try:
            on_trip()
        except Exception:
            pass
        with self.ctrl.lock:
            self.ctrl.status = "停住了：%s" % reason

    def stop(self):
        self.running = False
        if self.th is not None and self.th is not threading.current_thread():
            self.th.join(timeout=1.0)
        self.th = None
        self._trip = None


class _MoverAdapter:
    """舊程式碼用到 self.arm 的地方（cal、stop、close）接到雙臂 Mover。"""

    def __init__(self, mover):
        self.mover = mover
        self.cal = mover.arms["B"].cal

    def stop(self):
        for a in self.mover.arms.values():
            try:
                a.arm.dash.sendRecvMsg("DOExecute(1,0)")
                a.arm.dash.ResetRobot()
                a.arm.dash.DisableRobot()
            except Exception:
                pass

    def close(self):
        self.mover.close()


# ======================= 對弈控制器 =======================
class Controller:
    def __init__(self, use_robot=True):
        self.lock = threading.RLock()
        self.calib_lock = threading.Lock()
        self.board = [" "] * 9          # 認定的盤面（遊戲狀態）
        self.seen = None                # 相機最新讀到的盤面
        self.phase = "idle"             # idle／human／robot／over／paused
        self.result = None
        self.win_line = None
        self.last_move = None
        self.status = "按「新局」開始。"
        self.messages = []              # [{id, role, text, audio, t}]
        self.clients = {}               # 網頁分頁代號 → (最後輪詢時間, 有沒有開聲音)
        self.sound_owner = None         # 只有這個分頁出聲（開好幾個網頁時不會回聲，使用者 2026-11-16）
        self.reset_req = False          # 網頁按了「重置」：手上那一步做完就收棋盤、回到開局前（_do_reset）
        self.reset_n = 0                # 重置過幾次（網頁看到變了就清掉訊息）
        self.audio = {}                 # id → wav bytes
        self.stop_flag = False
        self.cam_proc = None
        self.cam_switch = False
        self.palm_vis = None            # 交接時給網頁看的 (palm, reason)
        self.hand_ref = None            # 交接用的參考深度（沒伸手時）
        self.guard = HandGuard(self)
        self.prev_phase = None
        self.calib_shift = None         # 底座位移（px），網頁顯示
        self.next_poll = 0.0
        self.counts = self._load_counts()
        self.plan = json.load(open(PLAN_FILE, encoding="utf-8"))
        self.arm = None
        self.mover = None
        self.flip = None                # 雙臂模式的場景（carrier.Scene）：讀盤種類、料疊、搬運都經過它
        self.last_stocks = []
        self.reach = {}
        self.arm_err = None
        if DRY or DRY_ARM:
            from robot_sim import SimHandArm
            self.arm = SimHandArm()
            try:
                self.reach = {int(k): v for k, v in json.load(open(REACH_CACHE, encoding="utf-8")).items()}
            except Exception:
                self.reach = {i: None for i in (0, 1, 2, 4, 5, 7, 8)}
        elif use_robot and ALWAYS_DEPTH:
            try:
                import carrier
                from mover import Mover
                self.mover = Mover()
                self.arm = _MoverAdapter(self.mover)
                self.flip = carrier.Scene(FileCam(), self.mover, stop=lambda: self.stop_flag)
                self.reach = {i: None for i in range(9)}             # 每步依當下料區重算
            except Exception as e:
                self.arm_err = "手臂連線失敗：%s" % e
        elif use_robot:
            try:
                from robot import ArmB
                self.arm = ArmB()
                self.reach = self.arm.reachable(self.plan)
                json.dump({str(k): [float(v[0]), float(v[1])] for k, v in self.reach.items()},
                          open(REACH_CACHE, "w", encoding="utf-8"))
            except Exception as e:
                self.arm_err = "手臂 B 連線失敗：%s" % e
        self.queue = []                 # 待執行的機器動作
        self.first = HUMAN              # 這局誰先下（決定繼續時輪到誰）
        self.tts_q = queue.Queue()      # 語音合成依序做，播放順序才不會亂
        threading.Thread(target=self._loop, daemon=True).start()
        threading.Thread(target=self._tts_worker, daemon=True).start()
        threading.Thread(target=self._prewarm_voice, daemon=True).start()

    # ---------- 狀態 ----------
    def _load_counts(self):
        if DRY:
            return {"X疊": 5, "O疊": 5}                 # 乾跑不動真實計數檔
        try:
            return json.load(open(COUNT_FILE, encoding="utf-8"))
        except Exception:
            return {"X疊": 0, "O疊": 0}

    def _save_counts(self):
        if DRY or DRY_ARM:                          # 乾跑不動真實計數檔
            return
        json.dump(self.counts, open(COUNT_FILE, "w", encoding="utf-8"), ensure_ascii=False)

    def touch_client(self, cid, snd):
        if cid:
            self.clients[cid] = (time.time(), snd)
            if len(self.clients) > 50:
                for k in sorted(self.clients, key=lambda k: self.clients[k][0])[:-20]:
                    del self.clients[k]

    def audio_owner(self):
        """出聲的分頁：最後按「開聲音」的那個；它關掉（15 秒沒輪詢）或關了聲音，就換最近輪詢、有開聲音的那個。"""
        now = time.time()
        c = self.clients.get(self.sound_owner)
        if c is None or now - c[0] > 15 or not c[1]:
            live = [(t, k) for k, (t, snd) in self.clients.items() if snd and now - t <= 15]
            self.sound_owner = max(live)[1] if live else None
        return self.sound_owner

    def snapshot(self):
        with self.lock:
            return {
                "sound_owner": self.audio_owner(), "reset_n": self.reset_n,
                "board": self.board, "seen": self.seen, "phase": self.phase, "result": self.result,
                "win_line": self.win_line, "last_move": self.last_move, "status": self.status,
                "robot_cells": sorted(self.reach),
                "counts": ({"O疊": sum(x["n"] for x in self.last_stocks if x["kind"] == "O"),
                            "X疊": sum(x["n"] for x in self.last_stocks if x["kind"] == "X")}
                           if self.mover else self.counts),
                "stocks": [{"kind": x["kind"], "n": x["n"]} for x in self.last_stocks], "arm_error": self.arm_err,
                "palm": ({"h": round(self.palm_vis[0]["h"]), "ok": self.palm_vis[1] is None}
                         if self.palm_vis and self.palm_vis[0] else None),
                "calib_shift": self.calib_shift, "calib_limit": drift_check.BASE_OK_PX, "human": HUMAN, "hint": getattr(self, "turn_hint", None),
                "camera_ok": DRY or camera_fresh(), "names": NAMES, "dry": DRY, "dry_arm": DRY_ARM,
                "messages": [{k: m[k] for k in ("id", "role", "text", "audio", "t")} for m in self.messages[-40:]],
            }

    # ---------- 說話 ----------
    def say(self, text, role="robot", speak=True):
        mid = uuid.uuid4().hex[:10]
        msg = {"id": mid, "role": role, "text": text, "audio": None, "t": time.time()}
        with self.lock:
            self.messages.append(msg)
            self.messages = self.messages[-80:]
        if speak and role == "robot":
            wav = services.tts_cached(text)
            if wav:                                            # 固定提示句：已有存檔，立刻可以播
                with self.lock:
                    self.audio[mid] = wav
                msg["audio"] = "/api/audio/%s.wav" % mid
            else:
                self.tts_q.put(msg)
        return msg

    PREWARM = [                                           # 對弈會講的固定句子（%s＝九格名稱）；2026-11-15 換掉已放棄的交接功能那批
        "我下%s。", "我下%s，這步之後我就贏定了！", "圈下%s。", "叉下%s。",
        "新局開始！我下圈，你下叉。你先請。", "新局開始！我下圈，你下叉。我先下。", "棋盤上還有棋子，我先收乾淨再開局。",
        "你連成一線了，你贏了！好厲害。", "我連成一線，我贏了！要不要再來一盤？", "和局，平手！再來一盤嗎？",
        "棋盤收好了，要再來一盤嗎？", "棋盤收好了。", "我正在動，等我一下。",
        "好，我自己跟自己下一盤：圈是高手、叉是普通程度。", "再來一盤！",
        "圈贏了！對下結束。", "叉贏了！對下結束。", "已經分不出勝負，和局！對下結束。", "和局！對下結束。",
        "我看不到圈的料疊，手先不要擋在上面。", "我看不到叉的料疊，手先不要擋在上面。",
        "有棋子被移動了，我把它放回去。", "放回去了，換你下。", "好，我停下來了。", "繼續！"]

    def _prewarm_voice(self):
        """固定提示句先合成存檔（Breeze 每句要 4～5 秒，臨時合成會慢半拍）。開機時在背景做，已經有存檔的跳過。"""
        if DRY:
            return
        texts = []
        for t in self.PREWARM:
            if "%d" in t:
                texts += [t % h for h in range(101, 160)]
            elif "%s" in t:
                texts += [t % n for n in NAMES]
            else:
                texts.append(t)
        n = 0
        for t in texts:
            if services.tts_cached(t) is None:
                try:
                    services.tts(t)
                    n += 1
                except Exception:
                    pass
        if n:
            print("[語音] 預先合成 %d 句" % n, flush=True)

    def _tts_worker(self):
        while True:
            msg = self.tts_q.get()
            try:
                wav = services.tts(msg["text"])
                with self.lock:
                    self.audio[msg["id"]] = wav
                    if len(self.audio) > 30:
                        for k in list(self.audio)[:-30]:
                            del self.audio[k]
                    msg["audio"] = "/api/audio/%s.wav" % msg["id"]
            except Exception as e:
                msg["text"] += "（語音合成失敗：%s）" % str(e)[:60]

    def _robot_turn(self):
        x, o = self.board.count(HUMAN), self.board.count(ROBOT)
        return o < x if self.first == HUMAN else o <= x

    # ---------- 對弈 ----------
    def calib_check(self):
        # 開機檢查與開新局可能同時呼叫：兩邊都拿同一張舊畫面估計、各自疊一次修正，就會修兩倍
        import align
        with self.calib_lock:
            before = align.load()
            r = self._calib_check()
            if self.flip and not np.allclose(before, align.load()):
                self.flip.reset()                        # 影像對齊改了：格線、支架都要重找
            return r

    def _calib_check(self):
        """相機跟手臂的相對位置有沒有變。回傳 (ok, 說明)。
        先看棋盤格線（桌面這個平面，最準）：偏了就自己對齊（Calib/align.py：算出影像修正量，相機子程式之後每張都套），
        不用人來重新校正。格線看不到（棋盤上有棋子擋住）才退回用底座比對。"""
        if DRY:
            return True, "（模擬）"
        img = latest_frame()
        if img is None:
            return True, "相機沒有影像，沒檢查"
        import align

        def frame_after(t, timeout=15.0):
            """等相機子程式寫出「t 之後」的畫面（套用了新對齊的那一張），不要用固定的 sleep 去猜。"""
            t_end = time.time() + timeout
            seen = set()                                 # cam_worker 每輪是「讀對齊檔 → 等影格 → 寫檔」：t 之後的第一張可能還是舊對齊
            while time.time() < t_end:
                try:
                    mt = os.path.getmtime(CAM_FILE)
                except OSError:
                    mt = 0.0
                if mt > t:
                    seen.add(mt)
                    if len(seen) >= 2:
                        return latest_frame()
                time.sleep(0.05)
            return None

        r = align.estimate(img)
        ps = None
        try:                                             # 棋盤上有棋子時格線被蓋住，估出來的偏移不準（2026-11-15 有 8 顆時誤判偏 10 px）
            ps = find_pieces.pieces(img, self.plan, blank_ok=bool(self.flip))   # 翻面的棋子看不到符號，也要算
            occupied = len(ps) > 0
        except Exception:
            occupied = True
        if occupied:
            # 棋盤上有棋子：用棋子位置粗對齊（只修平移），手臂才能準確收棋盤；收空、開局時再用格線精確對齊
            pe = align.estimate_from_pieces(img, ps=ps)
            if pe is not None:
                T, (dx, dy), n, spread = pe
                sh = float(np.hypot(dx, dy))
                if sh <= 5.0:
                    with self.lock:
                        self.calib_shift = sh
                    return True, "校正正常（用棋盤上 %d 顆棋子檢查，偏 %.1f 像素）" % (n, sh)
                M_old = align.load()
                align.save(align.compose(T, M_old), "pieces %.1f px n=%d" % (sh, n))
                t_save = time.time()
                print("[校正檢查] 棋盤上 %d 顆棋子一致偏了 (%+.0f, %+.0f) 像素 → 用棋子位置粗對齊" % (n, dx, dy), flush=True)
                img2 = frame_after(t_save)
                pe2 = None
                if img2 is not None:                     # 確認時也要算翻面的棋子（不然雙臂模式一定量不到、一定還原）
                    try:
                        pe2 = align.estimate_from_pieces(img2, ps=find_pieces.pieces(img2, self.plan, blank_ok=bool(self.flip)))
                    except Exception:
                        pe2 = None
                after = None if pe2 is None else float(np.hypot(*pe2[1]))
                if after is None or after > 6.0:          # 沒能確認變好（相機太慢等不到新畫面，或真的沒改善）：還原
                    align.save(M_old, "revert")
                    return False, ("棋子位置看起來偏了 %.0f 像素，%s，已還原；請把棋盤清空再檢查" %
                                   (sh, "相機太慢、沒等到新畫面確認" if after is None else "但粗對齊沒有改善"))
                with self.lock:
                    self.calib_shift = after if after is not None else sh
                self.say("相機好像被碰到了，偏了 %.0f 像素，我用棋盤上的棋子先粗略對齊好了。" % sh, speak=False)
                return True, "相機偏 %.0f 像素，已用棋子粗對齊（現在偏 %s 像素）；棋盤清空後會再用格線精確對齊" % (
                    sh, "?" if after is None else "%.1f" % after)
            if r is not None and r[1] <= 14.0:
                with self.lock:
                    self.calib_shift = r[1]
                return True, "棋盤上有棋子，格線只能粗看（偏 %.1f 像素），等棋盤清空再精確對齊" % r[1]
            r = None                                     # 偏很多：交給下面的底座比對
        if r is not None:
            M, shift, res = r
            with self.lock:
                self.calib_shift = shift
            if shift <= 5.0:
                return True, "校正正常（棋盤格線偏 %.1f 像素）" % shift
            M_old = align.load()
            align.save(align.compose(M, M_old), "auto %.1f px" % shift)       # 相機被碰歪：自動對齊
            t_save = time.time()
            print("[校正檢查] 棋盤格線偏了 %.1f 像素 → 自動對齊（殘差 %.1f）" % (shift, res), flush=True)
            img2 = frame_after(t_save)
            r2 = align.estimate(img2) if img2 is not None else None
            if r2 is not None and r2[1] <= 6.0:
                # 格線對齊只知道「棋盤在畫面上回到原位」。如果其實是棋盤被推動（相機沒動），固定在檯面上的兩個底座
                # 就會在對齊後反而偏掉——這種情況不能用對齊蓋過去，要還原並請人處理（手臂和棋盤的關係變了）
                bm, bdet = drift_check.base_shift(img2)
                bgood = [float(np.hypot(v[0], v[1])) for v in (bdet or {}).values() if len(v) >= 3 and v[2] >= 0.9]
                if bgood and max(bgood) > 24.0:
                    align.save(M_old, "revert: bases moved %.0f px" % max(bgood))
                    return False, ("棋盤格線偏了 %.0f 像素，但對齊後底座反而偏了 %.0f 像素："
                                   "看起來是棋盤被推動、不是相機被碰。已還原，請重新做現場校正。" % (shift, max(bgood)))
                with self.lock:
                    self.calib_shift = r2[1]
                self.say("相機好像被碰到了，偏了 %.0f 像素，我已經自己對齊好了。" % shift, speak=False)
                return True, "相機被碰歪 %.0f 像素，已自動對齊（現在偏 %.1f）" % (shift, r2[1])
            align.save(M_old, "revert")
            return False, "相機被移動太多，自動對齊沒成功（已還原），請重新校正。"
        m, det = drift_check.base_shift(img)
        # 比對分數低（< 0.9）的底座不採計；套過自動對齊後底座會有視差（底座比桌面高），門檻放寬
        good = [float(np.hypot(v[0], v[1])) for v in (det or {}).values() if len(v) >= 3 and v[2] >= 0.9]
        m = max(good) if good else None
        limit = drift_check.BASE_OK_PX if np.allclose(align.load(), align.I, atol=1e-6) else 16.0
        with self.lock:
            self.calib_shift = m
        if m is None:
            return True, "棋盤格線與底座都看不清楚，這次沒檢查"
        if m > limit:
            return False, "相機或手臂好像被移動過（底座偏了 %.0f 像素）。請把棋盤清空，我就能自己重新對齊。" % m
        return True, "校正正常（底座偏 %.1f 像素）" % m

    def measure_stacks(self, kinds=None):
        """用深度量料區每一疊的頂面高度與每顆厚度（棋子貼膠帶後約 3.8～4.0 mm，不是 3.7），給吸取用。
        雙臂：kinds（預設機器那一種；機器對下兩種都要）每種至少 4 顆。"""
        if self.mover:                                   # 雙臂版每步都重新偵測料區
            S = self._stocks()
            print("[料區] %s" % "；".join("%s×%d 影像(%d,%d)" % (x["kind"], x["n"], *x["px"]) for x in S), flush=True)
            n = {k: sum(x["n"] for x in S if x["kind"] == k) for k in ("O", "X")}
            short = [k for k in (kinds or (ROBOT,)) if n[k] < 4]
            return None if not short else "看到的圈 %d 顆、叉 %d 顆（%s至少要 4 顆）" % (
                n["O"], n["X"], "、".join(MARK[k] for k in short))
        if DRY or DRY_ARM or not self.arm:
            return None
        import robot
        try:
            if not set_camera_depth(True):
                return "相機深度沒出來"
            ds = capture_depth(6)
            if len(ds) < 3:
                return "深度影像不夠"
            d = np.pad(np.median(np.stack(ds), axis=0), 120)
            msg = []
            for name, v in robot.STOCK["stocks"].items():
                n = self.counts.get(name, 0)
                if n < 1:
                    continue
                u, w = int(v["px"][0]) + 120, int(v["px"][1]) + 120
                top = d[w - 12:w + 13, u - 12:u + 13]
                top = float(np.median(top[top > 0]))
                yy, xx = np.mgrid[-110:110, -110:110]
                rr = np.hypot(xx, yy)
                ring = d[w - 110:w + 110, u - 110:u + 110]
                table = float(np.median(ring[(rr > 75) & (rr < 105) & (ring > 0)]))
                th = (table - top) / n
                if not (3.0 <= th <= 5.0):
                    msg.append("%s 量到每顆 %.1f mm，數量可能不對（記 %d 顆）" % (name, th, n))
                    continue
                robot.STACK_Z[name] = (robot.Z_TOUCH_BOARD + (robot.STD_TOP_DEPTH - top), n, th)
                print("[料區] %s：%d 顆、每顆 %.1f mm、頂端接觸 Z=%.1f" % (name, n, th, robot.STACK_Z[name][0]), flush=True)
            return "；".join(msg) or None
        finally:
            set_camera_depth(False)

    def new_game(self, robot_first=False):
        with self.lock:                          # 手臂在動時不要量、不要改對齊（會跟搬運搶同一個場景）
            if self.phase in ("robot", "clearing", "handover", "selfplay", "fixing") or self.queue:
                return "我正在動，等我一下。"
            self.stop_flag = False               # 上次按過「停止」：開新局就清掉（不然收棋盤的工作會被丟掉，卡在「收棋盤中」）
            self.turn_hint = None
        ok, why = self.calib_check()
        if not ok:
            return why
        b = None
        for _ in range(3):                       # 相機子程序重啟或單幀偵測失敗時重試
            b = read_board()
            if b is not None:
                break
            time.sleep(1.0)
        if b is None:
            return "相機看不到棋盤，請檢查 Gemini 2。"
        if not any(c != " " for c in b):         # 棋盤上還有棋子時先不數料區（棋子都在棋盤上，數了一定不夠）；收完再開局時會再檢查
            warn = self.measure_stacks()
            if warn:
                return "料區有問題：%s。請確認後再開局。" % warn
            if self.flip and self.flip.last is not None:
                self.flip.forget_missing(self.flip.last[0])    # 舊紀錄不刪，人下在那附近會被當成那一種
        if any(c != " " for c in b):                     # 鐵律：不靠人手 → 機器自己收棋盤，收完再開局
            with self.lock:
                if self.phase in ("robot", "clearing", "handover"):
                    return "我正在動，等我一下。"
                self.phase, self.status = "clearing", "收棋盤中…"
            self.queue.append(("clear_then_new", robot_first))
            return "棋盤上還有棋子，我先收乾淨再開局。"
        with self.lock:
            self.board = [" "] * 9
            self.result, self.win_line, self.last_move = None, None, None
            self.stop_flag = False
            set_roles("X")                              # 每盤先回到預設（人叉、機器圈）；人第一手放圈會自動對調
            self.first = ROBOT if robot_first else HUMAN
            self.phase = "robot" if robot_first else "human"
            self.next_poll = time.time() + POLL_HUMAN
            self.status = "機器先下。" if robot_first else _turn_text()
        if robot_first:
            self.queue.append(("robot_move",))
        return "新局開始！我下圈，你下叉。%s" % ("我先下。" if robot_first else "你先請。" if self.flip else "你先請。想用圈也可以，直接下，我就改拿叉。")

    def stop(self):
        self.loop_play = False
        with self.lock:
            self.stop_flag = True
            if self.phase in ("clearing", "selfplay"):    # 收棋盤／機器對下中按停：回到閒置
                self.phase, self.status = "idle", "已停止收棋盤。"
            elif self.phase not in ("idle", "over"):      # 沒在下棋時只停手臂，不進入暫停
                self.phase = "paused"
                self.status = "已停止。說「繼續」或按「繼續」恢復。"
        if self.arm:
            self.arm.stop()
        return "好，我停下來了。"

    def reset(self):
        """重置（網頁「重置」鍵，使用者 2026-11-16）：中斷這盤。不像「停止」那樣馬上切電（吸著的棋子會掉在半路、收不回來），
        而是手上那一步放好就收手（機器對下的迴圈、主迴圈都會看 reset_req），再收好棋盤、盤面和訊息清空。
        手臂已經停住（停止、報警）就直接收。"""
        self.loop_play = False
        with self.lock:
            self.reset_req = True
            self.status = "重置中：手上這一步做完就收棋盤…" if self.phase in ("robot", "clearing", "selfplay", "fixing") else "重置中…"
        return "好，重置：這一步做完就收好棋盤，回到開局前。"

    def _do_reset(self):
        with self.lock:
            self.reset_req = False
            self.stop_flag = False
            self.loop_play = False
            self.queue.clear()
            self.board = [" "] * 9
            self.result, self.win_line, self.last_move = None, None, None
            self.turn_hint, self.prev_phase = None, None
            self._odd_since, self._restore_failed = None, None
            set_roles("X")
            self.messages = []
            self.reset_n += 1
            self.phase, self.status = "clearing", "重置：收棋盤中…"
        try:
            ok, why = self._clear_board()
        except Exception as e:
            ok, why = False, str(e)[:80]
        with self.lock:
            self.phase = "idle"
            self.status = "按「新局」開始。" if ok else "重置了，但收棋盤沒成功：%s" % why
        self.say("重置好了，棋盤收好了。" if ok else "重置了，可是%s" % why)

    def resume(self):
        with self.lock:
            if self.phase != "paused":
                return "目前沒有暫停喔。"
            self.stop_flag = False
            if self.prev_phase:                           # 交接中被停止：回到交接前的狀態
                self.phase, self.prev_phase = self.prev_phase, None
                self.next_poll = time.time() + POLL_HUMAN
                self.status = {"human": _turn_text(), "over": "這盤結束了。"}.get(self.phase, "按「新局」開始。")
                return "繼續！"
            if self.result:                               # 已結束的局不恢復
                self.phase = "over"
                return "這盤已經結束了，要再來一盤嗎？"
            self.phase = "robot" if self._robot_turn() else "human"
            self.next_poll = time.time() + POLL_HUMAN
            self.status = _turn_text() if self.phase == "human" else "換我下。"
        if self.phase == "robot":
            self.queue.append(("robot_move",))
        return "繼續！"

    def _check_end(self):
        w = winner(self.board)
        if w or " " not in self.board:
            self.phase = "over"
            self.result = w or "draw"
            self.win_line = next((list(l) for l in LINES if w and all(self.board[i] == w for i in l)), None)
            self.status = {HUMAN: "你贏了！", ROBOT: "我贏了！"}.get(w, "和局！")
            return True
        return False

    def _human_moved(self, cell, by_robot=False):
        with self.lock:
            self.board[cell] = HUMAN
            self.last_move = cell
            ended = self._check_end()
        if ended:
            self.say({HUMAN: "你連成一線了，你贏了！好厲害。", None: "和局，平手！再來一盤嗎？"}.get(self.result if self.result != "draw" else None, "結束了。"))
            self._queue_clear_end()
            return
        with self.lock:
            self.phase = "robot"
            self.status = "換我想一下…"
            self.turn_hint = None                       # 上一手的「換你」提示到這裡結束
        self.queue.append(("robot_move",))

    def _robot_move(self):
        if self.mover:
            return self._robot_move_mv()
        with self.lock:
            free = {i for i in self.reach if self.board[i] == " "}
            board = list(self.board)
        if not self.arm:
            self.say("我的手臂沒有連上，沒辦法下子。%s" % (self.arm_err or ""))
            with self.lock:
                self.phase = "paused"
            return
        if self.counts.get("O疊", 0) < 1:
            self.say("O 疊沒有棋子了，請補充後說「繼續」。")
            with self.lock:
                self.phase, self.status = "paused", "O 疊用完。"
            return
        if not free:
            self.say("剩下的空格我的手臂都搆不到，這盤只能下到這裡。")
            with self.lock:
                self.phase, self.result, self.status = "over", "stuck", "機器沒有可下的格。"
            return
        cell, score = best_move(board, ROBOT, allowed=free)
        full, fscore = best_move(board, ROBOT)
        line = "我下%s。" % NAMES[cell]
        if score == 1:
            line = "我下%s，這步之後我就贏定了！" % NAMES[cell]
        elif full != cell and fscore is not None and score is not None and fscore > score:
            line = "我很想下%s，可是手臂搆不到，只好下%s。" % (NAMES[full], NAMES[cell])
        self.say(line)
        with self.lock:
            self.status = "機器正在下%s…" % NAMES[cell]
        placed = False
        for attempt in range(3):                         # 沒吸起來（放完格子還是空的）就多壓 1.5 mm 再吸，不靠人
            try:
                self.arm.place("O疊", self.counts["O疊"], self.reach[cell], stop_flag=lambda: self.stop_flag, cell=cell,
                               extra_press=1.5 * attempt)
                if DRY and not (DRY_FAIL_PICKS and attempt < DRY_FAIL_PICKS):
                    VBOARD[cell] = ROBOT
            except Exception as e:
                if not self.stop_flag:                   # 使用者按停止造成的中斷，stop() 已經回過話
                    self.say(("%s。" % str(e)[:140]) if "手臂" in str(e) else ("下子時出了問題：%s。我先停下來。" % str(e)[:80]))
                    self.stop()
                return
            time.sleep(1.0)
            seen = None
            for _ in range(3):                           # 讀不到（被擋住）就再讀
                seen = read_board()
                if seen is not None:
                    break
                time.sleep(1.0)
            if seen is None or seen[cell] == ROBOT:
                placed = True
                break
            print("[下子] 第 %d 次：格 %d 看不到棋子 → 判斷沒從 O 疊吸起來（多壓 %.1f mm 再試）"
                  % (attempt + 1, cell, 1.5 * (attempt + 1)), flush=True)
            if attempt < 2:
                self.say("剛剛好像沒吸到棋子，我再試一次。")
        if not placed:
            self.say("我從 O 疊吸不到棋子，先停下來。O 疊的高度可能跟我量的不一樣。")
            self.stop()
            return
        self.counts["O疊"] -= 1
        self._save_counts()
        with self.lock:
            self.board[cell] = ROBOT
            self.last_move = cell
            ended = self._check_end()
            if not ended:
                self.phase = "human"
                self.next_poll = time.time() + 0.3
                self.status = _turn_text()
        if ended:
            self.say({ROBOT: "我連成一線，我贏了！要不要再來一盤？"}.get(self.result, "和局，平手！再來一盤嗎？"))

    def _clear_board(self):
        """把棋盤上每顆棋子收回料區（X→X 疊、O→O 疊）。每顆收完看相機確認，沒吸起就多壓 1.5 mm 再試（最多 3 次）。"""
        if self.mover:
            return self._clear_board_mv()
        if not self.arm:
            return False, "手臂沒連上，沒辦法收棋盤。"
        cal = self.arm.cal
        for _ in range(12):                               # 最多 9 顆＋重試
            if self.stop_flag:
                return False, "已停止。"
            if DRY:
                occ = [(i, VBOARD[i], (300.0, 0.0)) for i in range(9) if VBOARD[i] != " "]
            else:
                img = latest_frame()
                if img is None:
                    return False, "相機沒有影像，沒辦法收棋盤。"
                from world import px_to_arm
                occ = [(p["cell"], p["kind"], tuple(px_to_arm(cal, p["px"], depth=find_pieces.PIECE_TOP)))
                       for p in find_pieces.pieces(img, self.plan)]
            if not occ:
                return True, None
            cell, kind, center = occ[0]
            stack = kind + "疊"
            with self.lock:
                self.status = "收格 %d 的 %s…" % (cell, kind)
            done = False
            for attempt in range(3):
                try:
                    why = self.arm.clear_piece(center, cell, stack, self.counts.get(stack, 0),
                                               stop_flag=lambda: self.stop_flag, extra_press=1.5 * attempt)
                except Exception as e:
                    self.stop()
                    return False, "收棋盤時出了問題：%s" % str(e)[:80]
                if why:
                    return False, why + "。"
                if DRY:
                    VBOARD[cell] = " "
                time.sleep(1.0)
                b = read_board()
                if b is not None and b[cell] == " ":
                    self.counts[stack] = self.counts.get(stack, 0) + 1
                    self._save_counts()
                    print("[收棋盤] 格 %d 的 %s → %s（第 %d 次）" % (cell, kind, stack, attempt + 1), flush=True)
                    done = True
                    break
                print("[收棋盤] 格 %d 沒吸起來，多壓 1.5 mm 再試" % cell, flush=True)
            if not done:
                return False, "格 %d 的棋子我吸了三次都吸不起來，先停下來。" % cell
        return False, "收太多次了，棋盤還是沒清空。"

    # ---------- 機器對下（自我測試：O＝高手、X＝普通，兩邊都由 B 從各自的料區拿） ----------
    def self_play(self):
        with self.lock:
            if self.phase not in ("idle", "over"):
                return "現在不方便，等這盤結束再說。"
            if not self.arm:
                return "手臂沒連上。"
            self.phase, self.status = "selfplay", "機器對下準備中…"
            self.stop_flag = False
        self.queue.append(("selfplay",))
        return "好，我自己跟自己下一盤：圈是高手、叉是普通程度。"

    def _x_bar_info(self):
        """X 疊最上面那顆：(叉的角度°, 棋子中心相對疊中心的偏移 mm（B 座標）)。
        角度用綠叉邊緣的梯度方向（mod 90）統計，不依賴中心；中心用「綠＋黑」圓盤的距離轉換峰值。"""
        import robot
        img = latest_frame()
        if img is None:
            return None, None
        u, v = [int(t) for t in robot.STOCK["stocks"]["X疊"]["px"]]
        R = 75
        crop = img[v - R:v + R + 1, u - R:u + R + 1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (40, 50, 50), (95, 255, 255))
        dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 100))
        disc = cv2.morphologyEx(cv2.bitwise_or(green, dark), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        dt = cv2.distanceTransform(disc, cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
        g = cv2.GaussianBlur(green.astype(np.float32), (5, 5), 0)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        ang = (np.degrees(np.arctan2(gy, gx)) % 90.0)[mag > mag.max() * 0.3]
        if len(ang) < 50:
            return None, None
        hist = np.bincount(np.round(ang).astype(int) % 90, minlength=90).astype(float)
        hist = np.convolve(np.concatenate([hist[-3:], hist, hist[:3]]), np.ones(5) / 5, "same")[3:-3]
        deg = float(np.argmax(hist))                     # 叉的邊緣法線方向 ≡ 叉的方向（mod 90）
        from world import px_to_arm
        ctr = px_to_arm(self.arm.cal, (u - R + cx, v - R + cy), depth=535.0)
        shift = ctr - np.array(robot.STOCK["stocks"]["X疊"]["xy"])
        if np.hypot(*shift) > 15:                         # 偏太多多半是偵測錯，不採用
            shift = np.zeros(2)
        return deg, shift

    def _x_bar_deg(self):
        return self._x_bar_info()[0]

    def _wait_free(self, kind):
        """kind 現在可以下的格。沒有（料疊被手擋住、支架或格線被擋住量不到、搆不到）就講清楚原因、每 4 秒重看，
        最多約 3 分鐘，不要直接結束這盤。回傳 (格子, 最後的原因)。"""
        import carrier
        why = None
        for tries in range(1, 47):
            if self.stop_flag:
                return [], "已停止"
            try:
                free = self._free_for(kind)
                if free:
                    return free, None
                seen = [x for x in getattr(self, "last_stocks", []) if x["kind"] == kind]
                why = ("%s的料疊在固定位置上，可是空的格我都放不過去" % MARK[kind]) if seen \
                    else ("我看不到%s的料疊，手先不要擋在上面" % MARK[kind])
            except carrier.Stopped:
                return [], "已停止"
            except RuntimeError as e:                    # 支架、格線被擋住量不到等：講原因、重看
                why = "我看不清楚場地（%s），手先不要擋著" % str(e)[:50]
            print("[下子] 第 %d 次沒有可下的格：%s（料區：%s）" % (
                tries, why, [(x["kind"], x["n"], tuple(int(t) for t in x["px"])) for x in getattr(self, "last_stocks", [])]), flush=True)
            if tries == 46:
                break
            with self.lock:
                self.status = why + "…"
            if tries in (1, 12, 30):
                self.say(why + "。")
            time.sleep(4)
        return [], why

    def _robot_move_mv(self):
        with self.lock:
            board = list(self.board)
            self.turn_hint = None
        free, why = self._wait_free(ROBOT)
        if not free:
            if self.stop_flag:
                return
            self.say("等了很久還是拿不到%s，這盤只能下到這裡（%s）。" % (MARK[ROBOT], why))
            with self.lock:
                self.phase, self.result, self.status = "over", "stuck", "機器沒有可下的格：%s" % why
            return
        cell, score = best_move(board, ROBOT, allowed=set(free))
        full, fscore = best_move(board, ROBOT)
        line = "我下%s。" % NAMES[cell]
        if score == 1:
            line = "我下%s，這步之後我就贏定了！" % NAMES[cell]
        elif full != cell and fscore is not None and score is not None and fscore > score:
            line = "我很想下%s，可是手臂搆不到，只好下%s。" % (NAMES[full], NAMES[cell])
        self.say(line)
        with self.lock:
            self.status = "機器正在下%s…" % NAMES[cell]
            self.turn_hint = None

        def _released():                             # 棋子放開的瞬間：先讓畫面與聲音提示「換你」，手臂同時收回
            with self.lock:
                self.turn_hint = "human"
                self.status = ("我下好了，換你下%s！" % MARK[HUMAN])

        self._release_cb = _released
        try:
            ok, why = self._move_piece(ROBOT, cell)
        except Exception as e:
            self._release_cb = None
            if not self.stop_flag:
                self.say(("%s。" % str(e)[:140]) if "手臂" in str(e) else ("下子時出了問題：%s。我先停下來。" % str(e)[:80]))
                self.stop()
            return
        self._release_cb = None                         # 只給這一手用；收棋盤、代放子不要觸發「換你」
        if not ok:
            with self.lock:
                self.turn_hint = None
            self.say(why + "，我先停下來。")
            self.stop()
            return
        with self.lock:
            self.board[cell] = ROBOT
            self.last_move = cell
            ended = self._check_end()
            if not ended:
                self.phase = "human"
                self.next_poll = time.time() + 0.3
                self.status = _turn_text()
        if ended:
            self.say({ROBOT: "我連成一線，我贏了！要不要再來一盤？"}.get(self.result, "和局，平手！再來一盤嗎？"))
            self._queue_clear_end()

    def _board_odd(self, cur, nb):
        """人的回合，盤面跟紀錄對不上（一次放兩顆、拿走或挪動棋子）。雙臂：持續 10 秒、連續 3 次讀到一樣（手離開了）
        就自己恢復（carrier.restore）；恢復不了的同一個盤面不重試。單臂（舊路徑）：6 秒後講哪裡不一樣，每 30 秒提醒。"""
        now = time.time()
        if getattr(self, "_odd_since", None) is None or now - self._odd_last > 2 * POLL_HUMAN + 3:   # 上一回合留下的不算
            self._odd_since, self._odd_said, self._odd_prev, self._odd_same = now, 0.0, None, 0
        self._odd_last = now
        self._odd_same = self._odd_same + 1 if nb == self._odd_prev else 0
        self._odd_prev = list(nb)
        if now == self._odd_since:
            return
        # 雙臂：對不上超過 10 秒、連續 3 次讀到一樣（手已經離開）→ 自己恢復（不靠人手）
        if self.mover:
            if now - self._odd_since < 10 or self._odd_same < 3 or nb == getattr(self, "_restore_failed", None):
                return                                   # 上次恢復不了的同一個盤面：不要每 10 秒重試、重講
            with self.lock:
                if self.phase != "human":
                    return
                self.phase, self.status = "fixing", "有棋子被移動了，我把它放回去…"
            self._odd_since = None
            self.say("有棋子被移動了，我把它放回去。")
            self.queue.append(("restore",))
            return
        if now - self._odd_since < 6 or now - self._odd_said < 30:
            return
        diff = []
        for i in range(9):
            if cur[i] == nb[i]:
                continue
            if cur[i] == " ":
                diff.append("%s多了一顆" % NAMES[i])
            elif nb[i] == " ":
                diff.append("%s的%s不見了" % (NAMES[i], MARK.get(cur[i], "棋子")))
            else:
                diff.append("%s應該是%s" % (NAMES[i], MARK.get(cur[i], "棋子")))
        if not diff:
            return
        self._odd_said = now
        new = sum(cur[i] == " " and nb[i] != " " for i in range(9))
        tip = "一次只下一顆，多的請拿回去" if new > 1 else "棋子請放在格子正中間，被移動的請放回原位"
        self.say("盤面跟我記的不一樣：%s。%s。" % ("、".join(diff), tip))
        with self.lock:
            self.status = "盤面跟我記的不一樣：%s" % "、".join(diff)

    def _restore(self):
        """人的回合盤面被弄亂：照記的盤面恢復（carrier.restore），之後回到人的回合。"""
        import carrier
        with self.lock:
            board = list(self.board)
        try:
            carrier.begin(self.mover)
            try:
                ok, why = carrier.restore(self.flip, board)
            finally:
                self.mover.park_all()
        except Exception as e:
            ok, why = False, "已停止" if self.stop_flag else "恢復時出了問題：%s" % str(e)[:80]
        with self.lock:
            if self.phase == "fixing":
                self.phase = "human"
                self.next_poll = time.time() + POLL_HUMAN
                self.status = _turn_text() if ok else "盤面跟我記的不一樣：%s" % why
        self._odd_since = None
        nb = read_board(n=2, gap=0.2)
        self._restore_failed = None if ok else nb      # 恢復不了：同一個盤面不再試，盤面變了（人動過）才再試
        self.say("放回去了，換你下。" if ok else why + "。")

    def _queue_clear_end(self):
        """人機對戰結束：雙臂模式自己收棋盤（使用者 2026-11-15）。"""
        if self.mover:
            self.queue.append(("clear_end",))

    def _clear_end(self):
        """等 8 秒（讓人看結果、把手移開），這盤還是「結束」狀態（沒按新局、停止）才收。"""
        t0 = time.time()
        while time.time() - t0 < 8:
            if self.stop_flag or self.phase != "over":
                return
            time.sleep(0.25)
        with self.lock:
            if self.phase != "over" or self.stop_flag:
                return
            self.phase, self.status = "clearing", "這盤結束了，收棋盤中…"
        ok, why = self._clear_board()
        with self.lock:
            if self.phase == "clearing":
                self.phase = "over"
            self.status = "棋盤收好了，按「新局」再來一盤。" if ok else "收棋盤沒成功：%s" % why
        self.say("棋盤收好了，要再來一盤嗎？" if ok else why)

    def _place_verified(self, stack, cell, mark):
        if self.mover:
            ok, why = self._move_piece(mark, cell)
            self.last_fail = None if ok else why
            if not ok:
                print("[下子] %s" % why, flush=True)
            return ok
        """從 stack 拿一顆放到 cell，相機確認；沒吸起就多壓 1.5 mm 再試（最多 3 次）。回傳是否成功。
        X 疊：棋子沒貼膠帶，偏移吸取只能吸正中央或叉之間黑區中段（每次看最上面那顆的叉角度再算）。"""
        target, offset, shift = self.reach[cell], None, None
        if stack == "X疊" and hasattr(self.arm, "x_target"):
            deg, shift = self._x_bar_info()
            t = self.arm.x_target(self.plan, cell, deg if deg is not None else 0.0)
            if t is None:
                print("[下子] X 疊 → 格 %d：叉角度 %s，正中央與黑區都送不到" % (cell, deg), flush=True)
                return False
            target, offset = t
            print("[下子] X 疊叉角度 %s°、棋子偏 %s mm，格 %d 吸取偏移 %s mm"
                  % (deg, None if shift is None else np.round(shift, 1), cell, np.round(offset, 1)), flush=True)
        for attempt in range(3):
            self.arm.place(stack, self.counts[stack], target, stop_flag=lambda: self.stop_flag, cell=cell,
                           extra_press=1.5 * attempt, offset=offset, piece_shift=shift)
            if DRY and not (DRY_FAIL_PICKS and attempt < DRY_FAIL_PICKS):
                VBOARD[cell] = mark
            time.sleep(1.0)
            seen = None
            for _ in range(3):
                seen = read_board()
                if seen is not None:
                    break
                time.sleep(1.0)
            if seen is None or seen[cell] == mark:
                self.counts[stack] -= 1
                self._save_counts()
                return True
            print("[下子] %s 第 %d 次：格 %d 看不到棋子 → 多壓 1.5 mm 再試" % (stack, attempt + 1, cell), flush=True)
        return False

    def _selfplay(self):
        import random
        ok_game = False
        try:
            if self.flip:
                self.flip.reset()
            b = read_board()
            if b is None:
                self.say("相機看不到棋盤。")
                return
            n_o, n_x = b.count(ROBOT), b.count(HUMAN)
            resume = any(c != " " for c in b) and n_o - n_x in (0, 1) and not winner(b) and "?" not in b
            if any(c != " " for c in b) and not resume:
                ok, why = self._clear_board()
                if not ok:
                    self.say(why)
                    return
                b = [" "] * 9
                n_o = n_x = 0
            warn = self.measure_stacks(("O", "X"))
            if warn and not resume:
                self.say("料區有問題：%s" % warn)
                return
            with self.lock:
                self.board = list(b)                         # 棋盤上已有合法局面就接著下（圈先：圈＝叉 → 輪圈）
                self.result, self.win_line, self.last_move = None, None, None
            turn = ROBOT if n_o == n_x else HUMAN
            t0 = time.time()
            early_draw, stuck = False, None
            while not self.stop_flag and not self.reset_req:      # 重置：這一步放好就跳出，由 _do_reset 收棋盤
                if winner(self.board) or " " not in self.board:
                    break
                if all("O" in [self.board[i] for i in l] and "X" in [self.board[i] for i in l] for l in LINES):
                    early_draw = True                        # 每一條線都同時有圈和叉：誰都贏不了，不用下完（使用者 2026-11-15）
                    break
                free, stuck = self._wait_free(turn) if self.mover else (self._free_for(turn), "沒有可下的格")
                if not free:
                    break
                if turn == ROBOT:
                    cell, _ = best_move(list(self.board), ROBOT, allowed=set(free))
                else:                                        # 普通：能贏就贏、該擋就擋，其他隨機
                    if hasattr(self.arm, "x_target"):         # X 只能下在「叉角度允許的偏移」送得到的格
                        deg = self._x_bar_deg()
                        free = [i for i in free if self.arm.x_target(self.plan, i, deg or 0.0) is not None]
                        if not free:
                            break
                    cell = None
                    for me in (HUMAN, ROBOT):
                        for i in free:
                            bb = list(self.board)
                            bb[i] = me
                            if winner(bb) == me:
                                cell = i
                                break
                        if cell is not None:
                            break
                    if cell is None:
                        cell = random.choice(free)
                stack = turn + "疊"
                if not self.mover and self.counts.get(stack, 0) < 1:
                    self.say("%s 疊沒有棋子了。" % turn)
                    return
                with self.lock:
                    self.status = "機器對下：%s 下%s…" % (turn, NAMES[cell])
                self.say("%s下%s。" % ("圈" if turn == ROBOT else "叉", NAMES[cell]))
                if not self._place_verified(stack, cell, turn):
                    self.say("%s，先停下來。" % (getattr(self, "last_fail", None) or "%s 疊我吸了三次都吸不起來" % turn))
                    return
                with self.lock:
                    self.board[cell] = turn
                    self.last_move = cell
                turn = ROBOT if turn == HUMAN else HUMAN
            with self.lock:
                self._check_end()
                if self.result is None and early_draw:
                    self.result, self.phase = "draw", "over"
                if self.result is None:
                    self.result, self.phase = "stuck", "over"
                self.status = "機器對下結束（%.0f 秒）" % (time.time() - t0)
            w = winner(self.board)
            ok_game = self.result in ("O", "X", "draw")
            if self.result == "stuck":
                if not self.stop_flag and not self.reset_req:
                    self.say("這盤下不下去了：%s。" % (stuck or "沒有可下的格"))
            else:
                self.say({"O": "圈贏了！", "X": "叉贏了！"}.get(w, "已經分不出勝負，和局！" if early_draw else "和局！") + "對下結束。")
            print("[機器對下] 結果 %s，盤面 %s，%.0f 秒" % (w or "和局", "".join(c if c != " " else "." for c in self.board),
                                                     time.time() - t0), flush=True)
            if self.mover and ok_game and not self.stop_flag and not self.reset_req:      # 下完自己收：X 疊回 X 疊、O 疊回 O 疊（使用者 2026-11-15）
                time.sleep(3)
                with self.lock:
                    self.phase, self.status = "clearing", "對下結束，收棋盤中…"
                ok, why = self._clear_board()
                with self.lock:
                    if self.phase == "clearing":
                        self.phase = "over"
                    self.status = "機器對下結束，棋盤收好了。" if ok else "收棋盤沒成功：%s" % why
                self.say("棋盤收好了。" if ok else why)
        except Exception as e:
            if not self.stop_flag:
                self.say("機器對下時出了問題：%s。我先停下來。" % str(e)[:80])
                self.stop()
        finally:
            with self.lock:
                if self.phase == "selfplay":
                    self.phase = "over"
            if getattr(self, "loop_play", False) and not self.stop_flag:
                self.loop_fails = 0 if ok_game else getattr(self, "loop_fails", 0) + 1
                if self.loop_fails >= 2:
                    self.loop_play = False
                    self.say("連續兩盤都出問題，我先停下來等人看一下。")
                else:
                    def _again():
                        time.sleep(8)
                        if getattr(self, "loop_play", False) and not self.stop_flag:
                            self.say("再來一盤！")
                            self.self_play()
                    threading.Thread(target=_again, daemon=True).start()

    # ---------- 雙臂＋自動料區 ----------
    def _stocks(self):
        """相機偶爾會斷流（子程序會自己重開相機，約 30～40 秒）：沒影像就等一下重試，不要整盤停掉。"""
        for k in range(5):
            try:
                return self._stocks_once()
            except RuntimeError as e:
                if k == 4 or self.stop_flag or "相機沒有影像" not in str(e):
                    raise
                print("[料區] 相機沒有影像，15 秒後重試（第 %d 次）" % (k + 1), flush=True)
                time.sleep(15)

    def _stocks_once(self):
        """當下的料區（carrier.Scene.measure：彩色＋深度；種類照 ox_record.json 的固定位置）。"""
        _, piles = self.flip.measure()
        S = [x for x in piles if x["kind"] in ("O", "X")]
        self.last_stocks = S
        return S

    def _reconnect_arms(self):
        """手臂被重開電源／網路斷過：丟掉舊連線重連（不然每個查詢都失敗，會被誤判成搆不到）。"""
        import mover
        old = self.mover
        try:
            old.close()
        except Exception:
            pass
        self.mover = mover.Mover()
        self.arm = _MoverAdapter(self.mover)
        self.flip.attach(self.mover)
        print("[手臂] 連線斷了，已重新連線", flush=True)

    def _free_for(self, kind):
        try:
            return self._free_for_once(kind)
        except (ConnectionError, TimeoutError) as e:      # 只有連線問題才重連（寫檔失敗之類的 OSError 不要把兩台關掉重連）
            print("[手臂] %s → 重新連線" % str(e)[:60], flush=True)
            self._reconnect_arms()
            return self._free_for_once(kind)

    def _free_for_once(self, kind):
        free = [i for i in range(9) if self.board[i] == " "]
        if not self.mover:
            return [i for i in free if i in self.reach]
        import carrier
        self._stocks()
        seen = self.flip.read_board(self.flip.last[0])                 # 人搶先放了（還沒輪到他）的格也不能選：等這手下完會被讀成他的那一手
        free = [i for i in free if seen is None or seen[i] == " "]
        ok = [i for i in free if carrier.can_place(self.flip, kind, i, self.flip.last[1])]
        with self.lock:
            self.reach = {i: None for i in ok} if kind == ROBOT else self.reach
        return ok

    def _move_piece(self, kind, cell):
        """從 kind 的料疊拿一顆放到 cell（carrier.place：負責的手臂搆得到就直接放，搆不到經轉送點給另一台；放完量、偏了挪正）。"""
        import carrier
        carrier.begin(self.mover)
        try:
            r = carrier.place(self.flip, kind, cell, on_release=getattr(self, "_release_cb", None))
        finally:
            self.mover.park_all()
        print("[搬] %s → 格 %d：%s" % (kind, cell, {k: v for k, v in r.items() if k != "kind"}), flush=True)
        return (True, None) if r["ok"] else (False, r.get("why") or "沒放成功")

    def _clear_board_mv(self):
        """收棋盤（雙臂）：照紀錄 X 收回 X 疊（A）、O 收回 O 疊（B）；負責的手臂搆不到就經轉送點。"""
        import carrier
        with self.lock:
            self.status = "收棋盤中…"
        try:
            carrier.begin(self.mover)
            return carrier.clear(self.flip)
        except Exception as e:
            return False, "已停止。" if self.stop_flag else "收棋盤時出了問題：%s" % str(e)[:80]
        finally:
            self.mover.park_all()

    def place_for_human(self, cell):
        with self.lock:
            if self.phase != "human":
                return "現在不是你的回合喔。"
            if self.board[cell] != " ":
                return "%s已經有棋子了。" % NAMES[cell]
        if not self.mover and (cell not in self.reach or (hasattr(self.arm, "x_cells") and cell not in self.arm.x_cells)):
            return "%s我的手臂從 X 疊送不過去，請你自己把 X 放上去。" % NAMES[cell]
        if not self.mover and self.counts.get("X疊", 0) < 1:
            return "X 疊沒有棋子了，請你自己放。"
        with self.lock:
            self.phase = "robot"
            self.status = "幫你把 X 放到%s…" % NAMES[cell]
        self.queue.append(("place_x", cell))
        return "好，我幫你把 X 放在%s。" % NAMES[cell]

    def _place_x(self, cell):
        if self.mover:
            try:
                ok, why = self._move_piece(HUMAN, cell)
            except Exception as e:
                ok, why = False, str(e)[:80]
            if not ok:
                if not self.stop_flag:
                    self.say("幫你放%s時出了問題：%s。我先停下來。" % (MARK[HUMAN], why))
                    self.stop()
                return
            self._human_moved(cell, by_robot=True)
            return
        try:
            self.arm.place("X疊", self.counts["X疊"], self.reach[cell], stop_flag=lambda: self.stop_flag, cell=cell)
            if DRY:
                VBOARD[cell] = HUMAN
        except Exception as e:
            if not self.stop_flag:
                self.say("放 X 時出了問題：%s。我先停下來。" % str(e)[:80])
                self.stop()
            return
        self.counts["X疊"] -= 1
        self._save_counts()
        self._human_moved(cell, by_robot=True)

    # ---------- 花式交接：把棋子放到手掌上 ----------
    def hand_over(self, mark=HUMAN):
        mark = ROBOT if str(mark).upper() == "O" else HUMAN
        stack = mark + "疊"
        with self.lock:
            if self.phase not in ("human", "idle", "over"):
                return "現在不方便，等輪到你或這盤結束再說喔。"
            if not self.arm:
                return "我的手臂沒有連上，沒辦法拿給你。"
            if self.counts.get(stack, 0) < 1:
                return "%s 疊沒有棋子了。" % mark
            self.prev_phase = self.phase
            self.phase = "handover"
            self.status = "準備把 %s 放到你的手上…" % mark
        self.queue.append(("handover", stack))
        return "好，我拿一顆 %s 給你，等我準備一下。" % mark

    def _get_palm(self, timeout, ref):
        """等一個穩定（連兩張深度影像同位置）而且可放的手掌。回傳 (palm, t_seen)：t_seen＝這次第一次看到可放的手的時間
        （算反應時間用）；逾時回 None。每來一張新的深度影像就判斷一次（約 10 張／秒）。"""
        if DRY:
            time.sleep(1.0)
            t = time.time()
            return dict(xy=(330.0, 0.0), h=60.0, px=(640, 400), r_px=60, depth=480.0, half_mm=40.0), t
        t0, prev, t_seen, last_reason, said, last_t = time.time(), None, None, None, 0.0, 0.0
        self.t_hand_any = None                                 # 第一次看到像手的東西（含位置不對被拒絕的）
        while time.time() - t0 < timeout and not self.stop_flag:
            try:
                mt = os.path.getmtime(CAM_DEPTH_FILE)
            except OSError:
                mt = 0.0
            if mt == last_t:
                time.sleep(0.01)
                continue
            last_t = mt
            d = latest_depth()
            if d is None:
                continue
            palm, reason = hand.detect(d, ref, self.arm.cal, exclude=self._arm_footprint(d.shape))
            if palm is not None and reason is None:
                reason = self.arm.handover_plan(palm)[1]
            with self.lock:
                self.palm_vis = (palm, reason)
                self.status = reason or "看到你的手了"
            if palm is not None and "不像手掌" not in (reason or "") and self.t_hand_any is None:
                self.t_hand_any = time.time()
            if reason is not None and palm is not None and "不像手掌" not in reason:
                print("[交接] 手不行：B(%.0f, %.0f) 離 B %.0f mm、離棋盤 %.0f mm → %s"
                      % (palm["xy"][0], palm["xy"][1], np.hypot(*palm["xy"]), palm["h"], reason), flush=True)
            if reason is None:
                if t_seen is None:
                    t_seen = time.time()
                if hand.same_place(palm, prev):
                    print("[交接] 手掌 B(%.0f, %.0f) 離 B %.0f mm、離棋盤 %.0f mm、半寬 %.0f mm（確認花 %.2f s）"
                          % (palm["xy"][0], palm["xy"][1], np.hypot(*palm["xy"]), palm["h"], palm["half_mm"],
                             time.time() - t_seen), flush=True)
                    return palm, t_seen
                prev = palm
            else:
                prev, t_seen = None, None
                if palm is not None and "不像手掌" not in reason and reason != last_reason and time.time() - said > 3:
                    self.say(reason)                        # 手太高／太低／太近：提醒一次
                    last_reason, said = reason, time.time()
        return None

    def _arm_footprint(self, shape):
        """B 手臂目前在影像上占的區域（連桿＋末端方盒），偵測手掌時排除。"""
        from plan_cells import face_px
        m = np.zeros(shape, np.uint8)
        try:
            p = self.arm.arm.pose()
            base, tcp = face_px(self.arm.cal, (0.0, 0.0), p[2]), face_px(self.arm.cal, p[:2], p[2])
            cv2.line(m, (int(base[0]), int(base[1])), (int(tcp[0]), int(tcp[1])), 1, 200)
            cv2.circle(m, (int(tcp[0]), int(tcp[1])), 120, 1, -1)
        except Exception:
            pass
        return m

    def _handover(self, stack):
        """快速交接（使用者：手伸出去等不能超過 3 秒）：先吸好棋子在待命點等 → 才叫人伸手 → 看到手就直接過去放。"""
        mark = stack[0]
        stop = lambda: self.stop_flag                        # noqa: E731
        try:
            if not set_camera_depth(True):
                self.say("相機的深度沒有出來，這次先不交接。")
                return
            refs = [] if DRY else capture_depth(5)
            ref = hand.reference(refs) if refs else None
            if ref is None and not DRY:
                self.say("相機的深度沒有出來，這次先不交接。")
                return
            bg = None if DRY else cv2.imread(HAND_BG, cv2.IMREAD_UNCHANGED)
            if bg is not None and bg.shape == ref.shape:
                ref = np.maximum(ref, bg.astype(np.float32))  # 當下和背景取較遠的：手就算已經伸著也不會被當成背景
            elif not DRY:
                self.say("我還沒有沒放手時的背景，這次手如果已經伸著會看不到喔。")
            self.hand_ref = ref
            self.say("我先去拿一顆%s，馬上好。" % ("圈" if mark == "O" else "叉"))
            with self.lock:
                self.status = "去拿 %s…（手先不用伸）" % mark
            self.arm.hand_prepare(stack, self.counts[stack], stop_flag=stop)
            self.say("準備好了！把手掌張開，伸到棋盤中間上方。")
            with self.lock:
                self.status = "把手掌伸到棋盤中間上方"
            done = None
            for attempt in range(3):
                got = self._get_palm(30.0, ref)
                if got is None:
                    break
                palm, t_seen = got
                t_go = time.time()
                res = self.arm.hand_deliver(palm, guard=self.guard, stop_flag=stop)
                if res["ok"]:
                    done = t_go - t_seen + res["t_release"]
                    t_any = self.t_hand_any or t_seen
                    total = t_go - t_any + res["t_release"]
                    print("[交接] 完成：手伸進來 → 放開 %.2f s（調整位置 %.2f s＋確認 %.2f s＋移動 %.2f s）%s"
                          % (total, t_seen - t_any, t_go - t_seen, res["t_release"], "" if total <= 3.0 else "  ⚠ 超過 3 秒"),
                          flush=True)
                    break
                print("[交接] 第 %d 次沒放：%s" % (attempt + 1, res["why"]), flush=True)
                self.say("%s，再伸過來一次。" % res["why"])
                self.arm.hand_back_ready(stop_flag=stop)
            if done is not None:
                self.counts[stack] -= 1
                self._save_counts()
                self.say("放好囉！" + ("換你把它下到棋盤上。" if self.prev_phase == "human" else ""))
                self.arm.hand_finish()
            elif not self.stop_flag:
                self.say("這次先不給囉，我把棋子放回去。")
                self.arm.hand_put_back()
        except Exception as e:
            if not self.stop_flag:
                self.say("交接時出了問題：%s。我先停下來。" % str(e)[:80])
                self.stop()
        finally:
            set_camera_depth(False)
            with self.lock:
                self.palm_vis = None
                if self.phase == "handover":
                    self.phase, self.prev_phase = self.prev_phase or "idle", None
                    self.next_poll = time.time() + POLL_HUMAN
                    self.status = {"human": _turn_text(), "over": "這盤結束了。"}.get(self.phase, "按「新局」開始。")

    def save_hand_bg(self):
        """拍「沒有手」的背景深度存檔（手臂都要在停放點、棋盤上方淨空）。"""
        if DRY:
            return "（模擬）背景已存。"
        with self.lock:
            if self.phase not in ("idle", "human", "over"):
                return "手臂在動的時候不能拍背景。"
        try:
            if not set_camera_depth(True):
                return "相機的深度沒有出來。"
            refs = capture_depth(8)
            if len(refs) < 5:
                return "深度影像不夠，沒存成。"
            bg = hand.reference(refs)
            cv2.imwrite(HAND_BG, bg.clip(0, 65535).astype(np.uint16))
            return "背景存好了（棋盤中間深度 %.0f mm）。" % bg[360, 640]
        finally:
            set_camera_depth(False)

    def hint(self):
        with self.lock:
            b = list(self.board)
        if winner(b) or " " not in b:
            return "這盤已經結束了。"
        cell, score = best_move(b, HUMAN)
        tail = {1: "下了你就贏定了。", 0: "這樣可以守住和局。", -1: "不過局面對你不太妙。"}.get(score, "")
        return "建議你下%s，%s" % (NAMES[cell], tail)

    # ---------- 主迴圈 ----------
    def _loop(self):
        while True:
            try:
                if self.reset_req:                      # 網頁按了「重置」（上一個工作已經做完才會到這裡）
                    self._do_reset()
                    continue
                if self.queue:
                    job = self.queue.pop(0)
                    if not self.stop_flag:
                        if job[0] == "robot_move":
                            self._robot_move()
                        elif job[0] == "clear_then_new":
                            ok, why = self._clear_board()
                            with self.lock:
                                if self.phase == "clearing":
                                    self.phase = "idle"
                                    self.status = "按「新局」開始。"
                            self.say(self.new_game(robot_first=job[1]) if ok else why)
                        elif job[0] == "clear_end":
                            self._clear_end()
                        elif job[0] == "restore":
                            self._restore()
                        elif job[0] == "selfplay":
                            self._selfplay()
                        elif job[0] == "place_x":
                            self._place_x(job[1])
                        elif job[0] == "handover":
                            self._handover(job[1])
                    continue
                b = read_board(n=1)
                if b:
                    with self.lock:
                        self.seen = b
                if self.phase == "human" and time.time() >= self.next_poll:
                    self.next_poll = time.time() + POLL_HUMAN
                    nb = read_board(n=2, gap=0.2)
                    if nb:
                        with self.lock:
                            cur = list(self.board)
                        # 人第一手放的是機器那種棋子（預設是圈）：不要不理他，直接對調——人用圈、機器改下叉
                        if not self.flip and cur.count(" ") == 9 and nb.count(ROBOT) == 1 and nb.count(HUMAN) == 0:
                            nb_chk = read_board(n=2, gap=0.2)
                            if nb_chk == nb:
                                set_roles(ROBOT)
                                self.first = HUMAN          # 人先下的那一方，對調後是新的 HUMAN（不然暫停再繼續會輪錯人）
                                self.say("好，你用%s，那我下%s。" % (MARK[HUMAN], MARK[ROBOT]))
                        mine = (HUMAN, "?") if self.flip else (HUMAN,)       # 翻面的棋子看不到符號：新出現沒紀錄的就是人下的
                        new = [i for i in range(9) if cur[i] == " " and nb[i] in mine]
                        consistent = all(nb[i] == cur[i] for i in range(9) if cur[i] != " ")
                        if len(new) == 1 and consistent and sum(c in mine for c in nb) == cur.count(HUMAN) + 1:
                            nb2 = read_board(n=2, gap=0.2)                # 立即再確認一次
                            if nb2 == nb and (not self.flip or self.flip.mark_cell(latest_frame(), new[0], HUMAN)):
                                self.say("你下了%s。" % NAMES[new[0]], speak=False)
                                self._human_moved(new[0])
                            self._odd_since = None
                        elif nb != cur:
                            self._board_odd(cur, nb)
                        else:
                            self._odd_since = None
                time.sleep(0.25)                    # 網頁棋盤要即時顯示相機看到的盤面（使用者要求），所以讀盤頻率拉高
            except Exception as e:
                with self.lock:
                    self.status = "內部錯誤：%s" % str(e)[:80]
                time.sleep(2)


# ======================= 語音理解 =======================
SYSTEM = """你是井字棋（OOXX）對弈機器人「小鉑」，用兩隻 MG400 機械手臂和俯視相機跟使用者下棋。
你下 O，使用者下 X。格子：0左上 1上中 2右上 3左中 4中間 5右中 6左下 7下中 8右下。
兩隻手臂合作，九格都下得到（自己搆不到的會交給另一隻）。
使用者的話來自語音辨識，可能有錯字或同音字（例如「井字棋」被聽成「錦字奇」），請推測原意。
只輸出一個 JSON，不要其他文字：
{"intent": "...", "cell": 格號或null, "robot_first": true/false, "reply": "繁體中文口語回覆，40字內"}
intent 只能是：
- "place_x"：使用者要你幫他把 X 放到某格（例：幫我下右上角、我要下中間你幫我放）→ cell 必填
- "hint"：問該下哪裡、給建議
- "new_game"：開新局／再來一盤；使用者說你先下時 robot_first=true
- "stop"：停、暫停、危險、等一下
- "resume"：繼續
- "status"：問現在局面、輪到誰、誰贏
- "chat"：其他閒聊（用 reply 自然回應，可以俏皮一點）
"""


def understand(text, ctrl):
    s = ctrl.snapshot()
    board_txt = "\n".join(" ".join(c if c != " " else "·" for c in s["board"][r * 3:r * 3 + 3]) for r in range(3))
    ctx = "目前盤面（X 是使用者、O 是你）：\n%s\n階段：%s；狀態：%s" % (board_txt, s["phase"], s["status"])
    out = services.llm([{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": ctx + "\n\n使用者說：「%s」" % text}], max_tokens=200, temperature=0.3)
    m = re.search(r"\{.*\}", out, re.S)
    try:
        return json.loads(m.group(0)) if m else {"intent": "chat", "reply": out.strip()[:80]}
    except Exception:
        return {"intent": "chat", "reply": out.strip()[:80]}


def handle_text(text, ctrl):
    ctrl.say(text, role="user", speak=False)
    if re.search(r"^(停|停止|暫停|stop)", text.strip(), re.I):          # 停：不等 LLM，立刻執行
        ctrl.say(ctrl.stop())
        return
    try:
        it = understand(text, ctrl)
    except Exception as e:
        ctrl.say("我剛剛沒聽懂（%s），可以再說一次嗎？" % str(e)[:40])
        return
    intent = it.get("intent")
    cell = it.get("cell")
    if intent == "stop":
        reply = ctrl.stop()
    elif intent == "resume":
        reply = ctrl.resume()
    elif intent == "new_game":
        reply = ctrl.new_game(robot_first=bool(it.get("robot_first")))
    elif intent == "place_x" and isinstance(cell, int) and 0 <= cell <= 8:
        reply = ctrl.place_for_human(cell)
    elif intent == "hint":
        reply = ctrl.hint()
    elif intent == "status":
        s = ctrl.snapshot()
        reply = it.get("reply") or s["status"]
    else:
        reply = it.get("reply") or "嗯嗯。"
    ctrl.say(reply)


# ======================= HTTP =======================
ACCESS = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _authed(self):
        q = parse_qs(urlparse(self.path).query)
        if q.get("k", [None])[0] == ACCESS:
            return True
        cookie = self.headers.get("Cookie", "")
        return ("ooxx_k=%s" % ACCESS) in cookie

    def do_GET(self):
        path = urlparse(self.path).path
        if not self._authed():
            return self._send(401, "需要存取碼：網址加上 ?k=<存取碼>".encode("utf-8"), "text/plain; charset=utf-8")
        if path in ("/", "/index.html"):
            return self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8",
                              {"Set-Cookie": "ooxx_k=%s; Path=/; Max-Age=2592000; Secure; SameSite=Strict" % ACCESS})
        if path == "/api/state":
            q = parse_qs(urlparse(self.path).query)
            CTRL.touch_client(q.get("cid", [None])[0], q.get("snd", ["0"])[0] == "1")
            return self._send(200, CTRL.snapshot())
        if path == "/api/camera.jpg":
            img = latest_frame()
            if img is None:
                return self._send(503, {"error": "相機沒有影像"})
            pv = CTRL.palm_vis
            if pv is not None:
                img = hand.draw(img, pv[0], pv[1])
            ok, buf = cv2.imencode(".jpg", cv2.resize(img, (800, 450)), [cv2.IMWRITE_JPEG_QUALITY, 70])
            return self._send(200, buf.tobytes(), "image/jpeg")
        m = re.match(r"^/api/audio/([0-9a-f]+)\.wav$", path)
        if m:
            wav = CTRL.audio.get(m.group(1))
            return self._send(200, wav, "audio/wav") if wav else self._send(404, {"error": "no audio"})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if not self._authed():
            return self._send(401, {"error": "unauthorized"})
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        if path == "/api/voice":
            ctype = self.headers.get("Content-Type", "")
            suffix = ".mp4" if "mp4" in ctype else ".ogg" if "ogg" in ctype else ".webm"
            try:
                text = services.transcribe(raw, suffix)
            except Exception as e:
                CTRL.say("語音辨識失敗：%s" % str(e)[:60], speak=False)
                return self._send(200, {"ok": False, "error": str(e)})
            if not text:
                return self._send(200, {"ok": False, "error": "沒聽到聲音"})
            threading.Thread(target=handle_text, args=(text, CTRL), daemon=True).start()
            return self._send(200, {"ok": True, "text": text})
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            return self._send(400, {"error": "bad json"})
        if path == "/api/say":
            text = str(req.get("text", "")).strip()[:200]
            if text:
                threading.Thread(target=handle_text, args=(text, CTRL), daemon=True).start()
            return self._send(200, {"ok": True})
        if path == "/api/cmd":
            c = req.get("cmd")
            if c == "stop":
                r = CTRL.stop()
            elif c == "resume":
                r = CTRL.resume()
            elif c == "reset":                             # 中斷這盤：這一步做完就收棋盤、回到開局前
                r = CTRL.reset()
            elif c == "sound_owner":                       # 這個分頁要出聲（其他分頁靜音）
                CTRL.touch_client(req.get("cid"), True)
                CTRL.sound_owner = req.get("cid")
                r = "聲音改在這個網頁播。"
            elif c == "new_game":
                r = CTRL.new_game(robot_first=bool(req.get("robot_first")))
            elif c == "hint":
                r = CTRL.hint()
            elif c == "selfplay":
                r = CTRL.self_play()
            elif c == "selfplay_loop":                     # 直播用：一盤接一盤自己對下，直到按停止
                CTRL.loop_play, CTRL.loop_fails = True, 0
                r = CTRL.self_play()
            elif c == "loop_off":                          # 這盤下完（收好）就不再開下一盤；不打斷正在下的
                CTRL.loop_play = False
                r = "好，這盤下完收好就停。"
            elif c == "hand_over" and HANDOVER_ON:          # 2026-11-14 使用者決定放棄交接功能，預設關閉
                r = CTRL.hand_over(req.get("mark", "X"))
            elif c == "hand_bg":
                r = CTRL.save_hand_bg()
            elif c == "sim_tap" and DRY:                  # 乾跑：模擬人在某格擺一顆 X
                i = int(req.get("cell", -1))
                if not (0 <= i <= 8) or VBOARD[i] != " ":
                    return self._send(200, {"ok": False, "reply": "那格不能擺"})
                VBOARD[i] = HUMAN
                return self._send(200, {"ok": True})
            elif c == "sim_clear" and DRY:                # 乾跑：模擬把棋盤清空、棋子收回料區
                VBOARD[:] = [" "] * 9
                CTRL.counts = {"X疊": 5, "O疊": 5}
                r = "（模擬）棋盤已清空，料區補滿。"
            elif c == "set_counts":
                CTRL.counts = {"X疊": int(req.get("x", 0)), "O疊": int(req.get("o", 0))}
                CTRL._save_counts()
                r = "料區已設定：X 疊 %d 顆、O 疊 %d 顆。" % (CTRL.counts["X疊"], CTRL.counts["O疊"])
            else:
                return self._send(400, {"error": "unknown cmd"})
            CTRL.say(r)
            return self._send(200, {"ok": True, "reply": r})
        self._send(404, {"error": "not found"})


def make_cert():
    """自簽憑證（手機瀏覽器要 HTTPS 才能用麥克風；第一次開會跳警告，選「繼續前往」）。"""
    crt, key = os.path.join(HERE, "cert.pem"), os.path.join(HERE, "key.pem")
    if os.path.exists(crt) and os.path.exists(key):
        return crt, key
    import datetime
    import ipaddress
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ooxx-arena.local")])
    ips = [ipaddress.ip_address(a) for a in _local_ips()]
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(k.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow() - datetime.timedelta(days=1))
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost")] + [x509.IPAddress(i) for i in ips]), False)
            .sign(k, hashes.SHA256()))
    open(crt, "wb").write(cert.public_bytes(serialization.Encoding.PEM))
    open(key, "wb").write(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                          serialization.NoEncryption()))
    return crt, key


CTRL = None

if __name__ == "__main__":
    af = os.path.join(HERE, "access.txt")
    ACCESS = open(af).read().strip() if os.path.exists(af) else secrets.token_urlsafe(9)
    open(af, "w").write(ACCESS)
    CTRL = Controller(use_robot="--no-robot" not in sys.argv)
    if not DRY:
        CAM_DEPTH = ALWAYS_DEPTH                          # 雙臂模式：相機一開始就開深度
        threading.Thread(target=camera_supervisor, daemon=True).start()
    crt, key = make_cert()
    # 直播用：OBS 的瀏覽器來源不吃自簽憑證 → 只綁本機的 HTTP（外面連不到）
    local = ThreadingHTTPServer(("127.0.0.1", 18080), Handler)
    threading.Thread(target=local.serve_forever, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print("OOXX 對弈機器人：")
    for ip in _local_ips():
        print("  https://%s:%d/?k=%s" % (ip, PORT, ACCESS))
    def _startup_check():
        time.sleep(8)                                 # 等相機子程序出第一張
        print("[校正檢查] %s" % CTRL.calib_check()[1], flush=True)
    threading.Thread(target=_startup_check, daemon=True).start()
    print("機器能下的格：%s；%s" % (sorted(CTRL.reach), "乾跑模式（模擬手臂與盤面）" if DRY else
                                   "交接乾跑（手臂模擬，相機真的）" if DRY_ARM else CTRL.arm_err or "手臂 B 已連線"), flush=True)
    try:
        srv.serve_forever()
    finally:
        if CTRL.arm:
            CTRL.arm.close()
        if CTRL.cam_proc:
            CTRL.cam_proc.kill()
