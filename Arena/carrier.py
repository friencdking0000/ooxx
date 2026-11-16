"""翻面棋子的搬運（server.py 對弈與 grip_test.py 實機測試共用）：場景（相機＋換算＋O／X 紀錄）＋吸放、偏吸、轉送、挪正、收棋盤。

- 棋子翻面（光滑背面朝上，看不到 O／X 符號）：種類只靠 ox_record.json（使用者 2026-11-15，「不要自己搞混」）——
  料疊照「固定位置」分 O／X（X 疊在 A 那側、只有 A 拿；O 疊在 B 那側、只有 B 拿），棋盤上每顆照「放的時候記的紀錄」。
  沒有紀錄的棋子不猜、不收（例外：剛剛搬的那顆掉在棋盤上，種類就是正在搬的）。
- 位置不用舊的相機校正（約差 1 公分），改用「畫面上格線四個交叉點 ↔ 手動教點的交叉點」直接換算（teach_fit.json）；
  比棋盤高的東西（料疊）先依深度做視差修正。
- 搆不到中心：吸「棋子中心＋d」、吸盤放到「目標＋d」→ 棋子中心照樣落在目標（reach_shift，|d| ≤ GRIP_OFF_MAX；背面是平的，偏著吸也密合）。
  還不行就經棋盤外的固定轉送點交給另一台（不可以借用棋盤上的格子：對弈時會被當成一步，使用者 2026-11-15）。
- 高度：棋盤是平的，每台用自己教點中心格的 Z（BOARD_Z）；棋盤外（料疊、轉送點）用深度量。往下探從估計頂面上方 OVER mm 開始，
  真空一律碰到才開（mover.Arm.touch 的高度把關）。放子最深＝目標頂面＋PLACE_MARGIN，不再往下。
- 支架檢查用相機座標（Calib/posts.py，每次 reset 後用深度重新量），另外檢查不碰到停在停放點的另一台（Mover.clear_of_other）。
- 每個動作都是「先量現況再決定」：中途失敗、上次沒做完（轉送點上留著一顆）都接得上。
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Calib"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "MG400"))

import cv2                                      # noqa: E402
import numpy as np                              # noqa: E402

import align                                    # noqa: E402
import find_pieces                              # noqa: E402
import mover                                    # noqa: E402
import posts                                    # noqa: E402
import stocks                                   # noqa: E402
from common import atomic_write, enable, sync_move    # noqa: E402
from play import Z_HOVER, to_B                  # noqa: E402

FIT = os.path.join(os.path.dirname(HERE), "Calib", "teach_fit.json")
PLAN = os.path.join(os.path.dirname(HERE), "Calib", "cell_plan.json")
OUT = os.path.join(HERE, "out")
CROSS_CACHE = os.path.join(OUT, "crosses_last.json")
BIAS_FILE = os.path.join(OUT, "pile_bias.json")
RECORD = os.path.join(HERE, "ox_record.json")   # piles：兩疊的固定位置（B 座標）與負責的手臂；board：棋盤上每顆的位置與種類；
#                                                 hand：棋盤外的轉送點（B 座標）與轉送途中放在那裡的那顆
ARMS = ("A", "B")
# 原始影像（未對齊）上格線交叉點的大概位置：只當搜尋起點（±35 px）
# 2026-11-16 換場地重量：原始畫面跟教點方向差 180 度（cam_align.json 轉回來），所以 c0 在原始畫面的右下
# （舊場地 11-15：c0 (544.0, 284.5)、c3 (662.5, 433.5)）
SEED_RAW = {"c0": (832.0, 416.7), "c1": (712.4, 416.7), "c2": (832.0, 266.1), "c3": (712.4, 266.1)}
OVER = 12.0                                     # mm：接觸偵測從估計頂面上方這麼高開始慢慢往下（教點 Z 偏低時才不會撞上去）
PIECE = stocks.PIECE_MM
PLACE_MARGIN = 2.0                              # mm：放子最深只到「目標頂面＋2 mm」（使用者 2026-11-15）：碰到就停，沒碰到也在那裡放開
PILE_TOL = 60.0                                 # mm（B 座標）：料疊離記錄的固定位置這麼近才算那一疊
TRACK = 45.0                                    # mm（B 座標）：棋子離紀錄位置這麼近就算同一顆
CELL_NEAR = 45.0                                # mm：「格子上的棋子」＝離格心這麼近（機器放的誤差都在 5 mm 內）
CELL_READ = 60.0                                # mm：讀盤時人放的棋子離格心這麼近就算那一格（格距約 95 mm）
GRIP_OFF_MAX = 25.0                             # mm：偏著吸最多偏這麼多（棋子半徑 41－吸盤約 10－邊距 6）
PICK_TRIES = 3                                  # 放一格最多從料疊拿幾次（沒吸住就重拿）
MISS_DROP = 6          # 連續這麼多次量都對不到才刪紀錄（等料疊時每 4 秒量一次，手擋住十幾秒不能就把紀錄刪掉）；開新局另外清（forget_missing）
TOL = 4.0                                       # mm：放完離格子中心超過這麼多就在棋盤上吸起來挪正（最多兩次）
BELOW_FLAT = 2.5                                # mm：棋盤上的棋子（高度已知）往下探最多到估計頂面下這麼多（碰不到空棋盤）
BELOW_DEPTH = 11.0                              # mm：棋盤外（料疊、轉送點）高度靠深度估，誤差大：探深一點，靠高度把關不吸到桌面
#   2026-11-15：B 在轉送點用 2.5 mm 沒碰到棋子（那裡 B 的 Z 跟深度估的差 3 mm 以上）
# 棋盤面高度（吸盤碰到空棋盤的 Z）：棋盤是平的，用手動教點中心格（畫面格 4）的值當全盤標準。
# 交叉點的 Z 是手壓的，B 四點差到 3.5 mm，拿來擬合斜面反而會在邊格多降（使用者 2026-11-15）。A 在空的格 3 實際碰到 -151.9。
BOARD_Z = {"A": -152.26, "B": -158.27}


class Stopped(RuntimeError):
    pass


# ======================= 紀錄與小工具 =======================
def rec_load():
    try:
        r = json.load(open(RECORD, encoding="utf-8"))
    except (OSError, ValueError):
        r = {}
    r.setdefault("piles", {})
    r.setdefault("board", [])
    return r


def rec_save(r):
    atomic_write(RECORD, json.dumps(r, ensure_ascii=False, indent=1))


def tf(M, p):
    M = np.asarray(M, float)
    return M[:, :2] @ np.asarray(p, float) + M[:, 2]


def inv_tf(M, p):
    return tf(cv2.invertAffineTransform(np.asarray(M, float)), p)


def dist(a, b):
    return float(np.hypot(*(np.asarray(a, float) - np.asarray(b, float))))


def nearest(items, xy_of, target, tol):
    """items 裡 xy_of(item) 離 target 最近、而且 < tol 的那一個；沒有回 None。"""
    best = min(items, key=lambda q: dist(xy_of(q), target), default=None)
    return best if best is not None and dist(xy_of(best), target) < tol else None


def B_of(q):
    return q["arm_xy"]["B"]


# ======================= 格線交叉點（換算的基準） =======================
def dark_center(prof, lo):
    """一段亮度剖面裡最暗那條膠帶的中心（索引加上 lo）。"""
    thr = (prof.min() + np.median(prof)) / 2
    i = int(np.argmin(prof))
    a, b = i, i
    while a > 0 and prof[a - 1] < thr:
        a -= 1
    while b < len(prof) - 1 and prof[b + 1] < thr:
        b += 1
    return lo + (a + b) / 2


def find_crosses(gray, M):
    """對齊後的影像上，格線四個交叉點的像素位置（棋盤清空時最準）。"""
    s = {k: tf(M, v) for k, v in SEED_RAW.items()}
    yb = int(s["c2"][1])
    band = gray[yb + 40:yb + 90].astype(float).mean(0)            # 下排格子裡橫切：兩條直膠帶
    xs = [dark_center(band[int(s[k][0]) - 35:int(s[k][0]) + 35], int(s[k][0]) - 35) for k in ("c0", "c1")]
    xl = int(s["c0"][0])
    bandv = gray[:, xl - 90:xl - 40].astype(float).mean(1)        # 左欄格子裡直切：兩條橫膠帶
    ys = [dark_center(bandv[int(s[k][1]) - 35:int(s[k][1]) + 35], int(s[k][1]) - 35) for k in ("c0", "c2")]
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    if not (105 < dx < 135 and 135 < dy < 165):
        raise RuntimeError("格線找得不對（間距 %.1f, %.1f px），棋盤上要清空" % (dx, dy))
    return {"c0": (xs[0], ys[0]), "c1": (xs[1], ys[0]), "c2": (xs[0], ys[1]), "c3": (xs[1], ys[1])}


def px_maps(crosses, fit):
    """每台手臂：影像像素（棋盤面高度）→ 手臂 XY 的仿射，用四個交叉點擬合。回傳 {arm: (2x3, 殘差 mm)}。"""
    out = {}
    P = np.array([[*crosses[k], 1.0] for k in ("c0", "c1", "c2", "c3")])
    for name in ARMS:
        Q = np.array([fit[name]["cells"][k][:2] for k in ("c0", "c1", "c2", "c3")], float)
        A, *_ = np.linalg.lstsq(P, Q, rcond=None)
        out[name] = (A.T, float(np.sqrt(np.mean(np.sum((P @ A - Q) ** 2, 1)))))
    return out


def crosses_cached(gray):
    """沿用上次（棋盤空的時候）找到的交叉點，但要先確認相機沒動：上次存的每個交叉點周圍 70x70 的影像，
    在現在的影像 ±12 px 內找；4 點中至少 3 點對得上（分數 > 0.6）、位移都在 3 px 內才採用（加上平均位移）。"""
    try:
        c = json.load(open(CROSS_CACHE, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ref = cv2.imread(c["img"], cv2.IMREAD_GRAYSCALE)
    if ref is None:
        return None
    sh = []
    for x, y in c["crosses"].values():
        x, y = int(round(x)), int(round(y))
        tpl = ref[y - 35:y + 35, x - 35:x + 35]
        win = gray[y - 47:y + 47, x - 47:x + 47]
        if tpl.shape != (70, 70) or win.shape != (94, 94):
            continue
        r = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(r)
        if score > 0.6:
            sh.append((loc[0] - 12, loc[1] - 12))
    if len(sh) < 3 or max(np.hypot(*np.array(sh, float).T)) > 3.0:
        print("沿用格線失敗：對得上的交叉點 %d 個、位移 %s（相機可能動過）" % (len(sh), sh), flush=True)
        return None
    m = np.mean(np.array(sh, float), 0)
    return {k: (v[0] + m[0], v[1] + m[1]) for k, v in c["crosses"].items()}


# ======================= 相機 =======================
class DirectCam:
    """直接開相機（grip_test 用；server 用 cam_worker 寫的檔，見 server.FileCam）。grab() → (對齊後彩色, 對齊後深度 5 張中位數, 對齊矩陣)。"""

    def __init__(self):
        from common import OrbbecCamera
        self.cam = OrbbecCamera(depth=True)

    def grab(self):
        M = align.load()
        img, d0 = self.cam.grab(depth=True, skip=3, timeout=15)          # 彩色和第一張深度同一組影格
        ds = [d0] + [self.cam.grab(depth=True, skip=0, timeout=15)[1] for _ in range(4)]
        d = align.apply(M, np.median(np.stack(ds), axis=0).astype(np.float32), depth=True)
        self.frames = [align.apply(M, x.astype(np.float32), depth=True) for x in ds]   # 量支架用（posts.near_depth）
        return align.apply(M, img), d, M

    def close(self):
        self.cam.close()


# ======================= 場景 =======================
class Scene:
    """相機＋換算＋O／X 紀錄＋兩台手臂。measure() 拍一次，回傳 (影像, 料疊)；最後一次的結果在 last。
    stop()：外面要求停止時回 True（每個動作之間檢查，丟 Stopped）。"""

    def __init__(self, cam, M, stop=lambda: False):
        self.cam, self.stop = cam, stop
        self.fit = json.load(open(FIT, encoding="utf-8"))
        self.plan = json.load(open(PLAN, encoding="utf-8"))
        self.cells_px = [self.plan[str(i)]["px"] for i in range(9)]
        self.cellB = [np.array(self.fit["B"]["cells"][str(i)][:2], float) for i in range(9)]
        # 吸棋盤上的棋子用的「局部棋盤面 Z」：四個交叉點教點 Z 擬合的斜面。B 靠底座那側比中心格低約 3 mm
        # （c0／c1 −160.2／−159.8，c2／c3 −157.3／−156.7），只用 BOARD_Z 往下探 2.5 mm 在格 0、1、2 碰不到棋子（2026-11-15）。
        # 放子最深仍照 BOARD_Z（使用者：不要用斜面讓邊格多降）。
        self.tilt = {}
        for n in ARMS:
            ks = [k for k in ("c0", "c1", "c2", "c3") if k in self.fit[n].get("board_z", {})]
            if len(ks) >= 3:
                P = np.array([[*self.fit[n]["cells"][k][:2], 1.0] for k in ks])
                self.tilt[n] = np.linalg.lstsq(P, np.array([self.fit[n]["board_z"][k] for k in ks]), rcond=None)[0]
        self.rec = rec_load()
        if not all(k in self.rec["piles"] for k in ("O", "X")):
            raise RuntimeError("%s 沒有 O、X 兩疊的固定位置，不知道哪疊是什麼" % RECORD)
        self.maps = None
        self.last = None
        self._bp = (None, None, None)
        self.attach(M)

    def attach(self, M):
        """接上（或重連後換上）手臂：IK 快取、檢查不碰到另一台。"""
        self.M = M
        self.calB = M.arms["B"].cal
        for a in M.arms.values():
            cache_ik(a)
            a.halted = self.stop
            a.peer_ok = (lambda a: lambda pts: M.clear_of_other(a, [to_B(a.name, p, M.T) for p in pts]))(a)
        self.reset()

    def reset(self):
        """相機對齊改過、開新局、重連手臂：下次 measure 重新找格線、量支架。"""
        self.maps = None
        self.posts_ok = False
        self.looks = {}
        self._rs = {}

    def check(self):
        if self.stop():
            raise Stopped("已停止")

    def arm_for(self, kind):
        return self.M.arms[self.rec["piles"][kind]["arm"]]

    def other(self, arm):
        return self.M.arms["B" if arm.name == "A" else "A"]

    # ---- 拍照與換算 ----
    def setup_board(self, img, d, Ma):
        """找格線交叉點 → 兩台手臂的換算、棋盤面深度。棋盤上有棋子時先沿用上次（棋盤空的時候）找到的。"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        empty = not find_pieces.pieces(img, self.plan, blank_ok=True)
        cr = None if empty else crosses_cached(gray)
        if cr is None:
            cr = find_crosses(gray, Ma)
            if empty:
                os.makedirs(OUT, exist_ok=True)
                ipath = os.path.join(OUT, "crosses_last.png")
                cv2.imwrite(ipath, gray)
                atomic_write(CROSS_CACHE, json.dumps(dict(time=time.strftime("%H:%M:%S"), img=ipath, crosses=cr)))
        win = np.concatenate([d[int(y) - 5:int(y) + 6, int(x) - 5:int(x) + 6].ravel() for x, y in cr.values()])
        if not np.any(win > 0):
            raise RuntimeError("格線交叉點量不到深度")
        self.d_board = float(np.median(win[win > 0]))       # 交叉點（膠帶交叉）離格心 ≥ 69 mm，格子上有棋子也蓋不到
        self.Ma = Ma
        self.pp = tf(Ma, (posts.CX, posts.CY))              # 主點（對齊後影像）：視差修正的中心
        self.maps = px_maps(cr, self.fit)
        print("格線交叉點 %s；換算殘差 A %.1f、B %.1f mm；棋盤面深度 %.1f mm" % (
            {k: tuple(round(v, 1) for v in p) for k, p in cr.items()}, self.maps["A"][1], self.maps["B"][1], self.d_board), flush=True)

    def setup_posts(self, d):
        """用深度量兩側支架（相機座標），之後 mover 的支架檢查都改用它。d=None：不量，用 posts.use_last() 載入的存檔。"""
        if d is not None and posts.measure(d, self.d_board, cv2.invertAffineTransform(self.Ma)) is None:
            raise RuntimeError("深度看不到兩側支架（被擋住？），不動手臂")
        base = {n: self.cam_xy(n, (0.0, 0.0)) for n in ARMS}
        self.post_margin = lambda name, xy: posts.clearance(base[name], self.cam_xy(name, xy))
        # mover 的 post() 與路線檢查都經過 mover.hits_post：換掉它就全部換掉
        # （舊的 plan_cells.hits_post 用影像矩形＋舊校正：把格 3 擋掉，架子移位後也可能放過真的會撞的位置）
        mover.hits_post = lambda cal, xy, z: (lambda h: h[0] if h[1] < 0 else None)(self.post_margin(cal["arm"], xy))
        self.posts_ok = True
        self._rs = {}
        print("支架（相機座標 mm）：%s" % {k: tuple(round(x) for x in v) for k, v in posts.POSTS_CAM.items()}, flush=True)

    def measure(self):
        """拍一次（手臂都在棋盤外時才呼叫）：對齊後彩色＋深度、料疊（種類照固定位置）；棋盤紀錄跟著實際位置更新，
        連續 MISS_DROP 次都沒對到棋子的紀錄刪掉（棋子被人拿走了）。"""
        self.check()
        img, d, Ma = self.cam.grab()
        if self.maps is None:
            self.setup_board(img, d, Ma)
        for i in range(4):          # 鋁柱的深度會閃（2026-11-16 新場地：單張 0～7000 點，5 張中位數常變 0）：量不到就重拍
            if self.posts_ok:
                break
            try:
                self.setup_posts(posts.near_depth(self.cam.frames) if getattr(self.cam, "frames", None) else d)
            except RuntimeError:
                if i < 3:
                    img, d, Ma = self.cam.grab()
                    continue
                # 4 次都量不到（右側支架 2026-11-16 幾乎量不到）：支架跟相機是同一個架子，相機座標不會變 → 沿用上次量到的
                t = posts.use_last()
                if t is None:
                    raise           # 從來沒量到過：照舊不動手臂
                print("支架：深度量不到，沿用 %s 量到的位置" % t, flush=True)
                self.setup_posts(None)
        self.depth = d
        piles = []
        for s in stocks.detect(img, d, self.calB, self.cells_px):
            s["arm_xy"] = self.locate(s["px"], s["top"])
            s["z_top"] = {n: self.z_of(n, s["top"]) for n in ARMS}
            s["z_tab"] = {n: self.z_of(n, s["table"]) for n in ARMS}
            piles.append(s)
        self.assign_piles(piles)
        self.last = (img, piles)
        self.board_pieces(img, depth=d, update=True)
        self.sync_hand(piles)
        return img, piles

    def locate(self, px, depth):
        """影像像素（某個深度的表面上）→ 兩台手臂的 XY：先做視差修正（比棋盤高的東西在影像上會往外偏）。"""
        g = self.pp + (np.asarray(px, float) - self.pp) * depth / self.d_board
        return {n: tf(self.maps[n][0], g) for n in ARMS}

    def z_of(self, name, depth):
        """深度 → 這台手臂的 Z（棋盤面＝BOARD_Z）。"""
        return BOARD_Z[name] + (self.d_board - depth)

    def px_of(self, name, xy):
        """手臂 XY → 對齊後影像像素（棋盤面那層）。"""
        return inv_tf(self.maps[name][0], xy)

    def b_to(self, name, xyB):
        """B 座標 → 這台手臂的 XY（經棋盤面那層的像素）。"""
        return tf(self.maps[name][0], self.px_of("B", xyB))

    def cam_xy(self, name, xy):
        """手臂 XY → 相機座標水平位置（mm）：經棋盤面那層換算回原始影像像素，再乘深度。"""
        return posts.px_to_cam(inv_tf(self.Ma, self.px_of(name, xy)), self.d_board)

    def arm_from_cam(self, name, cam):
        """相機座標水平位置（mm，棋盤面）→ 手臂 XY。"""
        return tf(self.maps[name][0], tf(self.Ma, posts.cam_to_px(cam, self.d_board)))

    def surface_z(self, name, xy):
        """xy 那點最上面的表面在這台手臂的 Z（用深度量；棋盤外的桌面比棋盤低，不能用 BOARD_Z）。量不到回 None。"""
        u, v = (int(round(t)) for t in self.px_of(name, xy))
        win = self.depth[max(0, v - 6):v + 7, max(0, u - 6):u + 7]
        return self.z_of(name, float(np.median(win[win > 0]))) if np.any(win > 0) else None

    # ---- O／X 紀錄 ----
    def assign_piles(self, piles):
        """料疊種類只看位置：每個記錄的固定位置配離它最近、PILE_TOL 內的那疊，其他一律「?」（看到的符號只留在 seen）。
        記錄的位置沒有疊、人把料疊擺到別處了 → relearn_piles 用看到的重新設定。"""
        for s in piles:
            s["seen"], s["kind"] = s["kind"], "?"
        for k, h in self.rec["piles"].items():
            s = nearest(piles, B_of, h["B"], PILE_TOL)
            if s is not None:
                s["kind"] = k
        self.relearn_piles(piles)

    def relearn_piles(self, piles):
        """人換了料區的擺法（使用者 2026-11-16：要能照我擺的料區重新設定）：某一種的記錄位置附近沒有疊時，
        在看到、還沒配到的疊裡找新位置。翻面看不到圈叉，只能靠擺的位置分：候選要
          - 至少 2 顆（掉在桌上的單顆不算）、不在轉送點上；
          - 在負責那台的那一側（B 臂＝B 座標 X 比中間格小 50 mm 以上，A 臂＝大 50 mm 以上）；
          - 負責那台吸得到最上面那顆（支架、IK、路線都檢查：支架的腳、袋子之類會在這裡被排掉）；
          - 只有一個。多個或沒有就不改（開局會說料區有問題）。"""
        cx = float(self.cellB[4][0])
        hb = self.hand_B()
        for k, h in self.rec["piles"].items():
            if self.pile(piles, k) is not None:
                continue
            arm = self.M.arms[h["arm"]]
            side = (lambda x: x < cx - 50) if h["arm"] == "B" else (lambda x: x > cx + 50)
            cands = []
            for st in piles:
                xy = B_of(st)
                if st["kind"] != "?" or st["n"] < 2 or not side(xy[0]) or (hb is not None and dist(xy, hb) < PILE_TOL):
                    continue
                sp = self.pile_spot(st, "%s 疊" % k)
                n = arm.name
                if reach_shift(arm, [(sp["tag"], np.asarray(sp["xy"][n], float), sp["z_top"][n] - sp["below"])]) is None:
                    continue
                cands.append(st)
            if len(cands) != 1:
                if cands:
                    print("[料區] %s 疊不在記錄的位置，看到 %d 疊都可能是，不自己改" % (k, len(cands)), flush=True)
                continue
            st = cands[0]
            old = h["B"]
            h["B"] = np.round(B_of(st), 1).tolist()
            h["t"] = time.strftime("%Y-%m-%d %H:%M") + "（照人擺的位置重新設定，原本 %s）" % old
            st["kind"] = k
            rec_save(self.rec)
            print("[料區] %s 疊換位置：B %s → %s（%d 顆，%s 臂拿）" % (k, old, h["B"], st["n"], h["arm"]), flush=True)

    @staticmethod
    def pile(piles, kind):
        return next((s for s in piles if s["kind"] == kind), None)

    def mark(self, xyB, kind):
        """棋盤上 xyB（B 座標）放了一顆 kind：記下來（同一位置舊的紀錄換掉）。"""
        self.unmark(xyB, save=False)
        self.rec["board"].append(dict(B=np.round(xyB, 1).tolist(), kind=kind, t=time.strftime("%Y-%m-%d %H:%M:%S")))
        rec_save(self.rec)

    def unmark(self, xyB, save=True):
        self.rec["board"] = [e for e in self.rec["board"] if dist(e["B"], xyB) >= TRACK]
        self._bp = (None, None, None)                 # 紀錄變了：種類要重配
        if save:
            rec_save(self.rec)

    def board_pieces(self, img, depth=None, update=False):
        """棋盤上（含旁邊）的棋子：兩台手臂座標（棋子中心）、頂面 Z（有深度就量，疊起來的會比較高）、種類（只看紀錄，沒有是「?」）。
        update（measure 用）：紀錄的位置跟著實際位置走（差 2 mm 以上才寫檔）；沒對到棋子的紀錄記一次，連續 MISS_DROP 次就刪。
        同一張影像只算一次。"""
        if self._bp[0] is not img:
            if depth is None and self.last is not None and img is self.last[0]:
                depth = self.depth                     # 重算最近一次量的那張：照樣用它的深度（疊起來的棋子要看得出來）
            out = []
            for p in find_pieces.pieces(img, self.plan, blank_ok=True):
                top = self.d_board - PIECE
                if depth is not None:
                    u, v = (int(round(x)) for x in p["px"])
                    win = depth[v - 8:v + 9, u - 8:u + 9]
                    if np.any(win > 0):
                        top = float(np.median(win[win > 0]))
                out.append(dict(p, seen=p["kind"], kind="?", arm_xy=self.locate(p["px"], top),
                                z_top={n: self.z_of(n, top) for n in ARMS}))
            free, pairs = list(out), []                # 每筆紀錄配最近、TRACK 內的那顆
            for e in self.rec["board"]:
                q = nearest(free, B_of, e["B"], TRACK)
                if q is not None:
                    free.remove(q)
                    q["kind"] = e["kind"]
                pairs.append((e, q))
            self._bp = (img, out, pairs)
        out, pairs = self._bp[1], self._bp[2]
        dirty = False
        if update:
            for e, q in pairs:
                if q is None:
                    e["miss"] = e.get("miss", 0) + 1
                    continue
                if e.pop("miss", 0):
                    dirty = True
                if dist(B_of(q), e["B"]) > 2.0:
                    e["B"] = B_of(q).round(1).tolist()
                    dirty = True
            gone = [e for e in self.rec["board"] if e.get("miss", 0) >= MISS_DROP]
            if gone:
                print("紀錄裡 %s 已經不在棋盤上：刪掉" % [(e["kind"], e["B"]) for e in gone], flush=True)
                self.rec["board"] = [e for e in self.rec["board"] if e.get("miss", 0) < MISS_DROP]
                self._bp = (None, None, None)
                dirty = True
        if dirty:
            rec_save(self.rec)
        return out

    def forget_missing(self, img):
        """棋盤看起來是空的（開新局）：這張影像上對不到棋子的紀錄全部刪掉（被人拿走的），不等 MISS_DROP 次。"""
        self.board_pieces(img)
        gone = [e for e, q in self._bp[2] if q is None]
        if gone:
            print("開新局：紀錄裡 %s 已經不在棋盤上，刪掉" % [(e["kind"], e["B"]) for e in gone], flush=True)
            self.rec["board"] = [e for e in self.rec["board"] if all(e is not g for g in gone)]
            self._bp = (None, None, None)
            rec_save(self.rec)

    def cell_of(self, q, near=CELL_READ):
        i = min(range(9), key=lambda i: dist(B_of(q), self.cellB[i]))
        return i if dist(B_of(q), self.cellB[i]) < near else None

    def at_cell(self, img, cell):
        """格子中心 CELL_NEAR 內、離中心最近的那顆棋子（不分種類）。沒有回 None。"""
        return nearest(self.board_pieces(img), B_of, self.cellB[cell], CELL_NEAR)

    def read_board(self, img, depth=None):
        """9 格：紀錄的種類，有棋子但沒紀錄是「?」，空格 " "。只讀、不改紀錄（server 每 0.25 秒讀一次）。
        格線還沒找過時用這張影像找（要有深度）；找不到回 None。"""
        if self.maps is None:
            if depth is None:
                return None
            self.setup_board(img, depth, align.load())
        b = [" "] * 9
        for q in self.board_pieces(img):
            c = self.cell_of(q)
            if c is not None and b[c] in (" ", "?"):
                b[c] = q["kind"]
        return b

    def mark_cell(self, img, cell, kind):
        """讀盤認定格 cell 上新出現的棋子是 kind（人下的）：記進紀錄。看不到那顆回 False。"""
        q = None if img is None else nearest(self.board_pieces(img), B_of, self.cellB[cell], CELL_READ)
        if q is None:
            return False
        self.mark(B_of(q), kind)
        return True

    # ---- 轉送點 ----
    def hand_B(self):
        h = self.rec.get("hand")
        return None if not h else np.asarray(h["B"], float)

    def hand_kind(self):
        return (self.rec.get("hand") or {}).get("piece")

    def set_hand(self, kind):
        h = self.rec.get("hand")
        if h is not None and (h.get("piece") != kind or "miss" in h):
            h["piece"] = kind
            h.pop("miss", None)
            rec_save(self.rec)

    def hand_piece(self, piles):
        """轉送點上看到的那顆（料疊偵測的一疊）。"""
        hB = self.hand_B()
        return None if hB is None else nearest(piles, B_of, hB, TRACK)

    def sync_hand(self, piles):
        """紀錄說轉送點上有一顆、但連續 MISS_DROP 次量都看不到：清掉紀錄（單張漏掉不算，平躺的一顆深度只差約 2 mm）。"""
        h = self.rec.get("hand")
        if not self.hand_kind():
            return
        if self.hand_piece(piles) is not None:
            if h.pop("miss", 0):
                rec_save(self.rec)
            return
        h["miss"] = h.get("miss", 0) + 1
        if h["miss"] >= MISS_DROP:
            print("紀錄說轉送點上有一顆 %s，但連續 %d 次都看不到：清掉紀錄" % (self.hand_kind(), MISS_DROP), flush=True)
            self.set_hand(None)
        rec_save(self.rec)

    # ---- 搬運的起點／終點（各臂座標、Z） ----
    def cell_spot(self, cell):
        z = {n: BOARD_Z[n] + PIECE for n in ARMS}
        return dict(tag="格 %d" % cell, xy={n: np.array(self.fit[n]["cells"][str(cell)][:2], float) for n in ARMS},
                    z_top=z, z_surf=dict(BOARD_Z), below=BELOW_FLAT, z_put=z)

    @staticmethod
    def pile_spot(st, tag):
        """料疊（或轉送點上的那顆）：吸最上面那顆；放的時候放在它上面。
        只剩一顆躺在桌面：桌面高度靠深度估、誤差約 2 mm，「比底面高多少才開真空」放寬到 1 mm（2026-11-15 量到 2.6、以前 1.9 被擋）。"""
        return dict(tag=tag, xy=dict(st["arm_xy"]), z_top=st["z_top"], z_surf=st["z_tab"], below=BELOW_DEPTH,
                    z_put={n: st["z_top"][n] + PIECE for n in ARMS}, h_min=1.0 if st["n"] <= 1 else 2.0)

    def board_z(self, name, xy):
        """這台手臂在 xy 處的局部棋盤面 Z（教點斜面；沒有就用 BOARD_Z）。"""
        t = self.tilt.get(name)
        return BOARD_Z[name] if t is None else float(t[0] * xy[0] + t[1] * xy[1] + t[2])

    def piece_spot(self, q):
        """棋盤上的棋子：頂面＝局部棋盤面＋一顆；深度看起來疊在別顆上（比平面的估計高 2.5 mm 以上）才加上疊起來的高度。"""
        zt = {}
        for n in ARMS:
            zb = self.board_z(n, q["arm_xy"][n])
            stack = q["z_top"][n] - (BOARD_Z[n] + PIECE)          # 深度量的頂面跟平面估計差多少
            zt[n] = zb + PIECE + (stack if stack > 2.5 else 0.0)
        # 往下探 6 mm、比局部棋盤面高 5.5 mm 以上才開真空：實測碰到棋盤上的棋子是 8.6～9.7 mm，吸盤直接壓到空棋盤約 4.5 mm
        # （COMPRESS 加進去的）。2026-11-15 A 在格 8（離 A 底座很近）探 2.5 mm 沒碰到，收棋盤停下。
        return dict(tag="棋盤上的棋子", xy=dict(q["arm_xy"]), z_top=zt, z_surf={n: zt[n] - PIECE for n in ARMS},
                    below=6.0, z_put={n: zt[n] + PIECE for n in ARMS}, h_min=5.5)

    def table_spot(self, xyB, tag, piles, need_free=True):
        """棋盤外桌面上的一點（放東西用）：上面有東西（need_free）、量不到桌面高度都回 None。"""
        if xyB is None or (need_free and nearest(piles, B_of, xyB, PILE_TOL) is not None):
            return None
        xy = {n: self.b_to(n, xyB) for n in ARMS}
        zs = {n: self.surface_z(n, xy[n]) for n in ARMS}
        if any(z is None for z in zs.values()):
            return None
        z = {n: zs[n] + PIECE for n in ARMS}
        return dict(tag=tag, xy=xy, z_top=z, z_surf=zs, below=BELOW_DEPTH, z_put=z)

    def hand_spot(self, piles, need_free=True):
        """空的轉送點（棋盤外，兩台都搆得到）。"""
        return self.table_spot(self.hand_B(), "轉送點", piles, need_free)

    # ---- 規劃 ----
    def shift(self, arm, src, dst):
        """arm 從 src 吸、放到 dst 的偏移 d（reach_shift）；搆不到回 None。同一組點只算一次（reset 時清掉）。"""
        n = arm.name
        pts = [(src["tag"], np.asarray(src["xy"][n], float), src["z_top"][n] - src["below"]),
               (dst["tag"], np.asarray(dst["xy"][n], float), dst["z_put"][n] - 6.0)]
        key = (n,) + tuple(round(float(v)) for _, p, z in pts for v in (p[0], p[1], z))
        if key not in self._rs:
            self._rs[key] = reach_shift(arm, pts)
        return self._rs[key]

    def look(self, arm):
        """手臂退到拍照點（自己那側、棋盤外；第一次用時才算），搆不到就回停放點。"""
        if arm.name not in self.looks:
            v = view_spot(arm, self)
            print("%s 臂拍照點：%s" % (arm.name, "停放點" if v is None else "%s（%s 座標）" % (v.round(1), arm.name)), flush=True)
            self.looks[arm.name] = v
        v = self.looks[arm.name]
        arm.go_park() if v is None else arm.goto(v)


# ======================= 手臂動作 =======================
def cache_ik(arm):
    """IK 結果快取：MG400 四軸只有一組解，跟當下姿態無關；reach_shift／路線檢查會重複問同一點（每次都是一趟 TCP）。
    重複呼叫不會疊兩層。"""
    if getattr(arm, "_ik_raw", None) is None:
        arm._ik_raw, arm._ik_memo = arm.ik, {}

    def ik(p, z):
        k = (round(float(p[0]), 1), round(float(p[1]), 1), round(float(z), 1))
        if k not in arm._ik_memo:
            arm._ik_memo[k] = arm._ik_raw(p, z)
        return arm._ik_memo[k]
    arm.ik = ik


def begin(M):
    """開始動之前：兩台上電、碰撞偵測、真空關、都回停放點（clear_of_other 的前提：另一台在停放點）。"""
    for a in M.arms.values():
        enable(a.arm, speed=mover.TRAVEL_SPEED)
        a.arm.dash.sendRecvMsg("SetCollisionLevel(3)")
        a.arm.dash.sendRecvMsg("DOExecute(1,0)")
    for a in M.arms.values():
        if dist(a.arm.pose()[:2], a.park) > 5.0:       # park_all 已經停好的就不用再縮回、伸出一次
            a.go_park()


def lift_off(arm):
    """吸起或放下後抬離：碰撞保護若已觸發（壓到棋子或棋盤），先關碰撞偵測、清報警、上電，抬起來再開回來（往上抬不會撞到東西）。真空輸出不動。"""
    if arm.arm.mode() == 9:
        arm.reenable()
    d = arm.arm.dash
    d.sendRecvMsg("SetCollisionLevel(0)")
    try:
        arm.lift()
    finally:
        d.sendRecvMsg("SetCollisionLevel(3)")


def carry(arm, src, z_top, z_surf, below, dst, z_dst, check=lambda: None, on_release=None, h_min=2.0):
    """吸盤到 src、碰到才開真空吸起 → 搬到 dst → 慢慢往下、碰到才關真空（最深 z_dst＋PLACE_MARGIN）。
    回傳 (有沒有開真空, 放下接觸 Z)。check() 在每段之間呼叫（要停就丟例外）。on_release：棋子一放開就呼叫（手臂還在收回）。"""
    arm.goto(src)
    check()
    zc = arm.touch(src, z_top, vacuum_on=True, over=OVER, below=below, z_surf=z_surf, h_min=h_min)
    if zc is None:
        arm.arm.dash.sendRecvMsg("DOExecute(1,0)")
        arm.lift()
        return False, None
    lift_off(arm)
    check()
    if dist(dst, src) < 40:                          # 同一格裡挪：在懸停高度直接平移，不用縮回去轉
        sync_move(arm.arm, "MovL(%f,%f,%f,%f)" % (dst[0], dst[1], Z_HOVER, arm.r))
    else:
        arm.goto(dst)
    zp = arm.touch(dst, z_dst, vacuum_on=False, over=OVER, below=-PLACE_MARGIN)
    arm.arm.dash.sendRecvMsg("DOExecute(1,0)")
    time.sleep(0.6)
    if on_release:
        try:
            on_release()
        except Exception:
            pass
    lift_off(arm)                                   # 放下時棋子壓到棋盤也可能觸發碰撞保護（2026-11-15 格 5）
    return True, zp


def view_spot(arm, sc):
    """放完拍照用的位置：自己那側、棋盤外（相機看得到所有格子和料疊）。從底座往「畫面外側、遠離自己料疊」的方向伸 210～250 mm，
    第一個 IK、支架、路線都可以的點；都不行回 None（改回停放點）。"""
    base = sc.cam_xy(arm.name, (0.0, 0.0))
    u = np.array([0.83, -0.55]) if arm.name == "A" else np.array([-0.83, 0.55])   # A：往畫面右下；B：往畫面左上
    for r in (230.0, 250.0, 210.0):
        v = sc.arm_from_cam(arm.name, base + r * u)
        if arm.ik(v, Z_HOVER) and not arm.post(v, Z_HOVER) and arm.leg_ok(arm.park, v):
            return v
    return None


def reachable(arm, pts):
    """pts：[(說明, xy, z), …] 依序要去的點（前後接停放點）都可以嗎：支架（純計算）→ IK（TCP，有快取）→ 路線 → 不碰另一台，一不行就停。"""
    if any(arm.post(p, Z_HOVER) for _, p, _ in pts):
        return False
    if not all(arm.ik(p, z) and arm.ik(p, Z_HOVER) for _, p, z in pts):
        return False
    seq = [arm.park] + [p for _, p, _ in pts] + [arm.park]
    return all(arm.leg_ok(a, b) for a, b in zip(seq, seq[1:])) and arm.peer_ok([p for _, p, _ in pts])


def reach_shift(arm, pts):
    """pts：同一次搬運吸的那點和放的那點。找最小的偏移 d（|d| ≤ GRIP_OFF_MAX），讓每個 xy+d 都搆得到：
    吸棋子「中心＋d」、吸盤放到「目標＋d」→ 棋子中心正好落在目標。d＝(0, 0) 就是吸正中央；都不行回 None。
    每圈先試沿目標半徑方向。"""
    if reachable(arm, pts):
        return np.zeros(2)
    p0 = np.asarray(pts[-1][1], float)
    a0 = np.arctan2(p0[1], p0[0])
    angs = [a0 + np.radians(sg * k) for k in range(0, 181, 15) for sg in ((1,) if k in (0, 180) else (1, -1))]
    for rr in np.arange(2.0, GRIP_OFF_MAX + 0.1, 2.0):
        for a in angs:
            d = rr * np.array([np.cos(a), np.sin(a)])
            if reachable(arm, [(t, xy + d, z) for t, xy, z in pts]):
                return d
    return None


def deliver(sc, src, dst, arms, on_release=None):
    """依序試 arms：第一台搆得到（可偏 d）的就搬。回傳 (手臂, d, 有沒有開真空) 或 None（都搆不到）。"""
    for arm in arms:
        d = sc.shift(arm, src, dst)
        if d is None:
            continue
        n = arm.name
        peer = sc.other(arm)
        if dist(peer.arm.pose()[:2], peer.park) > 5.0:     # 另一台不在停放點（例如 B 的拍照點）：先回去，碰撞檢查才成立
            peer.go_park()
        print("   %s 臂：%s %s → %s %s%s" % (n, src["tag"], np.round(src["xy"][n], 1), dst["tag"], np.round(dst["xy"][n], 1),
                                          "（中心搆不到，吸盤偏 %s；棋子中心照樣落在目標）" % d.round(1) if np.any(d) else ""), flush=True)
        sc.check()
        ok, zp = carry(arm, src["xy"][n] + d, src["z_top"][n], src["z_surf"][n], src["below"],
                       dst["xy"][n] + d, dst["z_put"][n], check=sc.check, on_release=on_release, h_min=src.get("h_min", 2.0))
        if not ok:
            print("   沒碰到棋子、沒開真空", flush=True)
        else:
            print("   放下（%s）" % ("沒偵測到接觸，在最深點 %.1f 放開" % (dst["z_put"][n] + PLACE_MARGIN) if zp is None
                                  else "碰到 Z %.1f 就放開" % zp), flush=True)
        return arm, d, ok
    return None


# ======================= 放子 =======================
def bias_get(name, kind, pile_xy):
    """料疊吸取點補正：最近 5 次「從料疊放到格子」量到的（實際落點－格心＋當時的補正）平均；料疊移動超過 30 mm 的不算。"""
    try:
        b = json.load(open(BIAS_FILE, encoding="utf-8"))
    except (OSError, ValueError):
        return np.zeros(2), 0
    s = [e for e in b.get(name + kind, []) if dist(e["pile"], pile_xy) < 30 and np.hypot(*e["err"]) <= 15][-5:]   # 滑掉的那次不算
    if not s:
        return np.zeros(2), 0
    m = np.mean([e["err"] for e in s], 0)
    return m * min(1.0, 10.0 / max(float(np.hypot(*m)), 1e-6)), len(s)                                      # 最多補 10 mm


def bias_add(name, kind, pile_xy, err):
    try:
        b = json.load(open(BIAS_FILE, encoding="utf-8"))
    except (OSError, ValueError):
        b = {}
    b.setdefault(name + kind, []).append(dict(pile=np.round(pile_xy, 1).tolist(), err=np.round(err, 2).tolist(),
                                              t=time.strftime("%m-%d %H:%M:%S")))
    b[name + kind] = b[name + kind][-10:]
    os.makedirs(OUT, exist_ok=True)
    atomic_write(BIAS_FILE, json.dumps(b, indent=1))


def settle(sc, cell, arms, first):
    """量格子上那顆離格心多遠，超過 TOL 就在棋盤上吸起來挪到格心（arms 依序試，最多兩次）。first＝(影像, 那顆) 剛量好的。
    回傳 (每次量的誤差 mm, 最後一次量還看得到它嗎)。結束時 sc.last 是最後一次量的。"""
    tgt, dst = sc.cellB[cell], sc.cell_spot(cell)
    img, pc = first
    errs = []
    for k in range(3):
        if k:
            img, _ = sc.measure()
            pc = sc.at_cell(img, cell)
        if pc is None:
            print("   格 %d 上找不到棋子" % cell, flush=True)
            return errs, False
        e = B_of(pc) - tgt
        errs.append(round(float(np.hypot(*e)), 1))
        print("   量：離格子中心 %.1f mm（B 座標差 %s）" % (errs[-1], e.round(1)), flush=True)
        if errs[-1] <= TOL or k == 2:
            break
        print("   偏了，在棋盤上吸起來挪到格子中心", flush=True)
        r = deliver(sc, sc.piece_spot(pc), dst, arms)
        if r is None:
            print("   兩台都吸不到它，不挪", flush=True)
            break
        sc.look(r[0])
        if not r[2]:
            break
    return errs, True


def place(sc, kind, cell, on_release=None):
    """從 kind 疊拿一顆放到格 cell（畫面格 0～8），放完量、偏了挪正。回傳 dict(ok, why, err_mm, by)。
    每一輪先看現況再決定：格子上已經有它 → 挪正、完成；棋盤上多了一顆沒紀錄的（剛剛掉的）→ 撿起來放進格子；
    轉送點上有一顆 kind → 另一台（或自己）從轉送點放進格子；否則從料疊拿：搆得到（可偏 d）就直接放，搆不到先放到轉送點。"""
    own = sc.arm_for(kind)
    oth = sc.other(own)
    res = dict(cell=cell, kind=kind, ok=False)

    def fail(why):
        print("   " + why, flush=True)
        res["why"] = why
        return res

    img, piles = sc.measure()
    pc = sc.at_cell(img, cell)
    if pc is not None and pc["kind"] != kind:
        return fail("格 %d 上已經有一顆%s" % (cell, "沒有紀錄的棋子" if pc["kind"] == "?" else " " + pc["kind"]))
    unk0 = [B_of(q) for q in sc.board_pieces(img) if q["kind"] == "?"]
    dst = sc.cell_spot(cell)
    last, from_pile = None, None                      # 最後一次放進這格的 (手臂, d)；那次是不是從料疊直接放的 (料疊位置, 補正)
    last_ok = False                                   # 最後一次搬有沒有吸到、放開
    for _ in range(2 * PICK_TRIES + 2):
        if pc is not None:
            sc.mark(B_of(pc), kind)
            if from_pile is not None:                  # 從料疊直接放的落點誤差＝料疊中心估計的誤差：記成這疊的補正
                bias_add(own.name, kind, from_pile[0], pc["arm_xy"][own.name] - dst["xy"][own.name] + from_pile[1])
            arms = [last[0], sc.other(last[0])] if last else [own, oth]
            errs, seen = settle(sc, cell, arms, (img, pc))
            if not seen:                               # 挪正時掉了：照「剛掉的」再撿（紀錄會跟著它，或量不到就變成「?」）
                img, piles = sc.last
                pc = sc.at_cell(img, cell)
                continue
            res.update(ok=True, err_mm=errs, by=last[0].name if last else None)
            if on_release:                             # 「換你」等量好、挪正完才講（不然手臂還會回棋盤上，可能碰到人剛下的棋子）
                try:
                    on_release()
                except Exception:
                    pass
            return res
        drop = [q for q in sc.board_pieces(img) if q["kind"] == "?" and all(dist(B_of(q), u) >= TRACK for u in unk0)
                and sc.cell_of(q, 30) in (None, cell)]     # 正好在別格中心的是人剛下的，不是掉的
        if last_ok and not drop:                       # 剛放好卻看不到、也沒看到掉在哪：多半是人的手擋住，等一下再看，不要再拿一顆疊上去
            for _ in range(4):
                print("   看不到剛放進格 %d 的棋子（手擋住？）：等 2 秒再看" % cell, flush=True)
                time.sleep(2.0)
                sc.check()
                img, piles = sc.measure()
                pc = sc.at_cell(img, cell)
                if pc is not None:
                    break
            if pc is not None:
                continue
            last_ok = False
        hp = sc.hand_piece(piles) if sc.hand_kind() == kind else None
        from_pile = None
        if drop:
            print("   棋盤上多了一顆（剛剛掉的 %s）：撿起來放進格 %d" % (kind, cell), flush=True)
            r = deliver(sc, sc.piece_spot(drop[0]), dst, [own, oth])
            if r is None:
                return fail("掉在棋盤上的那顆兩台都搬不進格 %d" % cell)
        elif hp is not None:
            r = deliver(sc, sc.pile_spot(hp, "轉送點上的 %s" % kind), dst, [oth, own])
            if r is None:
                return fail("轉送點上的 %s 兩台都搬不進格 %d" % (kind, cell))
        else:
            st = sc.pile(piles, kind)
            if st is None:
                return fail("%s 疊的固定位置附近沒有料疊" % kind)
            src = sc.pile_spot(st, "%s 疊（%d 顆）" % (kind, st["n"]))
            bias, _ = bias_get(own.name, kind, st["arm_xy"][own.name])
            src["xy"][own.name] = src["xy"][own.name] + bias
            if sc.shift(own, src, dst) is not None:
                r = deliver(sc, src, dst, [own])
                from_pile = (st["arm_xy"][own.name], bias)
            else:
                h = sc.hand_spot(piles)
                if h is None and sc.hand_kind() not in (None, kind):
                    print("   轉送點上留著一顆 %s：先收回它的料疊" % sc.hand_kind(), flush=True)
                    if not flush_hand(sc, piles):
                        return fail("轉送點上留著一顆 %s，收不回它的料疊" % sc.hand_kind())
                    img, piles = sc.measure()
                    pc = sc.at_cell(img, cell)
                    continue
                if h is None or sc.shift(oth, h, dst) is None or sc.shift(own, src, h) is None:
                    return fail("格 %d：%s 臂搆不到，經轉送點也不行%s" % (cell, own.name, "（轉送點上有東西或量不到）" if h is None else ""))
                print("   %s 臂搆不到格 %d：先放到棋盤外的轉送點，再由 %s 臂放進去" % (own.name, cell, oth.name), flush=True)
                r = deliver(sc, src, h, [own])
                if r[2]:
                    sc.set_hand(kind)                  # 一放開就記（之後量之前被停止，也不會變成沒紀錄的棋子）
                own.go_park()                          # 另一台動之前，這台一定在停放點
                if not r[2]:
                    return fail("料疊沒吸到（沒碰到棋子）")
                img, piles = sc.measure()
                pc = sc.at_cell(img, cell)
                continue
        arm, d, ok = r
        last, last_ok = (arm, d), ok
        if ok:                                         # 一放開就先記在格心（量之前被停止、相機斷，這顆也不會變成沒紀錄的「?」）
            sc.mark(sc.cellB[cell], kind)
        sc.look(arm)
        img, piles = sc.measure()
        pc = sc.at_cell(img, cell)
        if pc is None and not ok:
            return fail("沒碰到棋子、沒開真空")
    return fail("試了好幾次都沒放上格 %d" % cell)


# ======================= 收棋盤 =======================
def flush_hand(sc, piles):
    """轉送點上留著一顆（紀錄有種類、也看得到）：由負責那種的手臂收回它的料疊。回傳有沒有收。"""
    k, hp = sc.hand_kind(), sc.hand_piece(piles)
    st = None if k is None else sc.pile(piles, k)
    if hp is None or st is None:
        return False
    r = deliver(sc, sc.pile_spot(hp, "轉送點上的 %s" % k), sc.pile_spot(st, "%s 疊" % k), [sc.arm_for(k)])
    if r is None:
        return False
    r[0].go_park()
    return r[2]


def drag_closer(sc, arm, q):
    """吸不到也轉送不了的棋子：吸盤壓在棋子靠可達那側、開真空增加摩擦，慢慢拖 35 mm 到搆得到的地方。回傳有沒有拖。"""
    sp = sc.piece_spot(q)
    n = arm.name
    c, zt = np.asarray(sp["xy"][n], float), sp["z_top"][n]
    r = float(np.hypot(*c))
    u = c / r
    sgn = 1.0 if r < 300 else -1.0                             # 太近往外拖、太遠往底座拖
    p = None
    for k in (16.0, 20.0, 24.0, 28.0, 31.0):
        cand = c + sgn * k * u
        if arm.ik(cand, zt - 4) and arm.ik(cand, Z_HOVER) and not arm.post(cand, Z_HOVER) and arm.leg_ok(arm.park, cand):
            p = cand
            break
    if p is None:
        return False
    e = p + sgn * 35.0 * u
    if not (arm.ik(e, zt - 4) and arm.ik(e, Z_HOVER)) or any(arm.post(p + t * (e - p), Z_HOVER) for t in np.linspace(0, 1, 8)) \
            or not arm.peer_ok([e]):
        return False
    print("   %s 臂把棋子 %s（半徑 %.0f）壓在 %s 往%s拖 35 mm" % (n, c.round(1), r, p.round(1), "外" if sgn > 0 else "內"), flush=True)
    peer = sc.other(arm)
    if dist(peer.arm.pose()[:2], peer.park) > 5.0:     # 跟 deliver 一樣：另一台先回停放點，碰撞檢查才成立
        peer.go_park()
    sc.check()
    arm.goto(p)
    zc = arm.touch(p, zt, vacuum_on=True, over=OVER, below=4.0, z_surf=zt - PIECE)
    if zc is not None:
        try:
            arm.arm.dash.SpeedFactor(4)
            sync_move(arm.arm, "MovL(%f,%f,%f,%f)" % (e[0], e[1], zc, arm.r))
        finally:
            arm.arm.dash.SpeedFactor(mover.TRAVEL_SPEED)
    arm.arm.dash.sendRecvMsg("DOExecute(1,0)")
    time.sleep(0.6)
    lift_off(arm)
    arm.go_park()
    return zc is not None


def tidy(sc, kind, rounds=12):
    """把棋盤上紀錄是 kind 的棋子（和轉送點上留著的 kind）收回 kind 的料疊。負責的手臂搆不到：另一台先放到轉送點；
    再不行就拖近。疊在別顆上的先收。回傳 (收完了沒, 原因)。"""
    own = sc.arm_for(kind)
    oth = sc.other(own)
    moved = None                                      # 上一輪從棋盤上拿走的那顆原本的位置（B）：看不到了才刪紀錄
    unk0 = []                                         # 那一輪搬之前棋盤上沒紀錄的棋子
    for _ in range(rounds):
        img, piles = sc.measure()
        bp = sc.board_pieces(img)
        if moved is not None and nearest(bp, B_of, moved, TRACK) is None:
            sc.unmark(moved)
            new = [q for q in sc.board_pieces(img) if q["kind"] == "?" and all(dist(B_of(q), u) >= TRACK for u in unk0)]
            if new:                                    # 搬的時候掉回棋盤上：種類是知道的，記回去再收
                print("   剛剛那顆 %s 掉在棋盤上 B %s：記回紀錄" % (kind, B_of(new[0]).round(1)), flush=True)
                sc.mark(B_of(new[0]), kind)
            bp = sc.board_pieces(img)
        moved = None
        unk0 = [B_of(q) for q in bp if q["kind"] == "?"]
        st = sc.pile(piles, kind)
        hp = sc.hand_piece(piles) if sc.hand_kind() == kind else None
        mine = [q for q in bp if q["kind"] == kind]
        if hp is None and not mine:
            return True, None
        if st is not None:
            dst = sc.pile_spot(st, "%s 疊" % kind)
        else:                                          # 那疊全用完了（O 先下最多用 5 顆）：在固定位置的桌面上重新起一疊
            dst = sc.table_spot(np.asarray(sc.rec["piles"][kind]["B"], float), "%s 疊的固定位置（空的，重新起一疊）" % kind, piles)
            if dst is None:
                return False, "%s 疊的固定位置上有別的東西或量不到桌面高度" % kind
        if hp is not None:
            r = deliver(sc, sc.pile_spot(hp, "轉送點上的 %s" % kind), dst, [own])
            if r is None:
                return False, "%s 臂搆不到轉送點" % own.name
            sc.look(own)
            if not r[2]:
                return False, "沒碰到轉送點上的棋子"
            continue
        q = max(mine, key=lambda q: q["z_top"][own.name])          # 疊在上面的那顆先收
        src = sc.piece_spot(q)
        print("收 %s：棋盤上 B %s → %s 疊" % (kind, B_of(q).round(1), kind), flush=True)
        r = deliver(sc, src, dst, [own])
        if r is None:
            h = sc.hand_spot(piles)
            if h is not None and sc.shift(own, h, dst) is not None and sc.shift(oth, src, h) is not None:
                print("   %s 臂搆不到：%s 臂先把它放到轉送點" % (own.name, oth.name), flush=True)
                r = deliver(sc, src, h, [oth])
                if r[2]:
                    sc.set_hand(kind)                  # 一放開就記
                oth.go_park()
                if not r[2]:
                    return False, "沒碰到棋盤上的棋子"
                moved = B_of(q)
                continue
            if not drag_closer(sc, own, q):
                return False, "棋盤上 B %s 的 %s 兩台都吸不到、也拖不動" % (B_of(q).round(1), kind)
            continue
        sc.look(own)
        if not r[2]:
            return False, "沒碰到棋子、沒開真空"
        moved = B_of(q)
    return False, "收了 %d 輪還沒收完" % rounds


def clear(sc):
    """收棋盤：X、O 各自收回自己的料疊（照紀錄）。回傳 (清空了沒, 原因)。"""
    for kind in ("X", "O"):
        ok, why = tidy(sc, kind)
        if not ok:
            return False, why
    unk = [q for q in sc.board_pieces(sc.last[0]) if q["kind"] == "?"]
    if unk:
        return False, "棋盤上還有 %d 顆沒有紀錄（不知道是圈還是叉）的棋子，我不敢亂收" % len(unk)
    return True, None


def move_back(sc, q, kind, cell, piles):
    """棋盤上的 q（種類 kind，已知）搬回格 cell：先試直接（負責那種的手臂先），兩台都搆不到就經轉送點。
    最後用 place 確認、挪正（看不到就從料疊補）。回傳 place 的結果 dict。"""
    own = sc.arm_for(kind)
    oth = sc.other(own)
    src, dst = sc.piece_spot(q), sc.cell_spot(cell)
    old = B_of(q)
    r = deliver(sc, src, dst, [own, oth])
    if r is None:
        h = sc.hand_spot(piles)
        for a in (own, oth):
            b = sc.other(a)
            if h is not None and sc.shift(a, src, h) is not None and sc.shift(b, h, dst) is not None:
                print("   兩台都不能直接搬回格 %d：%s 臂先放到轉送點" % (cell, a.name), flush=True)
                r = deliver(sc, src, h, [a])
                if r[2]:
                    sc.unmark(old)
                    sc.set_hand(kind)                  # 接下來 place 會讓另一台從轉送點放進格子
                a.go_park()
                break
    elif r[2]:
        sc.unmark(old)
        sc.mark(sc.cellB[cell], kind)
        sc.look(r[0])
    if r is None:
        print("   B %s 那顆兩台都搬不動，改從 %s 疊補一顆" % (old.round(1), kind), flush=True)
    return place(sc, kind, cell)


def restore(sc, board):
    """人的回合盤面被弄亂（移走、拿走棋子）：照 board（伺服器記的盤面，"O"／"X"／" "）恢復，不靠人手（使用者 2026-11-15）。
    - 該有棋子的格上有一顆：沒紀錄（「?」）就補記成該有的種類（翻面看不到，只能這樣）。
    - 該有的不見了：多出來的棋子裡有紀錄是同種的 → 搬回去；沒紀錄的「?」剛好跟不見的一樣多 → 刪去法（同種的話任意配；
      不同種要只剩一顆才配）；都沒有（被拿到棋盤外）→ 從料疊補一顆。
    - 多出來、又分不出是哪一顆的不亂動，回報。回傳 (恢復好了沒, 說明)。"""
    img, piles = sc.measure()
    bp = sc.board_pieces(img)
    used, missing = [], []
    for i in range(9):
        k = board[i]
        if k == " ":
            continue
        q = sc.at_cell(img, i)
        if q is not None and q["kind"] in (k, "?"):
            used.append(id(q))
            if q["kind"] == "?":
                sc.mark(B_of(q), k)
        else:
            missing.append(i)
    extra = [q for q in bp if id(q) not in used]
    if not missing:
        return (False, "多了 %d 顆棋子，分不出是哪一顆，我不亂動" % len(extra)) if extra else (True, None)
    plan = []                                         # (格, 種類, 棋子 或 None＝從料疊補)
    unk = [q for q in extra if q["kind"] == "?"]
    left = []
    for i in missing:
        k = board[i]
        same = [q for q in extra if q["kind"] == k]
        if same:
            q = min(same, key=lambda q: dist(B_of(q), sc.cellB[i]))
            extra.remove(q)
            plan.append((i, k, q))
        else:
            left.append(i)
    kinds = {board[i] for i in left}
    if left and unk and (len(unk) == len(left) and (len(kinds) == 1 or len(left) == 1)):
        for i in left:
            q = min(unk, key=lambda q: dist(B_of(q), sc.cellB[i]))
            unk.remove(q)
            extra.remove(q)
            plan.append((i, board[i], q))
        left = []
    elif left and unk:
        return False, "%s不見了，棋盤上又多了 %d 顆分不出是哪一顆的棋子，我不亂動" % (
            "、".join("格 %d 的%s" % (i + 1, {"O": "圈", "X": "叉"}[board[i]]) for i in left), len(unk))
    plan += [(i, board[i], None) for i in left]
    for i, k, q in plan:
        sc.check()
        print("恢復格 %d 的 %s：%s" % (i, k, "從料疊補一顆" if q is None else "把 B %s 那顆搬回去" % B_of(q).round(1)), flush=True)
        r = place(sc, k, i) if q is None else move_back(sc, q, k, i, sc.last[1])
        if not r["ok"]:
            return False, "恢復格 %d 時%s" % (i + 1, r.get("why") or "沒成功")
    rest = [q for q in extra if q["kind"] == "?"]
    if rest:
        return False, "放回去了，可是棋盤上還多 %d 顆不知道是什麼的棋子，我不亂動" % len(rest)
    return True, None


def can_place(sc, kind, cell, piles):
    """kind 可不可以放到格 cell（自己直接放，或經轉送點）。只算不動，用最近一次量的料疊。"""
    own = sc.arm_for(kind)
    st = sc.pile(piles, kind)
    if st is None:
        return False
    src, dst = sc.pile_spot(st, kind), sc.cell_spot(cell)
    if sc.shift(own, src, dst) is not None:
        return True
    h = sc.hand_spot(piles, need_free=False)
    return h is not None and sc.shift(own, src, h) is not None and sc.shift(sc.other(own), h, dst) is not None
