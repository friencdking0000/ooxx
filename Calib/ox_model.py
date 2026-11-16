"""訓練好的 O／X 棋子偵測（YOLOv8n → Vision/ox.onnx），用 OpenCV DNN 跑，執行期不需要 PyTorch。
現場光線變了不用再調顏色門檻。模型檔不存在或載入失敗時 available() 回 False，呼叫端退回傳統顏色判斷。

    from ox_model import detect, kind_at
    detect(img)            -> [(u, v, r, "O"|"X", conf), ...]   影像座標（1280x720）
    kind_at(img, u, v)     -> "O"／"X"／None（附近沒有偵測到棋子）
    python ox_model.py [影像]   # 存標註圖 out/ox_model.jpg
環境變數 OX_MODEL=0 可關掉（強制用傳統方法）。"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(os.path.dirname(HERE), "Vision", "ox.onnx")
IN_W, IN_H = 640, 384
NAMES = ("O", "X")
CONF = 0.35
RAW_CONF = 0.12           # 內部保留到這麼低；呼叫端各自用 conf 篩（料疊已由深度確認是棋子時，可以信低一點的分數）
_net = None
_fail = False
_cache = (None, None)
import threading as _threading
_LOCK = _threading.Lock()            # cv2.dnn.Net 不能多執行緒同時用（伺服器主迴圈、校正檢查、網頁請求會同時讀盤）


def available():
    global _net, _fail
    if _fail or os.environ.get("OX_MODEL") == "0":
        return False
    if _net is None:
        sys.path.insert(0, HERE)
        import piece_profile
        prof = piece_profile.load()
        if prof is not None and not prof.get("use_model", True):
            _fail = True                     # 棋子校正時模型認錯了現場這批棋子：不用模型
            print("[ox_model] 棋子校正說模型認不得這批棋子，改用校正學到的樣子／顏色判斷", flush=True)
            return False
    if _net is None:
        try:
            _net = cv2.dnn.readNetFromONNX(ONNX)
        except Exception as e:
            _fail = True
            print("[ox_model] 模型載不起來，改用傳統顏色判斷：%s" % str(e)[:80], flush=True)
            return False
    return True


def detect(img, conf=CONF):
    """同一張影像（同一個陣列物件）重複呼叫只推論一次。"""
    global _cache
    if not available():
        return []
    key = (img.shape, int(img[::97, ::89].sum()), int(img[50::61, 30::53].sum()))   # 同一張影像（就算是重新讀檔的新陣列）只推論一次
    with _LOCK:
        if _cache[0] == key:
            return [r for r in _cache[1] if r[4] >= conf]
        res = _infer(img)
        _cache = (key, res)
    return [r for r in res if r[4] >= conf]


def _infer(img):
    H, W = img.shape[:2]
    s = min(IN_W / W, IN_H / H)
    nw, nh = int(round(W * s)), int(round(H * s))
    pad = np.full((IN_H, IN_W, 3), 114, np.uint8)
    px, py = (IN_W - nw) // 2, (IN_H - nh) // 2
    pad[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    _net.setInput(cv2.dnn.blobFromImage(pad, 1 / 255.0, (IN_W, IN_H), swapRB=True))
    out = _net.forward()[0].T                                   # (N, 4+類別數)：cx, cy, w, h, 各類分數
    sc = out[:, 4:]
    cls = sc.argmax(1)
    cf = sc.max(1)
    keep = cf >= RAW_CONF
    boxes = [[float(x - w / 2), float(y - h / 2), float(w), float(h)] for x, y, w, h in out[keep, :4]]
    idx = cv2.dnn.NMSBoxes(boxes, cf[keep].tolist(), RAW_CONF, 0.5)
    res = []
    for i in np.array(idx).reshape(-1):
        x, y, w, h = out[keep][i, :4]
        res.append(((x - px) / s, (y - py) / s, (w + h) / 4 / s, NAMES[int(cls[keep][i])], float(cf[keep][i])))
    return res


def kind_at(img, u, v, max_px=32.0, conf=CONF):
    best = None
    for du, dv, r, k, c in detect(img, conf):
        d = np.hypot(du - u, dv - v)
        if d <= max_px and (best is None or d < best[0]):
            best = (d, k)
    return None if best is None else best[1]


if __name__ == "__main__":
    f = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), "Arena", "cam_full.jpg")
    im = cv2.imread(f)
    for u, v, r, k, c in detect(im):
        print("%s (%.0f, %.0f) r=%.0f conf=%.2f" % (k, u, v, r, c))
        col = (255, 120, 0) if k == "O" else (0, 0, 255)
        cv2.circle(im, (int(u), int(v)), int(r), col, 3)
        cv2.putText(im, "%s %.2f" % (k, c), (int(u) - 40, int(v) - int(r) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    cv2.imwrite(os.path.join(HERE, "out", "ox_model.jpg"), im)
