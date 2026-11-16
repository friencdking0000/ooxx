"""相機子程序（Gemini 2）：彩色＋深度一直開著，每一輪寫 cam_full.jpg（彩色）與 cam_depth.png（深度，有才寫）。

- 獨立子程序：Gemini 2 斷線時 Orbbec SDK 會直接結束整個程序，所以由 server.py 啟動、監看，掛了會重啟。
- 解析度由 common.OrbbecCamera 依連線自動決定（USB 3：彩色 1280x720；USB 2：640x360 放大回 1280x720）。
  不做「只開彩色／切到深度」的分時切換：2026-11-15 實測切換會讓相機卡到出幾張就停。
- 每張影像（彩色、深度）先套 cam_align.json 的對齊（相機被碰歪時由 server 的校正檢查算出來）。
- 深度不一定每張都有（手臂靠近相機時會停），彩色照常寫；彩色停了才算相機出問題。
- 出問題的處理順序：重開一次 → 叫相機重開機 → 結束程序讓 server 重啟。
- depth_request：server 要「這一刻之後的深度」時建立；寫完幾張深度後由這裡刪掉，server 看它消失就知道拿到了。
"""
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Calib"))

import align                                    # noqa: E402
from common import OrbbecCamera, reboot_device  # noqa: E402

OUT = os.path.join(HERE, "cam_full.jpg")
ERR = os.path.join(HERE, "cam_error.txt")
DEP = os.path.join(HERE, "cam_depth.png")      # 對齊到彩色、1280x720 的深度（16 位元 PNG，mm）
REQ = os.path.join(HERE, "depth_request")
REQ_FRAMES = 5                                  # 有人要深度時，寫滿這麼多張新的深度再刪掉 depth_request


def _atomic(path, buf):
    """先寫暫存檔再換名；server 正在讀時 Windows 會拒絕換名，稍等再試。"""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(buf)
    for _ in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.01)


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _err(msg):
    with open(ERR, "w", encoding="utf-8") as f:
        f.write(msg[:300])


def _open():
    """開相機；開不起來（或開了沒影格）就重試一次，再不行就叫相機重開機並結束程序。"""
    for k in range(2):
        try:
            return OrbbecCamera(depth=True)
        except Exception as e:
            last = e
            time.sleep(1.5)
    _err("相機開不起來（%s），相機重開機中" % str(last)[:120])
    try:
        reboot_device()
    except Exception:
        pass
    time.sleep(20)
    os._exit(1)


def main():
    cam = _open()
    _rm(ERR)
    M, m_t = align.load(), None
    n_req = 0
    while True:
        try:                                    # 對齊檔有更新就重讀；檔案壞掉就沿用上一個有效值
            mt = os.stat(align.FILE).st_mtime if os.path.exists(align.FILE) else 0.0
            if mt != m_t:
                M, m_t = align.load(strict=True), mt
        except (OSError, ValueError, KeyError):
            pass
        try:
            img, dep = cam.grab(skip=0, timeout=6, pair=True)
        except Exception:
            cam.close()                          # 彩色停了：重開一次，還是不行 _open() 會叫相機重開機
            time.sleep(1.0)
            cam = _open()
            continue
        ok, buf = cv2.imencode(".jpg", align.apply(M, img), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            _atomic(OUT, buf)
        if dep is not None:
            ok, buf = cv2.imencode(".png", align.apply(M, dep, depth=True).clip(0, 65535).astype("uint16"))
            if ok:
                _atomic(DEP, buf)
                if os.path.exists(REQ):
                    n_req += 1
                    if n_req >= REQ_FRAMES:
                        n_req = 0
                        _rm(REQ)
                else:
                    n_req = 0
        time.sleep(0.12)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _err(str(e))
        sys.exit(1)
