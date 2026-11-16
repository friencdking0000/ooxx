"""相機子程序：只讀 Gemini 2（Orbbec SDK），把最新一張 JPEG 原子寫入 cam_latest.jpg。

獨立成子程序的原因：Gemini 2 斷線時 Orbbec SDK 會直接結束整個 Python 程序（實測
"Device is deactivated/disconnected"），不能讓它拖垮手臂控制台。由 server.py 啟動與監看，掛了就重啟。
"""
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Calib"))
OUT = os.path.join(HERE, "cam_latest.jpg")
ERR = os.path.join(HERE, "cam_error.txt")


def main():
    from common import OrbbecCamera
    cam = OrbbecCamera()
    try:
        os.remove(ERR)
    except OSError:
        pass
    while True:
        img = cam.grab(skip=0, timeout=5)
        ok, buf = cv2.imencode(".jpg", cv2.resize(img, (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            tmp = OUT + ".tmp"
            with open(tmp, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp, OUT)
        time.sleep(0.3)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        with open(ERR, "w", encoding="utf-8") as f:
            f.write(str(e)[:300])
        sys.exit(1)
