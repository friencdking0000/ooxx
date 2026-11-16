"""Orbbec Gemini 2 RGBD 取像測試：列出裝置 → 開彩色+深度串流 → 抓幾張 → 存 PNG 並印統計。

    python capture_test.py            # 存到 ./out/
依賴：pip install pyorbbecsdk2 numpy opencv-python
"""
import os
import sys
import time

import cv2
import numpy as np
from pyorbbecsdk import (Config, Context, OBAlignMode, OBFormat, OBSensorType,
                         Pipeline)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
WARMUP = 15      # 丟掉前幾幀（自動曝光穩定）
TIMEOUT_MS = 2000


def color_to_bgr(frame):
    w, h = frame.get_width(), frame.get_height()
    data = np.asanyarray(frame.get_data())
    fmt = frame.get_format()
    if fmt == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == OBFormat.RGB:
        return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if fmt == OBFormat.BGR:
        return data.reshape(h, w, 3)
    if fmt == OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    raise RuntimeError("不支援的彩色格式 %s" % fmt)


def depth_to_mm(frame):
    w, h = frame.get_width(), frame.get_height()
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(h, w)
    return (raw.astype(np.float32) * frame.get_depth_scale())   # mm


def main():
    ctx = Context()
    devs = ctx.query_devices()
    if devs.get_count() == 0:
        sys.exit("找不到 Orbbec 裝置（檢查 USB3 線／埠、其他程式如 Orbbec Viewer 是否占用）")
    info = devs.get_device_by_index(0).get_device_info()
    print("裝置：%s  SN=%s  FW=%s" % (info.get_name(), info.get_serial_number(),
                                      info.get_firmware_version()))

    pipe = Pipeline()
    cfg = Config()
    cp = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
    dp = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    print("彩色 profile：%dx%d@%d %s" % (cp.get_width(), cp.get_height(), cp.get_fps(), cp.get_format()))
    print("深度 profile：%dx%d@%d %s" % (dp.get_width(), dp.get_height(), dp.get_fps(), dp.get_format()))
    cfg.enable_stream(cp)
    cfg.enable_stream(dp)
    try:
        cfg.set_align_mode(OBAlignMode.SW_MODE)       # 深度對齊到彩色
    except Exception as e:
        print("對齊模式設定失敗（略過）：", e)

    os.makedirs(OUT, exist_ok=True)
    pipe.start(cfg)
    try:
        color = depth = None
        t0 = time.time()
        for i in range(WARMUP + 30):
            frames = pipe.wait_for_frames(TIMEOUT_MS)
            if frames is None:
                continue
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            if c is not None and d is not None and i >= WARMUP:
                color, depth = color_to_bgr(c), depth_to_mm(d)
                break
        if color is None:
            sys.exit("逾時：未取得同時含彩色與深度的影格")
        print("取像耗時 %.1fs" % (time.time() - t0))

        valid = depth[depth > 0]
        h, w = depth.shape
        print("彩色：%s  深度：%s" % (color.shape, depth.shape))
        print("深度有效像素 %.1f%%，範圍 %.0f–%.0f mm，中心點 %.0f mm"
              % (100.0 * valid.size / depth.size, valid.min() if valid.size else 0,
                 valid.max() if valid.size else 0, depth[h // 2, w // 2]))

        vis = cv2.applyColorMap(
            cv2.convertScaleAbs(np.clip(depth, 0, 4000), alpha=255.0 / 4000), cv2.COLORMAP_JET)
        vis[depth == 0] = 0
        cv2.imwrite(os.path.join(OUT, "color.png"), color)
        cv2.imwrite(os.path.join(OUT, "depth_vis.png"), vis)
        cv2.imwrite(os.path.join(OUT, "depth_mm.png"), depth.astype(np.uint16))
        print("已存到", OUT)
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
