import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
p = Pipeline()
d = p.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
dd = p.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
print("預設彩色：", d.get_width(), d.get_height(), str(d.get_format()).split(".")[-1], d.get_fps(), "｜預設深度：", dd.get_width(), dd.get_height(), str(dd.get_format()).split(".")[-1], dd.get_fps(), flush=True)
def run(name, color, depth, secs=25):
    cfg = Config()
    cl = p.get_stream_profile_list(OBSensorType.COLOR_SENSOR); dl = p.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    try:
        cfg.enable_stream(cl.get_video_stream_profile(*color)); cfg.enable_stream(dl.get_video_stream_profile(*depth)); p.start(cfg)
    except Exception as e:
        print("%-40s 開不起來：%s" % (name, str(e)[:50]), flush=True); return
    t0 = time.time(); last = t0; gaps = []; n = 0
    while time.time() - t0 < secs:
        fs = p.wait_for_frames(1000)
        if fs is None or fs.get_color_frame() is None or fs.get_depth_frame() is None: continue
        now = time.time(); gaps.append(now - last); last = now; n += 1
    gaps.append(time.time() - last); p.stop(); time.sleep(1.5)
    g = np.array(gaps[1:]) if len(gaps) > 1 else np.array([secs])
    print("%-40s %4d 張（%.1f fps）；最長停頓 %.1f 秒；超過 1 秒 %d 次" % (name, n, n / secs, g.max(), int((g > 1.0).sum())), flush=True)
M, R, Y = OBFormat.MJPG, OBFormat.RLE, OBFormat.Y16
run("720p MJPG 5 ＋ 640x400 RLE 5", (1280, 720, M, 5), (640, 400, R, 5))
run("720p MJPG 10 ＋ 320x200 RLE 10", (1280, 720, M, 10), (320, 200, R, 10))
run("720p MJPG 5 ＋ 320x200 RLE 5", (1280, 720, M, 5), (320, 200, R, 5))
run("640x360 MJPG 10 ＋ 640x400 RLE 10", (640, 360, M, 10), (640, 400, R, 10))
run("預設＋預設（再測一次）", (d.get_width(), d.get_height(), d.get_format(), d.get_fps()), (dd.get_width(), dd.get_height(), dd.get_format(), dd.get_fps()))
