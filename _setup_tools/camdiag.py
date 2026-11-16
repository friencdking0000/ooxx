import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1"
sys.path.insert(0, os.path.join(R, "Calib")); os.chdir(os.path.join(R, "Calib"))
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, AlignFilter, OBStreamType
p = Pipeline()
info = p.get_device().get_device_info()
print("連線：", info.get_connection_type(), "韌體", info.get_firmware_version(), flush=True)

def profiles(sensor):
    out = []
    pl = p.get_stream_profile_list(sensor)
    for i in range(pl.get_count()):
        v = pl.get_stream_profile_by_index(i).as_video_stream_profile()
        out.append((v.get_width(), v.get_height(), str(v.get_format()).split(".")[-1], v.get_fps()))
    return out
cp, dp = profiles(OBSensorType.COLOR_SENSOR), profiles(OBSensorType.DEPTH_SENSOR)
print("彩色 1280x720 可用：", sorted({(f, fps) for w, h, f, fps in cp if (w, h) == (1280, 720)}))
print("深度可用：", sorted({(w, h, f, fps) for w, h, f, fps in dp})[:14])

def run(name, color, depth, secs=20):
    cfg = Config()
    cl = p.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    if color == "default": cfg.enable_stream(cl.get_default_video_stream_profile())
    else: cfg.enable_stream(cl.get_video_stream_profile(*color))
    if depth:
        dl = p.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth == "default": cfg.enable_stream(dl.get_default_video_stream_profile())
        else: cfg.enable_stream(dl.get_video_stream_profile(*depth))
    try:
        p.start(cfg)
    except Exception as e:
        print("%-34s 開不起來：%s" % (name, str(e)[:60]), flush=True); return
    t0 = time.time(); last = t0; gaps = []; n = nd = 0
    while time.time() - t0 < secs:
        fs = p.wait_for_frames(1000)
        if fs is None: continue
        c = fs.get_color_frame()
        if c is None: continue
        if depth and fs.get_depth_frame() is not None: nd += 1
        now = time.time(); gaps.append(now - last); last = now; n += 1
    gaps.append(time.time() - last)
    p.stop(); time.sleep(1.5)
    g = np.array(gaps[1:]) if len(gaps) > 1 else np.array([secs])
    print("%-34s %4d 張（%.1f fps）深度 %4d 張；最長停頓 %.1f 秒；停頓超過 1 秒 %d 次" % (name, n, n / secs, nd, g.max(), int((g > 1.0).sum())), flush=True)

run("彩色 預設（單獨）", "default", None)
run("彩色 1280x720 MJPG 15（單獨）", (1280, 720, OBFormat.MJPG, 15), None)
run("彩色預設＋深度預設", "default", "default")
run("彩色720p MJPG15＋深度640x400 Y16 15", (1280, 720, OBFormat.MJPG, 15), (640, 400, OBFormat.Y16, 15))
run("彩色720p MJPG10＋深度640x400 RLE 10", (1280, 720, OBFormat.MJPG, 10), (640, 400, OBFormat.RLE, 10))
