import os, time, sys
sys.stdout.reconfigure(encoding="utf-8")
A = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1\Arena"
for i in range(6):
    now = time.time()
    f = lambda n: ("%.1f" % (now - os.path.getmtime(os.path.join(A, n)))) if os.path.exists(os.path.join(A, n)) else "無"
    print("彩色 %s 秒前、深度 %s 秒前、錯誤檔：%s" % (f("cam_full.jpg"), f("cam_depth.png"), open(os.path.join(A, "cam_error.txt"), encoding="utf-8").read()[:40] if os.path.exists(os.path.join(A, "cam_error.txt")) else "無"), flush=True)
    time.sleep(3)
