import os, time, sys
sys.stdout.reconfigure(encoding="utf-8")
A = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1\Arena"
worst = 0; n = 0; t0 = time.time(); last = None
while time.time() - t0 < 45:
    m = os.path.getmtime(os.path.join(A, "cam_full.jpg"))
    if m != last: n += 1; last = m
    worst = max(worst, time.time() - m)
    time.sleep(0.05)
print("45 秒內彩色更新 %d 次（%.1f 張/秒），最久沒更新 %.1f 秒；深度檔 %.1f 秒前" % (n, n / 45, worst, time.time() - os.path.getmtime(os.path.join(A, "cam_depth.png"))))
