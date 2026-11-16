import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import cv2
for idx in range(5):
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    if not cap.isOpened():
        print(idx, "打不開"); continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, f = False, None
    for _ in range(5): ok, f = cap.read()
    print(idx, ok, None if f is None else f.shape, None if f is None else round(float(f.mean()), 1))
    if ok: cv2.imwrite(r"C:\Users\GIGABYTE\Desktop\cam_%d.jpg" % idx, f)
    cap.release()
