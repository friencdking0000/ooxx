import os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
def shot(name):
    import cv2
    p = os.path.join(HERE, "out", name + ".png")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(HERE, "obs_shot.ps1"), p], timeout=30)
    im = cv2.imread(p); h, w = im.shape[:2]
    cv2.imwrite(os.path.join(HERE, "out", name + ".jpg"), cv2.resize(im[int(h*0.2):int(h*0.42), int(w*0.24):int(w*0.5)], None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))
