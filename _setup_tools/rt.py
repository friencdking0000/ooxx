import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy, cv2, opencc, cryptography
print("numpy", numpy.__version__, "cv2", cv2.__version__)
try:
    import pyorbbecsdk
    print("pyorbbecsdk ok")
except Exception as e:
    print("pyorbbecsdk FAIL:", e)
sys.path.insert(0, r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1\Calib")
import os; os.chdir(r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1\Calib")
import ox_model, find_pieces, json
print("ox_model available:", ox_model.available())
