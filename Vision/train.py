"""訓練 O／X 棋子偵測（YOLOv8n）。要用有 GPU 版 PyTorch＋ultralytics 的環境：
    D:\anaconda3\envs\vision\python.exe train.py [epochs]
輸出 runs/ox/weights/best.pt，並轉成 ox.onnx（執行期用 OpenCV DNN 跑，不需要 PyTorch）。"""
import os
import shutil
import sys

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    m = YOLO(os.path.join(HERE, "work", "yolov8n.pt"))
    m.train(data=os.path.join(HERE, "dataset", "data.yaml"), epochs=ep, imgsz=640, batch=16, workers=2, device=0,
            project=os.path.join(HERE, "runs"), name="ox", exist_ok=True, patience=25, plots=False, verbose=False,
            degrees=180, flipud=0.5, fliplr=0.5, hsv_h=0.03, hsv_s=0.6, hsv_v=0.5, scale=0.3, translate=0.1, mosaic=1.0)
    best = YOLO(os.path.join(HERE, "runs", "ox", "weights", "best.pt"))
    print("VAL", best.val(data=os.path.join(HERE, "dataset", "data.yaml"), imgsz=640, device=0, plots=False, verbose=False).results_dict)
    onnx = best.export(format="onnx", imgsz=(384, 640), opset=12, simplify=False)
    shutil.copy(onnx, os.path.join(HERE, "ox.onnx"))
    print("ONNX", os.path.join(HERE, "ox.onnx"))
