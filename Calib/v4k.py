"""V4K Pro 側面監看：拍一張存 out/v4k_<名稱>.png（縮成 1088x816）。用法：python v4k.py <名稱>
以「最高解析度 3264x2448」辨認 V4K Pro（index 0 筆電鏡頭、Iriun 手機虛擬鏡頭都不會用到）。"""
import sys
import cv2


def snap(name, index=2):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3264)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2448)
    if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != 3264:
        cap.release()
        raise RuntimeError("index %d 不是 V4K Pro（無 3264x2448）" % index)
    ok, f = False, None
    for _ in range(12):
        ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("V4K Pro 取像失敗")
    out = "out/v4k_%s.png" % name
    cv2.imwrite(out, cv2.resize(f, (1088, 816)))
    return out


if __name__ == "__main__":
    print(snap(sys.argv[1] if len(sys.argv) > 1 else "now"))
