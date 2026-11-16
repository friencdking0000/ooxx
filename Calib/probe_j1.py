"""探測：單台手臂低速轉 J1，每個角度拍一張相機圖，看手臂在畫面中掃過哪裡。

    python probe_j1.py A                 # 只讀狀態、不動
    python probe_j1.py A --go --d=-20,0,20
J1 以「相對目前角度」的度數給定；J2/J3/J4 維持不動（高度不變）。結束回到原角度並下使能。
"""
import sys
import time

import cv2

from common import OUT, Camera, connect_arm, enable, joint_angles, sync_move


def main():
    name = sys.argv[1]
    go = "--go" in sys.argv
    deltas = [float(v) for v in next((a[4:] for a in sys.argv if a.startswith("--d=")), "-20,0,20").split(",")]
    arm = connect_arm(name)
    cam = Camera()
    try:
        arm.wait_feed()
        j0 = joint_angles(arm)
        print("%s 目前姿態 %s 關節 %s" % (name, arm.pose().round(1), j0))
        cv2.imwrite("%s/probe_%s_base.png" % (OUT, name), cam.grab(n=3))
        if not go:
            print("（未加 --go，不動）計畫 J1 delta：", deltas)
            return
        enable(arm, speed=10)
        for d in deltas:
            j = [j0[0] + d, j0[1], j0[2], j0[3]]
            sync_move(arm, "JointMovJ(%f,%f,%f,%f)" % tuple(j))
            time.sleep(0.6)
            f = "%s/probe_%s_j1_%+d.png" % (OUT, name, int(d))
            cv2.imwrite(f, cam.grab(n=3))
            print("J1 %+.0f° → 姿態 %s，存 %s" % (d, arm.pose().round(1), f))
        sync_move(arm, "JointMovJ(%f,%f,%f,%f)" % tuple(j0))
        print("已回原位")
    except Exception as e:
        print("停止：", e)
        arm.dash.ResetRobot()
    finally:
        if go:
            arm.dash.DisableRobot()
        cam.close()
        arm.close()


if __name__ == "__main__":
    main()
