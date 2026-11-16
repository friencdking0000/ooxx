"""單台手臂依序執行運動步驟（每步先做 IK 檢查；任一步失敗即停止）。

    python move_arm.py A "L:204.5,39,60" "J:60,340,60" --speed=10       # R 不給就沿用目前 R
步驟：L:x,y,z[,r] = MovL；J:x,y,z[,r] = MovJ；可加 --keep 結束後不下使能（預設下使能）。
"""
import sys

from common import connect_arm, enable, sync_move


def main():
    name = sys.argv[1]
    steps = [a for a in sys.argv[2:] if a[:2] in ("L:", "J:")]
    speed = int(next((a[8:] for a in sys.argv if a.startswith("--speed=")), "10"))
    arm = connect_arm(name)
    try:
        arm.wait_feed()
        r0 = float(arm.pose()[3])
        plan = []
        for s in steps:
            v = [float(x) for x in s[2:].split(",")]
            if len(v) == 3:
                v.append(r0)
            cmd = "MovL" if s[0] == "L" else "MovJ"
            if not arm.dash.InverseSolution(*v, 0, 0).startswith("0,"):
                raise RuntimeError("IK 不可達：%s %s" % (cmd, v))
            plan.append((cmd, v))
        enable(arm, speed=speed)
        for cmd, v in plan:
            sync_move(arm, "%s(%f,%f,%f,%f)" % (cmd, *v))
            print("%s %s → 到位 %s" % (cmd, v, arm.pose().round(1)))
    except Exception as e:
        print("停止：", e)
        arm.dash.ResetRobot()
    finally:
        if "--keep" not in sys.argv:
            arm.dash.DisableRobot()
        arm.close()


if __name__ == "__main__":
    main()
