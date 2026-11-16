"""讀取 MG400 狀態：模式、報警（含中文說明）、笛卡爾姿態、關節角度。不會讓手臂動。

    python read_state.py
"""
import re
import sys

from dobot_api import DobotApiDashboard, alarmAlarmJsonFile

IP = next((a[5:] for a in sys.argv[1:] if a.startswith("--ip=")), "192.168.1.6")  # 用法：--ip=192.168.2.7
MODE_NAMES = {1: "初始化", 2: "抱閘鬆開", 3: "本體下電", 4: "未使能", 5: "使能空閒",
              6: "拖曳", 7: "運行中", 8: "軌跡錄製", 9: "報警", 10: "暫停", 11: "點動"}


def values(reply):
    m = re.search(r"\{([^}]*)\}", reply or "")
    return [float(v) for v in m.group(1).split(",") if v.strip()] if m else []


def describe(err_id, controller, servo):
    for table, name in ((controller, "Controller"), (servo, "Servo")):
        for item in table:
            if item["id"] == err_id:
                return "%s %d：%s" % (name, err_id, item["zh_CN"]["description"])
    return "未知錯誤 %d" % err_id


def main():
    controller, servo = alarmAlarmJsonFile()
    d = DobotApiDashboard(IP, 29999)
    try:
        mode = int(values(d.RobotMode())[0])
        errs = [int(n) for n in re.findall(r"-?\d+", d.GetErrorID())[1:]]
        pose = values(d.GetPose())[:4]
        angle = values(d.GetAngle())[:4]
    finally:
        d.close()

    print("\n======== MG400 狀態 ========")
    print("模式：%d（%s）" % (mode, MODE_NAMES.get(mode, "?")))
    if errs:
        for e in errs:
            print("報警：" + describe(e, controller, servo))
    else:
        print("報警：無")
    print("姿態：X %.2f  Y %.2f  Z %.2f  R %.2f" % tuple(pose))
    print("關節：J1 %.2f°  J2 %.2f°  J3 %.2f°  J4 %.2f°" % tuple(angle))


if __name__ == "__main__":
    main()
