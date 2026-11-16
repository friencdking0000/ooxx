"""清除 MG400 報警（ClearError）。只清報警，不會使能、不會讓手臂動。

⚠️ 實體急停按鈕被按下時，軟體清不掉——要先到現場把急停旋開。

    python clear_error.py
"""
import re
import sys
import time

from dobot_api import DobotApiDashboard

IP = next((a[5:] for a in sys.argv[1:] if a.startswith("--ip=")), "192.168.1.6")  # 用法：--ip=192.168.2.7


def snapshot(d):
    mode = re.search(r"\{(\d+)\}", d.RobotMode()).group(1)
    errs = [int(n) for n in re.findall(r"-?\d+", d.GetErrorID())[1:]]
    return mode, errs


def main():
    d = DobotApiDashboard(IP, 29999)
    try:
        mode, errs = snapshot(d)
        print("\n清除前：模式 %s，報警 %s" % (mode, errs))
        d.ClearError()
        time.sleep(1.0)
        mode, errs = snapshot(d)
        print("清除後：模式 %s，報警 %s" % (mode, errs))
        if errs:
            print("仍有報警：若是 85（急停），請先旋開實體急停按鈕再執行一次。")
    finally:
        d.close()


if __name__ == "__main__":
    main()
