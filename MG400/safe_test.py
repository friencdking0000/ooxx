"""MG400 實機安全測試：清報警 → 使能 → MovJ 到安全點 → 小範圍 MovL 來回 → 下使能。

和官方 main.py 的差別：
  - 每個指令都檢查回傳碼（main.py 使能失敗回 -1 也照樣往下跑）
  - 不用 input()／共用鎖，報警時直接停下並印出原因（main.py 在無互動終端會卡死）
  - 點位以本台手臂量過的姿態為準，先 MovJ 再 MovL（main.py 的範例點 MovL 會報 23 超出工作空間）
  - 預設只讀狀態、印出計畫；加 --go 才會使能與運動

用法（在本目錄下）：
    python safe_test.py          # 只讀狀態與計畫，不會動
    python safe_test.py --go     # 實際運動（確認周圍淨空、急停在手邊）
Ctrl+C 會送 ResetRobot()（停止）與 DisableRobot()。
"""
import re
import sys
import time

from dobot_api import DobotApiDashboard, DobotApiMove

IP = next((a[5:] for a in sys.argv[1:] if a.startswith("--ip=")), "192.168.1.6")  # 用法：--ip=192.168.2.7

# 安全點：在目前停放位置（約 X275, Y-35, Z121）附近，高度不往下降，避免撞到桌上的東西
SAFE_X, SAFE_Z = 270.0, 100.0
SWING_Y = 40.0          # MovL 在 Y=±40mm 之間來回
CYCLES = 3
SPEED = 20              # 全域速度比例 %
ARRIVE_TOL = 1.0        # mm／度
MOVE_TIMEOUT = 30.0     # 秒

MODE_NAMES = {1: "初始化", 2: "抱閘鬆開", 3: "本體下電", 4: "未使能", 5: "使能空閒",
              6: "拖曳", 7: "運行中", 8: "軌跡錄製", 9: "報警", 10: "暫停", 11: "點動"}


def code(reply):
    """回傳字串 "ErrorID,{...},Cmd();" 的 ErrorID。"""
    m = re.match(r"\s*(-?\d+)", reply or "")
    return int(m.group(1)) if m else None


def values(reply):
    m = re.search(r"\{([^}]*)\}", reply or "")
    return [float(v) for v in m.group(1).split(",") if v.strip()] if m else []


def check(reply, what):
    c = code(reply)
    if c != 0:
        raise RuntimeError("%s 失敗，回傳 %r" % (what, reply.strip()))
    return reply


def mode(dash):
    v = values(dash.RobotMode())
    return int(v[0]) if v else None


def errors(dash):
    return [int(n) for n in re.findall(r"-?\d+", dash.GetErrorID())[1:]]


def pose(dash):
    return values(dash.GetPose())[:4]


def wait_arrive(dash, target):
    t0 = time.time()
    while time.time() - t0 < MOVE_TIMEOUT:
        m = mode(dash)
        if m == 9:
            raise RuntimeError("運動中報警，錯誤碼 %s" % errors(dash))
        p = pose(dash)
        if m == 5 and all(abs(a - b) <= ARRIVE_TOL for a, b in zip(p, target)):
            return p
        time.sleep(0.1)
    raise RuntimeError("等待到位逾時（%.0fs），目前 %s" % (MOVE_TIMEOUT, pose(dash)))


def main(go):
    dash = DobotApiDashboard(IP, 29999)
    move = DobotApiMove(IP, 30003)
    try:
        m = mode(dash)
        p = pose(dash)
        print("\n目前模式 %s（%s），錯誤 %s，姿態 X%.1f Y%.1f Z%.1f R%.1f"
              % (m, MODE_NAMES.get(m, "?"), errors(dash), *p))

        r = p[3]                            # R 維持目前值，避免 J4 無謂轉動
        safe = [SAFE_X, 0.0, SAFE_Z, r]
        a = [SAFE_X, -SWING_Y, SAFE_Z, r]
        b = [SAFE_X, SWING_Y, SAFE_Z, r]
        print("計畫：速度 %d%%，MovJ → %s，再 MovL %s ↔ %s 共 %d 次，最後回安全點並下使能"
              % (SPEED, safe, a[:3], b[:3], CYCLES))
        if not go:
            print("（未加 --go，只讀狀態，不會動）")
            return

        if m == 9:
            print("清除報警 %s" % errors(dash))
            check(dash.ClearError(), "ClearError")
            time.sleep(0.5)
        check(dash.SpeedFactor(SPEED), "SpeedFactor")
        check(dash.EnableRobot(), "EnableRobot")
        time.sleep(1.0)
        if mode(dash) != 5:
            raise RuntimeError("使能後模式不是 5（使能空閒），而是 %s" % mode(dash))

        check(move.MovJ(*safe), "MovJ 安全點")
        print("到位", wait_arrive(dash, safe))
        for i in range(CYCLES):
            for pt in (a, b):
                check(move.MovL(*pt), "MovL")
                print("第 %d 次到位" % (i + 1), wait_arrive(dash, pt))
        check(move.MovL(*safe), "MovL 回安全點")
        print("回安全點", wait_arrive(dash, safe))
        print("\n完成 ✓")
    except KeyboardInterrupt:
        print("\n中斷：停止手臂")
        dash.ResetRobot()
    except Exception as e:
        print("\n停止：%s" % e)
        dash.ResetRobot()
    finally:
        if go:
            dash.DisableRobot()
        dash.close()
        move.close()


if __name__ == "__main__":
    main("--go" in sys.argv)
