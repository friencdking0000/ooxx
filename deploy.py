"""現場一鍵部署（桌面「OOXX一鍵部署.bat」呼叫）。

    python deploy.py            # 停掉舊的 → 檢查兩台手臂 → 啟動對弈伺服器 → 等它好 → 開網頁 → 選模式
    python deploy.py --check    # 只看現況（不停、不啟動任何東西）

手臂只讀 30004 即時回饋（不送指令，不會跟伺服器搶 29999）。
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
ARENA = os.path.join(ROOT, "Arena")
sys.path.insert(0, os.path.join(ROOT, "MG400"))
import numpy as np                                  # noqa: E402
from dobot_api import MyType                        # noqa: E402
from dual_sync import ARMS                          # noqa: E402  （site.json 可覆蓋 IP）

MODES = {1: "初始化", 2: "抱閘鬆開", 3: "本體下電", 4: "未上電（正常，開局時會自己上電）", 5: "上電、空閒",
         6: "拖曳模式", 7: "運行中", 8: "軌跡錄製", 9: "報警", 10: "暫停", 11: "點動"}
PS = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"]


def say(t=""):
    print(t, flush=True)


def arm_status(name, ip):
    """讀一包 30004 回饋：(連得上嗎, 說明)。"""
    try:
        s = socket.create_connection((ip, 30004), timeout=3)
        s.settimeout(3)
        buf = b""
        while len(buf) < 1440:
            c = s.recv(1440 - len(buf))
            if not c:
                raise ConnectionError("連線被關")
            buf += c
        s.close()
        rec = np.frombuffer(buf, dtype=MyType)[0]
        mode = int(rec["robot_mode"])
        x, y, z = (float(v) for v in rec["tool_vector_actual"][:3])
        msg = "模式 %d %s；吸盤 (%.0f, %.0f, %.0f)" % (mode, MODES.get(mode, ""), x, y, z)
        if mode == 9:
            msg += "\n      → 有報警：先跑 MG400\\clear_error.py --ip=%s；清不掉（67／73）要把大臂往前推回限位內" % ip
        if mode == 6:
            msg += "\n      → 在拖曳模式：開局時會自動退出，沒關係"
        return True, msg
    except Exception as e:
        return False, "連不上 %s（%s）\n      → 檢查網路線、這台電腦的網卡 IP（要跟手臂同網段，例如 %s.100）、手臂有沒有開機" % (
            ip, str(e)[:60], ".".join(ip.split(".")[:3]))


def key():
    try:
        return open(os.path.join(ARENA, "access.txt")).read().strip()
    except OSError:
        return None


def state():
    k = key()
    try:
        return json.load(urllib.request.urlopen("http://127.0.0.1:18080/api/state?k=%s" % k, timeout=3))
    except Exception:
        return None


def cmd(c):
    req = urllib.request.Request("http://127.0.0.1:18080/api/cmd?k=%s" % key(), data=json.dumps({"cmd": c}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def lan_urls():
    try:
        L = open(os.path.join(ARENA, "out", "server_run.log"), encoding="utf-8", errors="replace").read().splitlines()
        return [l.strip() for l in L if l.strip().startswith("https://") and "127.0.0.1" not in l]
    except OSError:
        return []


def stop_all():
    """停掉舊的對弈伺服器、相機、教點工具、控制台（它們都會連手臂或相機，不能同時兩份）。"""
    subprocess.run(PS + ["Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match "
                         "'server\\.py|cam_worker\\.py|camera_worker\\.py|jog\\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"])
    time.sleep(2)
    # 用系統管理員或經 ssh 開的伺服器：這裡讀不到命令列、也殺不掉，上面會默默略過 → 用 18080 還有沒有人聽來確認
    try:
        socket.create_connection(("127.0.0.1", 18080), timeout=1).close()
    except OSError:
        return True
    pid = subprocess.run(PS + ["(Get-NetTCPConnection -State Listen -LocalPort 18080).OwningProcess"],
                         capture_output=True, text=True).stdout.split()
    say("   對弈伺服器停不掉（PID %s 還在跑、占著相機）：它是用系統管理員權限或經 ssh 開的，這個視窗沒權限關。\n"
        "   → 對桌面「停止OOXX.bat」按右鍵「以系統管理員身分執行」，或回到開它的視窗／ssh 關掉，再重跑一次。"
        % "、".join(pid or ["?"]))
    return False


def start_console_watch():
    """手臂監控網頁（Console 只看不控，http://127.0.0.1:8800）。"""
    py = sys.executable
    subprocess.run(PS + ["Start-Process -FilePath '%s' -ArgumentList '-u','%s','--watch' -WorkingDirectory '%s' -WindowStyle Hidden "
                         "-RedirectStandardOutput '%s' -RedirectStandardError '%s'" % (
                             py, os.path.join(ROOT, "Console", "server.py"), os.path.join(ROOT, "Console"),
                             os.path.join(ROOT, "Console", "watch_out.log"), os.path.join(ROOT, "Console", "watch_err.log"))])


def main():
    if "--console" in sys.argv:                     # 只開手臂監控（桌面「手臂監控.bat」）
        subprocess.run(PS + ["Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match "
                             "'Console.server\\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"])
        start_console_watch()
        time.sleep(3)
        webbrowser.open("http://127.0.0.1:8800/")
        say("手臂監控（只看不控）：http://127.0.0.1:8800/")
        return
    check = "--check" in sys.argv
    say("=" * 60)
    say(" OOXX 對弈機器人：%s" % ("現況檢查" if check else "一鍵部署"))
    say(" 程式位置：%s" % ROOT)
    say("=" * 60)
    if not check:
        say("\n[1/5] 停掉舊的伺服器、相機程式")
        if not stop_all():
            input("按 Enter 結束")
            return
    say("\n[%s] 兩台手臂（只讀狀態，不會動）" % ("1/2" if check else "2/5"))
    ok_all = True
    for n, ip in ARMS.items():
        ok, msg = arm_status(n, ip)
        ok_all &= ok
        say("   %s 臂 %s：%s" % (n, "OK" if ok else "失敗", msg))
    if check:
        s = state()
        say("\n[2/2] 對弈伺服器：%s" % ("沒在跑" if s is None else "%s／%s／相機 %s" % (
            s.get("phase"), s.get("status"), "OK" if s.get("camera_ok") else "有問題")))
        return
    if not ok_all:
        say("\n有手臂連不上。照上面的提示處理後再執行一次；要先不管手臂、照樣啟動，按 Enter。")
        input()
    say("\n[3/5] 啟動對弈伺服器（雙臂）")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(ARENA, "start_server.ps1")])
    say("\n[4/5] 等伺服器、相機準備好（最多 90 秒）")
    s, t0 = None, time.time()
    while time.time() - t0 < 90:
        s = state()
        if s is not None and s.get("camera_ok"):
            break
        time.sleep(2)
        print(".", end="", flush=True)
    say()
    if s is None:
        say("   伺服器沒起來：看 %s" % os.path.join(ARENA, "out", "server_err.log"))
        input("按 Enter 結束")
        return
    say("   伺服器 OK；相機 %s；手臂 %s" % ("OK" if s.get("camera_ok") else "還沒好（相機程式會自己重試，USB 要插 3.0 的孔）",
                                         s.get("arm_error") or "OK"))
    say("\n[5/5] 開網頁")
    k = key()
    webbrowser.open("http://127.0.0.1:18080/?k=%s" % k)
    start_console_watch()
    time.sleep(3)
    webbrowser.open("http://127.0.0.1:8800/")
    say("   對弈網頁（這台）：http://127.0.0.1:18080/?k=%s" % k)
    for u in lan_urls():
        say("   手機／平板（同一個網路，第一次要按「繼續前往」）：%s" % u)
    say("   手臂監控（只看不控）：http://127.0.0.1:8800/")
    say("\n要做什麼？")
    say("   1  人機對弈（網頁按「新局」開始；棋盤上有棋子會先自己收）")
    say("   2  機器自己對下，一盤接一盤（直播用；網頁按「停止」才停）")
    say("   直接按 Enter：什麼都不做，網頁上自己操作")
    c = input("> ").strip()
    if c == "2":
        say("   %s" % cmd("selfplay_loop"))
    elif c == "1":
        say("   在網頁按「新局」（或「我先」讓機器先下）。")
    say("\n完成。這個視窗可以關掉（伺服器在背景繼續跑；要停就用桌面的「停止OOXX.bat」）。")
    input("按 Enter 關閉")


if __name__ == "__main__":
    main()
