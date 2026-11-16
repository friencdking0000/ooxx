"""對伺服器下指令／看狀態（走本機 HTTP 18080）。
    python ctl.py                 # 只看狀態
    python ctl.py selfplay        # 機器對下一盤
    python ctl.py selfplay_loop   # 一盤接一盤（直播用），按 stop 才停
    python ctl.py loop_off        # 這盤下完、收好就停（不打斷正在下的；要更新程式、重啟伺服器時用）
    python ctl.py new_game        # 人機對弈新局（棋盤有子會先自己收）
    python ctl.py reset           # 重置：中斷這盤，手上那一步做完就收好棋盤，回到開局前（網頁「↺ 重置」）
    python ctl.py stop | resume
"""
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
K = open(os.path.join(HERE, "access.txt")).read().strip()


def call(path, body=None):
    req = urllib.request.Request("http://127.0.0.1:18080" + path + "?k=" + K,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        r = call("/api/cmd", {"cmd": sys.argv[1]})
        print("cmd ->", {k: v for k, v in r.items() if k in ("ok", "error", "msg", "say")})
    s = call("/api/state")
    print({k: s.get(k) for k in ("phase", "status", "board", "seen", "result", "arm_error", "camera_ok")})
    for m in s.get("messages", [])[-5:]:
        print("  ", m.get("text"))
