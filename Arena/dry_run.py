"""乾跑自動測試：對 `python server.py --dry` 打 API，跑完幾種情境並檢查狀態是否合理。

    python dry_run.py            # 全部情境
    python dry_run.py game1      # 只跑某個情境
"""
import json
import os
import ssl
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Calib"))
from game import best_move, winner  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://127.0.0.1:8443"
K = open(os.path.join(HERE, "access.txt")).read().strip()
CTX = ssl._create_unverified_context()
FAILS = []


def call(path, body=None):
    url = BASE + path + ("&" if "?" in path else "?") + "k=" + K
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, context=CTX, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def st():
    return call("/api/state")


def cmd(c, **kw):
    return call("/api/cmd", dict(cmd=c, **kw))


def say(text):
    call("/api/say", {"text": text})


def check(cond, what):
    print(("  ✓ " if cond else "  ✗ ") + what)
    if not cond:
        FAILS.append(what)


def wait(pred, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = st()
        if pred(s):
            return s, time.time() - t0
        time.sleep(0.3)
    s = st()
    check(False, "%s（%ds 內沒等到；phase=%s status=%s）" % (what, timeout, s["phase"], s["status"]))
    return s, None


def last_robot_msgs(n=3):
    return [m["text"] for m in st()["messages"] if m["role"] == "robot"][-n:]


def fresh(robot_first=False):
    cmd("sim_clear")
    if st()["phase"] == "paused":
        pass
    r = cmd("new_game", robot_first=robot_first)
    check("新局開始" in r["reply"], "開新局：%s" % r["reply"])


def play_out(human_policy, label):
    """人照 human_policy 下，直到結束。回傳最後狀態。"""
    for _ in range(9):
        s, dt = wait(lambda s: s["phase"] in ("human", "over", "paused"), 40, "%s：等輪到人或結束" % label)
        if s["phase"] != "human":
            break
        cell = human_policy(s)
        t0 = time.time()
        cmd("sim_tap", cell=cell)
        s2, dt = wait(lambda s: s["board"][cell] == "X", 15, "%s：人下格 %d 被偵測到" % (label, cell))
        if dt is not None:
            print("    人下格 %d → %.1f s 後偵測到" % (cell, time.time() - t0))
    s = st()
    print("    終局：%s  result=%s  status=%s" % ("".join(c if c != " " else "·" for c in s["board"]), s["result"], s["status"]))
    return s


def human_best(s):
    free = [i for i in range(9) if s["board"][i] == " "]
    c, _ = best_move(s["board"], "X", allowed=set(free))
    return c


def sanity(s, label):
    b = s["board"]
    check(b.count("O") <= b.count("X") + 1 and b.count("X") <= b.count("O") + 1, "%s：棋子數合理（X%d O%d）" % (label, b.count("X"), b.count("O")))
    o_cells = [i for i in range(9) if b[i] == "O"]
    check(all(i in s["robot_cells"] for i in o_cells), "%s：機器只下在搆得到的格 %s" % (label, o_cells))
    if s["phase"] == "over":
        w = winner(b)
        exp = w or ("draw" if " " not in b else "stuck")
        check(s["result"] == exp, "%s：結果判定 %s（應為 %s）" % (label, s["result"], exp))


def game1():
    print("\n[game1] 人先、人走最佳步")
    fresh(False)
    sanity(play_out(human_best, "game1"), "game1")


def game2():
    print("\n[game2] 機器先、人走最佳步")
    fresh(True)
    sanity(play_out(human_best, "game2"), "game2")


def game3():
    print("\n[game3] 人專攻機器搆不到的格（3、6）")
    fresh(False)
    order = [6, 0, 3, 8, 4, 2, 5, 1, 7]

    def pol(s):
        return next(i for i in order if s["board"][i] == " ")
    s = play_out(pol, "game3")
    sanity(s, "game3")


def guards():
    print("\n[guards] 防呆")
    cmd("sim_clear")
    fresh(False)
    cmd("sim_tap", cell=4)
    wait(lambda s: s["phase"] == "robot" or s["board"][4] == "X", 15, "人下中間")
    r = cmd("new_game")
    check("清空" in r["reply"] or "棋子" in r["reply"], "棋盤有子時拒開新局：%s" % r["reply"])
    wait(lambda s: s["phase"] == "human", 30, "等機器下完")
    # 人一次擺兩顆（作弊／誤觸）→ 不應被接受
    s = st()
    free = [i for i in range(9) if s["board"][i] == " "][:2]
    cmd("sim_tap", cell=free[0]); cmd("sim_tap", cell=free[1])
    time.sleep(12)
    s = st()
    check(s["board"][free[0]] == " " and s["board"][free[1]] == " " and s["phase"] == "human",
          "一次多兩顆 X 時不認定（等人修正）")
    r = cmd("hint")
    check("建議" in r["reply"], "提示：%s" % r["reply"])


def stop_resume():
    print("\n[stop_resume] 機器動作中按停止，再繼續")
    fresh(True)
    time.sleep(2.5)                                  # 模擬手臂約 7.6 s，停在中途
    r = cmd("stop")
    s = st()
    check(s["phase"] == "paused", "停止後 phase=paused（%s）" % r["reply"])
    time.sleep(1.5)
    check(s["board"].count("O") == 0, "中途停止沒有記成已下子")
    errs = [t for t in last_robot_msgs(4) if "出了問題" in t]
    check(not errs, "停止不會再多報一次錯誤")
    r = cmd("resume")
    check("繼續" in r["reply"], "繼續：%s" % r["reply"])
    s, _ = wait(lambda s: s["board"].count("O") == 1 and s["phase"] == "human", 30, "繼續後機器把那步下完")
    r = cmd("resume")
    check("沒有暫停" in r["reply"], "沒暫停時按繼續：%s" % r["reply"])


def voice():
    print("\n[voice] 文字意圖（走 LLM）")
    fresh(False)
    say("幫我下中間")
    s, _ = wait(lambda s: s["board"][4] == "X", 40, "「幫我下中間」→ 手臂代放 X 到中間")
    wait(lambda s: s["phase"] == "human", 40, "代放後換機器下完")
    say("我該下哪裡")
    time.sleep(6)
    check(any("建議" in t for t in last_robot_msgs(3)), "「我該下哪裡」→ 給建議：%s" % last_robot_msgs(1))
    say("左中幫我放")                                # 機器搆不到 3
    time.sleep(6)
    check(any("搆不到" in t for t in last_robot_msgs(3)) or st()["board"][3] == " ",
          "「左中幫我放」→ 說搆不到：%s" % last_robot_msgs(1))
    say("停")
    time.sleep(1.5)
    check(st()["phase"] == "paused", "「停」→ 暫停")
    say("繼續")
    time.sleep(6)
    check(st()["phase"] in ("human", "robot"), "「繼續」→ 恢復（phase=%s）" % st()["phase"])
    say("你好啊 你叫什麼名字")
    time.sleep(6)
    print("    閒聊回覆：", last_robot_msgs(1))


def tts_check():
    print("\n[tts] 語音檔")
    time.sleep(4)
    ms = [m for m in st()["messages"] if m["role"] == "robot"]
    with_audio = [m for m in ms if m["audio"]]
    failed = [m["text"] for m in ms if "語音合成失敗" in m["text"]]
    check(not failed, "沒有語音合成失敗（%s）" % failed[:1])
    print("    機器訊息 %d 則，已有語音 %d 則" % (len(ms), len(with_audio)))


ALL = {"game1": game1, "game2": game2, "game3": game3, "guards": guards, "stop_resume": stop_resume,
       "voice": voice, "tts": tts_check}

if __name__ == "__main__":
    names = sys.argv[1:] or list(ALL)
    t0 = time.time()
    for n in names:
        try:
            ALL[n]()
        except Exception as e:
            check(False, "%s 例外：%r" % (n, e))
    print("\n共 %.0f s；%s" % (time.time() - t0, "全部通過" if not FAILS else "失敗 %d 項：\n- " % len(FAILS) + "\n- ".join(FAILS)))
