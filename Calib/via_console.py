"""透過控制台 API 動手臂（控制台開著時，不要另外連 29999／30003，會互搶）。"""
import json, time, urllib.request
API = "http://127.0.0.1:8800"


def cmd(arm, c, **args):
    req = urllib.request.Request(API + "/api/cmd", data=json.dumps({"arm": arm, "cmd": c, "args": args}).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=20))[arm]
    print("  %s %-8s %s → %s" % (arm, c, {k: round(v, 1) for k, v in args.items()}, r["reply"]))
    if not r["ok"]:
        raise SystemExit("停止：%s 失敗" % c)
    return r


def state(arm):
    return json.load(urllib.request.urlopen(API + "/api/state", timeout=5))["arms"][arm]


def wait_at(arm, x, y, z, tol=0.5, timeout=40):
    t0, stable = time.time(), None
    while time.time() - t0 < timeout:
        s = state(arm)
        p = s.get("pose") or [1e9] * 4
        if s.get("mode") == 9:
            raise SystemExit("%s 報警 %s" % (arm, s.get("errors")))
        if s.get("mode") == 5 and abs(p[0] - x) < tol and abs(p[1] - y) < tol and abs(p[2] - z) < tol:
            stable = stable or time.time()
            if time.time() - stable > 0.3:
                return p
        else:
            stable = None
        time.sleep(0.1)
    raise TimeoutError("等待到位逾時，目前 %s" % state(arm).get("pose"))
