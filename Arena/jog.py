"""簡易 XYZ 手動操作（網頁）：選手臂 → 上電 → 方向鍵一步一步移動 → 把吸盤帶到格子中心 → 「記錄格 N」。

    python jog.py            # 開 http://127.0.0.1:18090/

- 不能跟對弈伺服器（server.py）同時開：兩邊都會連手臂。啟動時發現對弈伺服器在跑就拒絕。
- 方向鍵是「相機畫面上的方向」（B 底座在畫面上方、A 在下方），程式自己換成各手臂的 X／Y。
- 每一步都先檢查：控制器 IK、相機支架禁入區、最低高度；速度 10%，真空不會開。
- 記錄的點存在 teach_cells.json（{手臂: {格: [X, Y, Z]}}），給現場校正用。
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Calib"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "MG400"))

import numpy as np                      # noqa: E402

import mover                            # noqa: E402
from common import atomic_write, enable, sync_move    # noqa: E402
from fit_teach import CROSS, TEACH_FILE as TEACH      # noqa: E402

PORT = 18090
Z_MIN = {"B": -160.0, "A": -157.0}      # 吸盤輕碰棋盤面附近；再低就壓進棋盤
Z_MAX = -10.0                           # 不能往上太高：方盒、真空管靠近相機與線材（曾經掃到相機）
POSE_MAX_AGE = 0.5                      # 秒：手臂回饋超過這麼久沒更新就不信（30004 曾經卡住不送資料）
STEP_MAX = 20.0
# 畫面方向 → 手臂座標（B 的 +X 朝畫面下方、+Y 朝右；A 轉了 180°：+X 朝上、+Y 朝左）
SCREEN = {"B": {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)},
          "A": {"up": (1, 0), "down": (-1, 0), "left": (0, 1), "right": (0, -1)}}
NAMES = ["左上", "上中", "右上", "左中", "中間", "右中", "左下", "下中", "右下"]
# 格線（黑膠帶）的四個交叉點：c0＝畫面上格 0、1、3、4 之間，c1＝1、2、4、5，c2＝3、4、6、7，c3＝4、5、7、8。
# 九格的最近、最遠兩排剛好在手臂可達範圍的邊緣（離底座 190～226、396～423 mm），交叉點在 242～362 mm，兩台都好碰，
# 而且有膠帶十字當記號，比「格子中心」好對準。
KEYS = [str(i) for i in range(9)] + list(CROSS)


def key_name(k, arm):
    if k in CROSS:
        return "交叉點（格 %s 之間）" % "、".join(str(c) for c in CROSS[k])
    c = int(k)
    return "點%d（畫面格 %d %s）" % ((c + 1) if arm == "A" else (9 - c), c, NAMES[c])

M = None
LOCK = threading.Lock()
MSG = {"text": "選一台手臂，按「上電」開始。"}


def load_teach():
    """沒有檔案＝空的；檔案壞掉就丟例外（不能當成空的，不然下一次記錄會把所有點蓋掉）。"""
    if not os.path.exists(TEACH):
        return {"A": {}, "B": {}}
    with open(TEACH, encoding="utf-8") as f:
        return json.load(f)


def save_teach(t):
    atomic_write(TEACH, json.dumps(t, indent=1, ensure_ascii=False))


_TEACH_VIEW = {"key": None, "t": None}


def teach_for_view():
    """給 /state（每 300 ms）用：檔案沒變就不重讀。只拿來顯示；記錄／刪除仍用 load_teach()。"""
    try:
        st = os.stat(TEACH)
    except OSError:
        return {"A": {}, "B": {}}
    key = (st.st_mtime_ns, st.st_size)
    if _TEACH_VIEW["key"] != key:
        _TEACH_VIEW["t"], _TEACH_VIEW["key"] = load_teach(), key
    return _TEACH_VIEW["t"]


def fresh_pose(a):
    """手臂目前 [X,Y,Z,R]；回饋太舊就丟例外。"""
    t, q, _ = a.arm.samples[-1]
    if time.perf_counter() - t > POSE_MAX_AGE:
        raise RuntimeError("%s 的位置回饋 %.1f 秒沒更新，先不要記錄或移動（重開這個工具或手臂）" % (a.name, time.perf_counter() - t))
    return [float(v) for v in q[:4]]


def joints(a):
    """30004 即時回饋裡的四個關節角（度）；還沒收到完整回饋時回 None。"""
    rec = getattr(a.arm, "last_rec", None)
    return [round(float(v), 2) for v in rec["q_actual"][:4]] if rec is not None else None


def safe_move(a, cmd):
    """跟其他程式一樣：運動指令失敗就 ResetRobot 停下來，再把錯誤往上丟。"""
    try:
        sync_move(a.arm, cmd)
    except Exception:
        try:
            a.arm.dash.ResetRobot()
        except Exception:
            pass
        raise


def arm_state(a):
    try:
        x, y, z, r = [round(float(v), 2) for v in a.arm.pose()[:4]]
        j = joints(a)
        return {"pose": [x, y, z, r], "joints": j, "mode": int(a.arm.mode())}
    except Exception as e:
        return {"pose": None, "mode": None, "err": str(e)[:80]}


def do(cmd, req):
    a = M.arms[req.get("arm", "B")]
    d = a.arm.dash
    if cmd == "enable":
        enable(a.arm, speed=10)
        d.sendRecvMsg("SetCollisionLevel(3)")
        d.sendRecvMsg("DOExecute(1,0)")
        return "%s 已上電（速度 10%%）" % a.name
    if cmd == "disable":
        d.DisableRobot()
        return "%s 已斷電" % a.name
    if cmd == "clear":
        d.ClearError()
        return "%s 已清報警" % a.name
    if cmd == "teach":
        cell = str(req["cell"])
        if cell not in KEYS:
            return "不認得的點：%s" % cell
        q = [round(v, 2) for v in fresh_pose(a)]             # 只讀一次（含新鮮度檢查），XYZ 與細節來自同一筆回饋
        x, y, z = q[:3]
        j = joints(a)
        t = load_teach()
        t.setdefault(a.name, {})[cell] = [x, y, z]
        t.setdefault("_detail", {}).setdefault(a.name, {})[cell] = {
            "xyzr": q, "joints": j, "mode": int(a.arm.mode()), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_teach(t)
        return "已記錄 %s 臂 %s = (%.1f, %.1f, %.1f)" % (a.name, key_name(cell, a.name), x, y, z)
    if cmd == "forget":
        t = load_teach()
        t.get(a.name, {}).pop(str(req["cell"]), None)
        save_teach(t)
        return "已刪除 %s 臂 格 %s 的紀錄" % (a.name, req["cell"])
    if a.arm.mode() != 5:
        return "%s 還沒上電（模式 %s），先按「上電」" % (a.name, a.arm.mode())
    if cmd == "park":
        q = fresh_pose(a)
        if q[2] < mover.Z_HOVER - 1:
            safe_move(a, "MovL(%f,%f,%f,%f)" % (q[0], q[1], mover.Z_HOVER, a.r))
        a.go_park()
        return "%s 回到停放點" % a.name
    if cmd == "move":
        step = min(abs(float(req.get("step", 1))), STEP_MAX)
        x, y, z, r = fresh_pose(a)
        dirn = req.get("dir")
        if dirn in ("zup", "zdown"):
            z += step if dirn == "zup" else -step
        else:
            dx, dy = SCREEN[a.name][dirn]
            x, y = x + dx * step, y + dy * step
        if z < Z_MIN[a.name]:
            return "不能再低了（最低 Z=%.0f）" % Z_MIN[a.name]
        if z > Z_MAX:
            return "不能再高了（最高 Z=%.0f，再高會靠近相機）" % Z_MAX
        p = np.array([x, y])
        if not a.ik(p, z):
            return "手臂到不了 (%.1f, %.1f, %.1f)" % (x, y, z)
        if dirn != "zup" and a.post(p, z):
            return "太靠近相機支架，不動"
        safe_move(a, "MovL(%f,%f,%f,%f)" % (x, y, z, r))
        return "%s → (%.1f, %.1f, %.1f)" % (a.name, x, y, z)
    return "不認得的指令"


def reconnect():
    global M
    try:
        M.close()
    except Exception:
        pass
    M = mover.Mover()
    print("已重新連線兩台手臂", flush=True)


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/state"):
            return self._send(200, {"A": arm_state(M.arms["A"]), "B": arm_state(M.arms["B"]),
                                    "teach": {k: v for k, v in teach_for_view().items() if k != "_detail"}, "msg": MSG["text"], "busy": LOCK.locked()})
        return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if not LOCK.acquire(blocking=False):
            return self._send(200, {"msg": "上一個動作還沒做完"})
        try:
            try:
                MSG["text"] = do(req.get("cmd"), req)
            except Exception as e:
                # 手臂重開電源、或別的程式同時連過手臂後，舊連線會一直回空字串：重連。
                # 只有重做也沒關係的指令才自動重送；移動、回停放點可能已經動了，重送會動兩次 → 只重連、不重送
                if "''" not in str(e) and not isinstance(e, ConnectionError):
                    raise
                reconnect()
                if req.get("cmd") in ("enable", "disable", "clear", "teach", "forget"):
                    MSG["text"] = do(req.get("cmd"), req)
                else:
                    MSG["text"] = "手臂連線剛剛斷了，已重新連線；這個動作沒有重送，確認位置後再按一次"
        except Exception as e:
            MSG["text"] = "出錯：%s" % str(e)[:120]
        finally:
            LOCK.release()
        return self._send(200, {"msg": MSG["text"]})

    def log_message(self, *a):
        pass


PAGE = r"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>手臂 XYZ 操作</title>
<style>
body{font-family:"Microsoft JhengHei",system-ui,sans-serif;background:#FAF9F5;color:#262624;margin:0;padding:16px;max-width:760px}
h1{font-size:20px;margin:0 0 12px}.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:10px 0}
button{font-size:16px;padding:10px 14px;border-radius:10px;border:1px solid #c9c6ba;background:#fff;cursor:pointer}
button.on{background:#D97757;color:#fff;border-color:#D97757}button:active{transform:scale(.97)}
.pad{display:grid;grid-template-columns:repeat(3,72px);grid-template-rows:repeat(3,64px);gap:6px}
.pad button{font-size:24px;padding:0}.pose{font:600 20px Consolas,monospace;background:#e9e6dc;color:#262624;padding:8px 14px;border-radius:10px}.pose.sel{background:#262624;color:#f3f1ea}
.msg{background:#f0ede4;padding:10px 12px;border-radius:10px;min-height:22px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;max-width:420px}
.grid button{font-size:14px;padding:8px 4px;line-height:1.4}.grid button.done{background:#e7f2e3;border-color:#7aa86b}
.hint{color:#6b6a64;font-size:13px}
</style></head><body>
<h1>手臂 XYZ 操作</h1>
<div class="row">手臂：<button id="aB" onclick="pick('B')">B（畫面上方）</button><button id="aA" onclick="pick('A')">A（畫面下方）</button>
<button onclick="cmd('enable')">上電</button><button onclick="cmd('disable')">斷電</button><button onclick="cmd('clear')">清報警</button><button onclick="cmd('park')">回停放點</button></div>
<div class="pose" id="poseB">B —</div><div class="pose" id="poseA" style="margin-top:6px">A —</div><div class="hint" id="upd"></div>
<div class="row">每步：<span id="steps"></span></div>
<div class="row" style="gap:24px;align-items:flex-start">
 <div><div class="hint">平面（照相機畫面方向）</div><div class="pad">
  <span></span><button onclick="mv('up')">↑</button><span></span>
  <button onclick="mv('left')">←</button><span></span><button onclick="mv('right')">→</button>
  <span></span><button onclick="mv('down')">↓</button><span></span></div></div>
 <div><div class="hint">高度</div><div class="pad" style="grid-template-columns:72px">
  <button onclick="mv('zup')">Z＋</button><span></span><button onclick="mv('zdown')">Z－</button></div></div>
</div>
<div class="msg" id="msg"></div>
<h1 style="margin-top:18px">現場校正：記錄 5 個點 <span id="prog" class="hint"></span></h1>
<div class="hint">吸盤對準黑膠帶十字交叉的正中心、輕碰到棋盤面，再按對應的按鈕（記錄的是「目前選的」那台；不用上電也可以用手推過去）。從側面和正面各看一次確認對準。中心點用來驗算，碰不到可以不做。按鈕排列＝站在這台手臂後面看棋盤的樣子。</div>
<div id="cal" class="grid" style="grid-template-columns:repeat(3,1fr);max-width:420px"></div>
<details style="margin-top:12px"><summary class="hint">其他格子中心（選用，之前記錄的在這裡）</summary><div class="grid" id="grid" style="margin-top:6px"></div></details>
<div class="hint" style="margin-top:8px">鍵盤：方向鍵移動、PageUp／PageDown 調高度、1～5 換步長。</div>
<script>
let arm="B", step=1, busy=false, gridSig=""; const STEPS=[0.5,1,5,10,20], NAMES=["左上","上中","右上","左中","中間","右中","左下","下中","右下"];
const $=s=>document.querySelector(s);
$("#steps").innerHTML=STEPS.map((s,i)=>`<button id="s${i}" onclick="setStep(${i})">${s} mm</button>`).join(" ");
function setStep(i){step=STEPS[i];STEPS.forEach((_,j)=>$("#s"+j).classList.toggle("on",i==j));}
function pick(a){arm=a;$("#aA").classList.toggle("on",a=="A");$("#aB").classList.toggle("on",a=="B");refresh();}
async function cmd(c,extra={}){if(busy)return;busy=true;try{const r=await fetch("/cmd",{method:"POST",body:JSON.stringify(Object.assign({cmd:c,arm},extra))});$("#msg").textContent=(await r.json()).msg;}catch(e){$("#msg").textContent="連不到操作程式";}busy=false;refresh();}
function mv(d){cmd("move",{dir:d,step});}
async function refresh(){try{const s=await (await fetch("/state",{cache:"no-store"})).json();
 for(const n of ["A","B"]){const a=s[n],e=$("#pose"+n);e.classList.toggle("sel",n==arm);
  const m={4:"未上電",5:"已上電",6:"拖曳模式",7:"運動中",9:"報警"}[a.mode]||("模式 "+a.mode);
  e.innerHTML=`${n}${n==arm?"（目前選的）":""}  `+(a.pose?`X ${a.pose[0].toFixed(1)}  Y ${a.pose[1].toFixed(1)}  Z ${a.pose[2].toFixed(1)}  R ${a.pose[3].toFixed(1)}°`:"讀不到位置")+`   ${m}`
   +(a.joints?`<br><span style="font-size:15px">J1 ${a.joints[0].toFixed(2)}°  J2 ${a.joints[1].toFixed(2)}°  J3 ${a.joints[2].toFixed(2)}°  J4 ${a.joints[3].toFixed(2)}°</span>`:"");}
 $("#upd").textContent="更新於 "+new Date().toLocaleTimeString();
 if(!busy&&s.msg)$("#msg").textContent=s.msg;
 const t=(s.teach||{})[arm]||{};
 const sig=arm+"|"+JSON.stringify(t); if(sig===gridSig)return; gridSig=sig;   // 記錄沒變就不重建按鈕（重建會吃掉正在按的那一下）
 const btn=(k,label,sub)=>{const p=t[k];return `<button class="${p?"done":""}" onclick="cmd('teach',{cell:'${k}'})" oncontextmenu="event.preventDefault();cmd('forget',{cell:'${k}'})">${label}<span class="hint">${sub}</span><br>${p?p.map(v=>v.toFixed(1)).join(", "):"未記錄"}</button>`};
 // 從「目前選的手臂」的角度排：站在手臂後面看棋盤，遠的那排在上、手臂的左手邊在左 → 點1＝左上…點9＝右下。
 // 兩台面對面，所以是轉 180 度（不是鏡像）：A 的點 k ＝ 畫面格 k-1；B 的點 k ＝ 畫面格 9-k。
 $("#grid").innerHTML=[0,1,2,3,4,5,6,7,8].map(i=>{const c=arm=="A"?i:8-i;return btn(String(c),`點${i+1}`,`（畫面格${c} ${NAMES[c]}）`)}).join("");
 const ck=j=>"c"+(arm=="A"?j:3-j);           // 畫面上的交叉點 → 依手臂視角排（B 轉 180 度）
 $("#cal").innerHTML=[btn(ck(0),"交叉點一","（左遠）"),"<span></span>",btn(ck(1),"交叉點二","（右遠）"),
   "<span></span>",btn("4","中心點","（中間格中心，驗算用）"),"<span></span>",
   btn(ck(2),"交叉點三","（左近）"),"<span></span>",btn(ck(3),"交叉點四","（右近）")].join("");
 const n=["c0","c1","c2","c3"].filter(k=>t[k]).length;$("#prog").textContent=`（${arm}：交叉點 ${n}/4${t["4"]?"、中心點已記":""}）`;
}catch(e){}}
document.addEventListener("keydown",e=>{const k={ArrowUp:"up",ArrowDown:"down",ArrowLeft:"left",ArrowRight:"right",PageUp:"zup",PageDown:"zdown"}[e.key];
 if(k){e.preventDefault();mv(k);}else if("12345".includes(e.key))setStep(+e.key-1);});
pick("B");setStep(1);(function loop(){(document.hidden?Promise.resolve():refresh()).finally(()=>setTimeout(loop,300));})();
</script></body></html>"""


def main():
    global M
    import socket
    for port, who in ((18080, "對弈伺服器"), (8800, "雙臂控制台")):      # 兩邊都會連手臂，同時開會互搶
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            raise SystemExit("%s還開著（%d），請先關掉再開這個。" % (who, port))
        except OSError:
            pass
    M = mover.Mover()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)      # 只給本機開：這個頁面可以讓手臂動，不能讓網路上其他人連
    print("手臂 XYZ 操作：http://127.0.0.1:%d/" % PORT, flush=True)
    try:
        srv.serve_forever()
    finally:
        for a in M.arms.values():
            try:
                a.arm.dash.DisableRobot()
            except Exception:
                pass
        M.close()


if __name__ == "__main__":
    main()
