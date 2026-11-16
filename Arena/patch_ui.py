import sys
p=sys.argv[1]; s=open(p,encoding="utf-8").read()
def rep(a,b):
    global s
    assert s.count(a)==1,a; s=s.replace(a,b)
rep('''        <button class="btn" data-cmd="hint">提示</button>''','''        <button class="btn" data-cmd="hint">提示</button>
        <button class="btn" data-cmd="hand_over" title="手掌張開伸到棋盤上方，手臂把一顆 X 放到你手上">✋ 放一顆到我手上</button>''')
rep('''    $("#simClear").hidden = !s.dry;''','''    $("#simClear").hidden = !s.dry;
    if (s.phase === "handover" && !$("#camBox").open) $("#camBox").open = true;''')
rep('''      s.phase === "robot" ? "手臂動作中，請勿把手伸進棋盤或料區。" :''','''      s.phase === "robot" ? "手臂動作中，請勿把手伸進棋盤或料區。" :
      s.phase === "handover" ? (s.palm ? `看到手掌（離棋盤 ${s.palm.h} mm）${s.palm.ok ? "，保持不動" : ""}` : "掌心朝上、張開，伸到棋盤上方約 5 公分；說「停」隨時中斷。") :''')
open(p,"w",encoding="utf-8").write(s); print("ui patched")
