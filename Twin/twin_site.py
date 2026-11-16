# 井字遊戲：兩個「模型」（下棋程式）對弈，兩台手臂幫它們下
# 左邊 a1 下 O（先手）、右邊 a2 下 X，棋子各疊在自己旁邊。每一步先「看」棋盤（從棋子位置讀出每格是什麼），
# 再問輪到的模型要下哪。模型就是一個函式 model(board, me, opp) → (列, 行)，內建三個：
#   高手：把整棵棋局算完（minimax），絕對不會輸
#   普通：能贏就贏 → 對手要贏了就擋 → 中間 → 角落 → 邊（遇到「雙殺」會輸）
#   新手：能贏就贏；對手要贏了，一半的時候看得到要擋；其他亂下
# 下 GAMES 盤，每盤兩個模型交換先後手（換邊），最後印比分。改 MODELS 換對手，或寫自己的模型來挑戰高手
import math
import random
from dtlink import Twin
from arm import Arm, ReachError

t = Twin()
t.wait_connected(600)
print(t.layout("arm2", x=597.3, y=2.0, yaw=178.4))
print(t.layout("board", x=291.4, y=-1.2))
a1 = Arm(t)                                  # 左邊那台：下 O，座標原點就是它的基座
a2 = Arm(t, root="MAIN.FG_AT450_2")      # 右邊那台：下 X，基座在 a1 的 x=620 mm，轉了 180°（面對面）
# 兩台會撞在一起：都伸到棋盤中間就撞了。孿生算兩台的最近距離（a1.arm_gap），太近 arm.py 會停車丟 InterferenceError。
# 所以輪流：沒輪到的那台轉到 PARK（J1 轉 90°、背對棋盤）讓位
READY = [0, 30, 40, 0]
PARK1, PARK2 = [90, 0, 20, 0], [-90, 0, 20, 0]    # 兩台都轉向後面（a2 轉了 180°，所以它的 -90° 才是後面）


A2 = (620, 0, 180)                          # a2 基座在 a1 座標的 x、y（mm）與朝向（度；面對面＝180）
BOARD, BOARD_Z = (310, 0), 0              # 棋盤中心（a1 座標）、棋盤面高度（mm）
try:                                         # 本機模式：照 t.layout() 設的實際擺位走（東西搬過，程式不用改）
    L = t.layout()
    A2 = (L["arm2"]["x"], L["arm2"]["y"], L["arm2"]["yaw"])
    BOARD_Z += L["board"]["z"] - -7.5            # 棋盤整塊墊高或降低
    BOARD = (L["board"]["x"], L["board"]["y"])
except (RuntimeError, TimeoutError):         # 網頁模式沒有擺位指令：用上面的預設
    pass
print("擺位：B 台基座", A2, "棋盤中心", BOARD, "棋盤面高", BOARD_Z)


def to_a2(x, y):                             # a1 座標 → a2 座標（面對面時：x 從對面量回來、y 反過來）
    c, s = math.cos(math.radians(A2[2])), math.sin(math.radians(A2[2]))
    dx, dy = x - A2[0], y - A2[1]
    return c * dx + s * dy, -s * dx + c * dy
STEP_X, STEP_Y = 90, 98                    # 一格的大小（mm）：x 方向（朝對面那台）、y 方向
H = 8                                        # 棋子厚度（mm）
MARKS = "OOOOOXXXX"                        # 第 n 個棋子是 O 還是 X（跟 parts() 的順序一樣）
SAFE, FAST, SLOW = 100, 150, 50
LINES = [[(r, 0), (r, 1), (r, 2)] for r in range(3)] + [[(0, c), (1, c), (2, c)] for c in range(3)] + \
        [[(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]]


def cell_xy(r, c):                           # 第 r 列、第 c 行的格子中心（a1 座標）
    return BOARD[0] + (c - 1) * STEP_X, BOARD[1] + (1 - r) * STEP_Y


def where(p):                                # 棋子在哪一格；不在棋盤上回傳 None
    c = round((p["x"] - BOARD[0]) / STEP_X) + 1
    r = 1 - round((p["y"] - BOARD[1]) / STEP_Y)
    return (r, c) if 0 <= r < 3 and 0 <= c < 3 and p["z"] > BOARD_Z else None


def read_board():                            # 看棋盤（孿生回報每個棋子的位置，就像相機看到的）
    board = [[" "] * 3 for _ in range(3)]
    for mark, p in zip(MARKS, a1.parts()):
        rc = where(p)
        if rc:
            board[rc[0]][rc[1]] = mark
    return board


def winner(board):
    for line in LINES:
        v = {board[r][c] for r, c in line}
        if len(v) == 1 and v != {" "}:
            return v.pop()
    return None


def empties(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == " "]


def finish(board, who, empty):               # who 下一步就能連線的格子（沒有回傳 None）
    for r, c in empty:
        board[r][c] = who
        won = winner(board) == who
        board[r][c] = " "
        if won:
            return r, c


_memo = {}


def value(board, me, opp):                   # 輪到 me、兩邊都下最好時的結果：1 贏、0 平、-1 輸
    key = "".join(sum(board, [])) + me
    if key not in _memo:
        empty = empties(board)
        if winner(board):
            _memo[key] = -1                  # 對手上一手剛連線
        elif not empty:
            _memo[key] = 0
        else:
            _memo[key] = max(after(board, rc, me, opp) for rc in empty)
    return _memo[key]


def after(board, rc, me, opp):               # me 下在 rc 之後，對 me 來說的結果
    board[rc[0]][rc[1]] = me
    v = -value(board, opp, me)
    board[rc[0]][rc[1]] = " "
    return v


def expert(board, me, opp):                  # 高手：minimax；一樣好的走法隨機挑一個，每盤才會不一樣
    vals = {rc: after(board, rc, me, opp) for rc in empties(board)}
    top = max(vals.values())
    return random.choice([rc for rc, v in vals.items() if v == top])


def rules(board, me, opp):                   # 普通：照固定順序的規則
    empty = empties(board)
    rc = finish(board, me, empty) or finish(board, opp, empty)
    if rc:
        return rc
    for rc in [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)]:
        if rc in empty:
            return rc


def novice(board, me, opp):                  # 新手
    empty = empties(board)
    rc = finish(board, me, empty)
    if rc:
        return rc
    rc = finish(board, opp, empty)
    if rc and random.random() < 0.5:
        return rc
    return random.choice(empty)


def carry(a, src, src_z, dst, dst_z):
    # 用手臂 a 把 src 上方那個棋子（吸嘴口高度 src_z）搬到 dst、放在 dst_z（都是 a 自己的座標）
    a.move_l(x=src[0], y=src[1], z=SAFE, rx=180, ry=0, rz=0, speed=FAST)
    a.move_l(z=src_z, speed=SLOW)
    if not a.grip():                         # 開真空，回傳有沒有吸到
        raise RuntimeError("沒吸到，停下來檢查高度")
    a.move_l(z=SAFE, speed=SLOW)
    a.move_l(x=dst[0], y=dst[1], speed=FAST)
    a.move_l(z=dst_z, speed=SLOW)
    a.release()                              # 破真空，棋子放下
    a.move_l(z=SAFE, speed=SLOW)


def play(a, mark, r, c):                     # 手臂 a 從自己那疊拿一個 mark 棋子，下在 (r, c)
    pieces = [p for m, p in zip(MARKS, a1.parts()) if m == mark and where(p) is None]
    p = max(pieces, key=lambda q: q["z"])    # 那疊最上面那個
    src, dst = (p["x"], p["y"]), cell_xy(r, c)
    if a is a2:
        src, dst = to_a2(*src), to_a2(*dst)
    a.move_j(READY)                          # 從讓位姿態轉回來
    try:
        carry(a, src, p["z"] + H / 2, dst, BOARD_Z + H + 1)
    except ReachError as e:                  # 擺位改過（t.layout()）之後最容易遇到：格子或棋子堆離手臂太近／太遠
        raise ReachError(f"{'a1' if a is a1 else 'a2'} 下第 {r + 1} 列第 {c + 1} 行時{e}；"
                         "檢查擺位：每格離兩台基座都要在約 220～425 mm 內") from None
    a.move_j(PARK1 if a is a1 else PARK2)    # 讓位，換對手


def show(board):
    print("\n".join(" " + " | ".join(row) for row in board))
    print()


MODELS = [("高手", expert), ("新手", novice)]   # 對弈的兩個模型（名字, 函式），可換成 rules 或自己寫的
GAMES = 3                                    # 下幾盤（一盤幾分鐘）；每盤交換先後手

score = {name: 0 for name, _ in MODELS}
score["平手"] = 0
for g in range(GAMES):
    first, second = MODELS if g % 2 == 0 else MODELS[::-1]
    print(f"=== 第 {g + 1} 盤：{first[0]}（O，左邊先下）對 {second[0]}（X，右邊）===")
    a1.reset_parts()                         # 棋子疊回原位、兩台都破真空
    t.place_parts({0: (431, -178), 1: (431, -178), 2: (431, -178), 3: (333, -191), 4: (333, -191),
                   5: (109, -211), 6: (109, -211), 7: (109, -211), 8: (109, -211)})   # 現場料區位置
    a2.move_j(PARK2)                         # 先讓兩台都轉開，再輪流下（一起伸到中間會撞）
    a1.move_j(PARK1)
    players = [(a2, "O", "X", first), (a1, "X", "O", second)]
    for turn in range(9):
        a, me, opp, (name, model) = players[turn % 2]
        r, c = model(read_board(), me, opp)
        print(f"第 {turn + 1} 步：{name}（{me}）下在第 {r + 1} 列第 {c + 1} 行")
        play(a, me, r, c)
        board = read_board()
        show(board)
        if winner(board):
            break
    w = winner(read_board())
    who = {"O": first[0], "X": second[0]}.get(w, "平手")
    score[who] += 1
    print(f"第 {g + 1} 盤結果：" + (f"{who}（{w}）贏" if w else "平手"))
    print()

print("比分：", "、".join(f"{k} {v}" for k, v in score.items()))
print("棋盤上有", a1.tray_count, "個棋子（最後一盤）")
