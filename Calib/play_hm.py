"""人機對弈：人下 X（手擺棋子），機器下 O（B 從 O 疊吸子放到格上；只有 A 夠得到的格由 B 送到交接點、A 再下）。

    python play_hm.py --dry       # 只看目前盤面、機器會選哪格、由誰下
    python play_hm.py             # 開始對弈：每次偵測到你多下一顆 X，機器回一手 O；分出勝負或沒有可下的格就結束
    python play_hm.py --o-count=3 # O 疊目前顆數（預設讀 stock_count.json 的 O疊）
選項：--first=O 機器先下；--use-A 啟用 A＋交接點（預設關，格 3 需先實測左側支架淨空）。
"""
import json
import os
import sys
import time

import numpy as np

from board import find_grid_robust
from common import Camera, connect_arm, enable, sync_move
from game import best_move, describe, winner
from pickplace import STOCK, Z_TOUCH_BOARD, seg_clear, travel, travel_points
from plan_cells import hits_post
from play import PARK, Z_HOVER
from world import load

HERE = os.path.dirname(os.path.abspath(__file__))
PIECE = STOCK["piece_mm"]
STACK_PRESS = 5.0          # 疊上吸取多壓（實測 2 mm 吸不到）
Z_RELEASE = Z_TOUCH_BOARD + 1.5   # 放到棋盤：棋子底離板 1.5 mm 放開
COUNT_FILE = os.path.join(HERE, "stock_count.json")
USE_A = "--use-A" in sys.argv
ROBOT, HUMAN = "O", "X"


def opt(name, default):
    for a in sys.argv[1:]:
        if a.startswith("--%s=" % name):
            return type(default)(a.split("=", 1)[1])
    return default


def read_board(cam, frames=3):
    """多次讀盤取多數決，降低 O／X 誤判（實測單張會把 X 判成 O）。"""
    reads = []
    for _ in range(frames):
        _, g, cls = find_grid_robust(cam)
        reads.append([("O" if k == "O" else "X" if k == "X" else " ") for k, _, _ in cls])
        time.sleep(0.3)
    out = []
    for i in range(9):
        vals = [r[i] for r in reads]
        out.append(max(set(vals), key=vals.count))
    return out


def b_target(arm, cell, plan, r):
    """B 放到某格的落點：格中心；放開高度 IK 不可達時沿徑向找 ≤25 mm 的可達點（棋子大致仍在格內）。"""
    c = np.array(plan[str(cell)]["B"])
    u = c / np.hypot(*c)
    for d in np.arange(0, 25.5, 1.0):
        for sgn in (1, -1):
            p = c + sgn * d * u
            if all(arm.dash.InverseSolution(p[0], p[1], z, r, 0, 0).startswith("0,") for z in (Z_RELEASE, Z_HOVER)):
                return p
    return None


def robot_cells(armB, plan):
    """機器（B 直接）能下的格與落點。"""
    calB = load("B")
    r = float(armB.pose()[3])
    stk = np.array(STOCK["stocks"]["O疊"]["xy"])
    out = {}
    for i in range(9):
        p = b_target(armB, i, plan, r)
        if p is None or hits_post(calB, p, Z_RELEASE) or hits_post(calB, p, Z_HOVER):
            continue
        if not (seg_clear(calB, travel_points(stk, p), Z_HOVER) and seg_clear(calB, travel_points(p, np.array(PARK["B"][:2])), Z_HOVER)):
            continue
        out[i] = p
    return out


def b_place(arm, n_stack, target):
    """B：O 疊頂端吸一顆 → 放到 target → 回停放。"""
    r = float(arm.pose()[3])
    stk = np.array(STOCK["stocks"]["O疊"]["xy"])
    z_top = STOCK["stocks"]["O疊"]["z_first"] - 1.5 + (n_stack - 1) * PIECE
    enable(arm, speed=10)
    p = arm.pose()
    if p[2] < Z_HOVER - 1:
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (p[0], p[1], Z_HOVER, r))
    travel(arm, stk, r)
    sync_move(arm, "MovL(%f,%f,%f,%f)" % (stk[0], stk[1], z_top - STACK_PRESS, r))
    arm.dash.sendRecvMsg("DOExecute(1,1)")
    time.sleep(0.8)
    sync_move(arm, "MovL(%f,%f,%f,%f)" % (stk[0], stk[1], Z_HOVER, r))
    travel(arm, target, r)
    sync_move(arm, "MovL(%f,%f,%f,%f)" % (target[0], target[1], Z_RELEASE, r))
    arm.dash.sendRecvMsg("DOExecute(1,0)")
    time.sleep(0.6)
    sync_move(arm, "MovL(%f,%f,%f,%f)" % (target[0], target[1], Z_HOVER, r))
    travel(arm, np.array(PARK["B"][:2]), r)
    arm.dash.DisableRobot()


def main():
    dry = "--dry" in sys.argv
    plan = json.load(open(os.path.join(HERE, "cell_plan.json"), encoding="utf-8"))
    counts = json.load(open(COUNT_FILE, encoding="utf-8"))
    n_o = opt("o-count", int(counts.get("O疊", 0)))
    if USE_A:
        raise SystemExit("A＋交接點尚未啟用：需先在使用者目視下實測格 3 與左側支架的淨空")
    armB = connect_arm("B")
    cam = Camera()
    try:
        armB.wait_feed()
        cells = robot_cells(armB, plan)
        print("機器能下的格（B 直接）：%s；O 疊 %d 顆" % (sorted(cells), n_o))
        board = read_board(cam)
        print(describe(board))
        turn_robot = opt("first", "X") == ROBOT if board.count("X") == board.count("O") else board.count("X") > board.count("O")
        if dry:
            if turn_robot:
                mv, sc = best_move(board, ROBOT, allowed=set(cells))
                print("輪到機器：會下格 %s（分數 %s），落點 B %s" % (mv, sc, cells.get(mv)))
            else:
                print("輪到你（X）")
            return
        while True:
            w = winner(board)
            if w or " " not in board:
                print("\n結束：%s" % ("%s 勝" % w if w else "和局"))
                break
            if not turn_robot:
                print("\n輪到你下 X（擺好後手離開棋盤）…")
                n_x = board.count("X")
                while True:                                     # 每 5 秒判斷一次；沒有新落子就再等 5 秒（使用者指定）
                    time.sleep(5.0)
                    nb = read_board(cam)
                    if nb.count("X") == n_x + 1 and all(nb[i] == board[i] for i in range(9) if board[i] != " "):
                        nb2 = read_board(cam)                    # 立即再讀一次確認（手已離開、盤面穩定）
                        if nb2 == nb:
                            board = nb
                            break
                    print("  %s 尚未偵測到新的 X，5 秒後再看" % time.strftime("%H:%M:%S"), flush=True)
                print(describe(board))
                turn_robot = True
                continue
            free = {i for i in cells if board[i] == " "}
            if not free:
                print("\n機器沒有可下的格（剩下的空格手臂都下不到），結束")
                break
            if n_o < 1:
                print("\nO 疊用完了，結束")
                break
            mv, sc = best_move(board, ROBOT, allowed=free)
            print("\n機器下 O → 格 %d（分數 %s），B 從 O 疊（第 %d 顆）取子" % (mv, sc, n_o))
            b_place(armB, n_o, cells[mv])
            n_o -= 1
            counts["O疊"] = n_o
            json.dump(counts, open(COUNT_FILE, "w", encoding="utf-8"), ensure_ascii=False)
            time.sleep(1.0)
            nb = read_board(cam)
            if nb[mv] != ROBOT:
                print("⚠ 相機沒看到格 %d 的 O（讀到 %r），請確認棋子是否放好；以預期盤面繼續" % (mv, nb[mv]))
            board[mv] = ROBOT
            print(describe(board))
            turn_robot = False
    except BaseException as e:
        print("停止：", e)
        try:
            armB.dash.sendRecvMsg("DOExecute(1,0)")
            armB.dash.ResetRobot()
        except Exception:
            pass
        raise
    finally:
        try:
            armB.dash.DisableRobot()
        except Exception:
            pass
        cam.close()
        armB.close()


if __name__ == "__main__":
    main()
