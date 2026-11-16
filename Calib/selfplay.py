"""完整下一盤井字棋（虛擬盤面）：X、O 都用 minimax，每一步由夠得到該格的手臂實際執行（懸停→下降→DO 開關→回停放）。

真空未接、棋子無法自動放，所以盤面由程式記錄（不靠相機讀後續盤面）。碰撞規則與 play.py 相同：
一次只動一台、另一臂先移到該格對應的停放點、淨空 ≥ 150 mm；淨空不足的格（格 7）不選。
    python selfplay.py               # 乾跑：只印整盤棋譜與每步指派
    python selfplay.py --go --do=1   # 實際執行
"""
import json
import os
import sys
import time

import numpy as np

from common import connect_arm, enable, sync_move
from game import best_move, describe, winner
from play import guard_camera, EXCLUDE_CELLS, in_keepout, path_clear, approach, CLEAR_MM, PARK, Z_HOVER, Z_PLACE, a_to_b_transform, at, best_park, clearance, go_park

HERE = os.path.dirname(os.path.abspath(__file__))


def result_text(b):
    """如實回報：有人連線才算勝；盤面全滿才算和局；否則是「可下格用盡、未分勝負」。"""
    if winner(b):
        return winner(b) + " 勝"
    if " " not in b:
        return "和局"
    return "可下格用盡、未分勝負（空格 %s 因可達性／碰撞限制不能下）" % [i for i in range(9) if b[i] == " "]


def plan_game(plan, T):
    allowed = {int(i) for i, v in plan.items()
               if v["arm"] and int(i) not in EXCLUDE_CELLS and not in_keepout(v["arm"], v[v["arm"]])
               and path_clear(v["arm"], PARK[v["arm"]], v[v["arm"]]) and clearance(v["arm"], v[v["arm"]], T, best_park(v["arm"], v[v["arm"]], T)) >= CLEAR_MM}
    board, turn, moves = [" "] * 9, "X", []
    while not winner(board) and " " in board:
        cell, score = best_move(board, turn, allowed=allowed)
        if cell is None:                       # 剩下的空格都不能下（例如只剩格 7）
            break
        board[cell] = turn
        moves.append((turn, cell, score))
        turn = "O" if turn == "X" else "X"
    return moves, board, allowed


def execute(arms, plan, T, turn, cell, do_idx):
    active = plan[str(cell)]["arm"]
    other = "B" if active == "A" else "A"
    txy = plan[str(cell)][active]
    PARK[other] = best_park(active, txy, T)
    c = clearance(active, txy, T)
    if c < CLEAR_MM:
        raise RuntimeError("格 %d 淨空 %.0f mm 不足" % (cell, c))
    o = arms[other]
    if not at(o, PARK[other]):
        enable(o, speed=15)
        go_park(o)
    o.dash.DisableRobot()
    if in_keepout(active, txy) or not path_clear(active, PARK[active], txy):
        raise RuntimeError("格 %d 的目標或路線會碰到支架" % cell)
    a = arms[active]
    enable(a, speed=15)
    if not at(a, PARK[active]):
        go_park(a)                              # 路線只從停放點出發做過檢查
    r = float(a.pose()[3])
    approach(a, txy)
    sync_move(a, "MovL(%f,%f,%f,%f)" % (txy[0], txy[1], Z_PLACE, r))
    pose_at_place = a.pose().round(1)
    a.dash.sendRecvMsg("DOExecute(%d,1)" % do_idx)
    time.sleep(0.4)
    a.dash.sendRecvMsg("DOExecute(%d,0)" % do_idx)
    sync_move(a, "MovL(%f,%f,%f,%f)" % (txy[0], txy[1], Z_HOVER, r))
    go_park(a)                                  # 每手結束都回停放點，下一手從已檢查過的路線出發
    a.dash.DisableRobot()
    return active, other, PARK[other], c, pose_at_place


def main():
    guard_camera()
    go = "--go" in sys.argv
    do_idx = int(next((a[5:] for a in sys.argv if a.startswith("--do=")), "1"))
    plan = json.load(open(os.path.join(HERE, "cell_plan.json")))
    T = a_to_b_transform()
    moves, final, allowed = plan_game(plan, T)
    print("可下格：%s（格 7 淨空不足已排除）" % sorted(allowed))
    for k, (turn, cell, score) in enumerate(moves, 1):
        print("第 %d 手：%s 下格 %d（由 %s 執行）" % (k, turn, cell, plan[str(cell)]["arm"]))
    print("終局：\n%s\n結果：%s" % (describe(final), result_text(final)))
    if not go:
        return

    arms = {n: connect_arm(n) for n in "AB"}
    board = [" "] * 9
    t0 = time.perf_counter()
    try:
        for a in arms.values():
            a.wait_feed()
        for k, (turn, cell, _) in enumerate(moves, 1):
            active, other, park, c, pz = execute(arms, plan, T, turn, cell, do_idx)
            board[cell] = turn
            print("第 %d 手 %s→格 %d：%s 執行（放置高度姿態 %s），%s 停在 %s，淨空 %.0f mm ✓"
                  % (k, turn, cell, active, pz, other, park[:2], c))
        # 收尾：兩臂都回停放（先動不在路徑上的那台）
        for n in "AB":
            x = arms[n]
            PARK[n] = PARK[n]
            if not at(x, PARK[n]):
                enable(x, speed=15)
                go_park(x)
                x.dash.DisableRobot()
        print("\n整盤完成（%.0f s）：\n%s\n結果：%s" % (time.perf_counter() - t0, describe(board), result_text(board)))
    except Exception as e:
        print("停止：", e)
        for x in arms.values():
            try:
                x.dash.ResetRobot()
            except Exception:
                pass
    finally:
        for x in arms.values():
            try:
                x.dash.sendRecvMsg("DOExecute(%d,0)" % do_idx)
                x.dash.DisableRobot()
            except Exception:
                pass
            x.close()


if __name__ == "__main__":
    main()
