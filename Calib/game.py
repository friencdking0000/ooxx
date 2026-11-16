"""井字棋決策：讀相機盤面 → minimax 選最佳落點 → 依可達性指派手臂。不會動手臂。

    python game.py            # 以目前相機畫面決定「輪到誰」並建議下一步
盤面用長度 9 的 list，元素 'X'／'O'／' '；格子 0..8 = 影像由上到下、由左到右。
輪到誰：數量少的一方；數量相同時預設 X 先手（可用 --first=O 改）。
"""
import sys
from functools import lru_cache

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(b):
    for a, c, d in LINES:
        if b[a] != " " and b[a] == b[c] == b[d]:
            return b[a]
    return None


def other(p):
    return "O" if p == "X" else "X"


@lru_cache(maxsize=None)
def _value(b, turn, me):
    w = winner(b)
    if w:
        return 1 if w == me else -1
    if " " not in b:
        return 0
    vals = [_value(b[:i] + (turn,) + b[i + 1:], other(turn), me) for i in range(9) if b[i] == " "]
    return max(vals) if turn == me else min(vals)


def best_move(board, me, allowed=None):
    """回傳 (最佳格, 分數)。allowed：限制可選格（例如兩台手臂都夠不到的格子要排除）。同分時偏好中心、角、邊。"""
    b = tuple(board)
    pref = [4, 0, 2, 6, 8, 1, 3, 5, 7]
    cands = [i for i in pref if b[i] == " " and (allowed is None or i in allowed)]
    for i in cands:                                   # 能立刻贏就直接贏
        if winner(b[:i] + (me,) + b[i + 1:]) == me:
            return i, 1
    scored = [(_value(b[:i] + (me,) + b[i + 1:], other(me), me), -pref.index(i), i) for i in cands]
    if not scored:
        return None, None
    s, _, i = max(scored)
    return i, s


def whose_turn(board, first="X"):
    x, o = board.count("X"), board.count("O")
    if x == o:
        return first
    return "X" if x < o else "O"


def describe(board):
    return "\n".join(" ".join(board[r * 3 + c] if board[r * 3 + c] != " " else "." for c in range(3)) for r in range(3))


if __name__ == "__main__":
    from board import find_grid_robust
    from common import Camera
    from reach import assignment, cell_report
    first = next((a[8:] for a in sys.argv if a.startswith("--first=")), "X")
    cam = Camera()
    try:
        img, g, cls = find_grid_robust(cam)
    finally:
        cam.close()
    board = [("O" if k == "O" else "X" if k == "X" else " ") for k, _, _ in cls]
    print(describe(board))
    print("X=%d O=%d" % (board.count("X"), board.count("O")))
    w = winner(board)
    if w or " " not in board:
        print("已結束：", "%s 贏" % w if w else "和局")
        sys.exit(0)
    me = whose_turn(board, first)
    asg = assignment(cell_report(g["cells"]))
    reachable = {i for i, a in asg.items() if a}
    move, score = best_move(board, me, allowed=reachable)
    full_move, full_score = best_move(board, me)
    print("輪到 %s。最佳落點（不考慮可達性）：格 %s，分數 %s" % (me, full_move, full_score))
    print("考慮可達性後：格 %s（由手臂 %s 執行），分數 %s" % (move, asg.get(move), score))
    if move != full_move:
        print("⚠ 不考慮可達性的最佳格 %s 沒有手臂夠得到，這是近格自身干涉造成的限制" % full_move)
