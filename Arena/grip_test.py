"""實機放子／收子測試（不開伺服器）：做法都在 carrier.py，跟對弈伺服器同一套。

    python grip_test.py A X                      # 只量（料疊、支架、格線、棋盤），不動
    python grip_test.py --go --place=3,4,5 A X   # 從 X 疊拿 X 放到畫面格 3、4、5（搆不到的格由 B 經轉送點接手）
    python grip_test.py --go --place=0,1,2 B O
    python grip_test.py --go --tidy A X          # 照紀錄把棋盤上（和轉送點上）的 X 收回 X 疊
    python grip_test.py --go --clear A X         # 收棋盤：X、O 都收

手臂和種類都要寫，而且要跟 ox_record.json 的料疊負責手臂一致（不要搞混）。
記錄：Arena/out/grip_test_<時間>_<臂><種類>.json ＋ 放完的照片 place_<時間>.jpg。
"""
import datetime
import json
import os
import sys
import time

import cv2

import carrier
import mover

USAGE = "用法：python grip_test.py [--go] [--place=0,1,2 | --tidy | --clear] <A|B> <X|O>（手臂和種類都要寫，不猜）"


def main():
    go = "--go" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(pos) != 2 or pos[0] not in ("A", "B") or pos[1] not in ("O", "X"):
        raise SystemExit(USAGE)
    name, kind = pos
    cells = next(([int(v) for v in a[8:].split(",")] for a in sys.argv if a.startswith("--place=")), None)
    os.makedirs(carrier.OUT, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rec = dict(time=stamp, arm=name, kind=kind)
    cam = carrier.DirectCam()
    M = mover.Mover()
    moved = False
    try:
        sc = carrier.Scene(cam, M)
        if sc.rec["piles"][kind]["arm"] != name:
            raise SystemExit("紀錄裡 %s 疊是 %s 臂負責的，不是 %s 臂（不要搞混）" % (kind, sc.rec["piles"][kind]["arm"], name))
        print("紀錄：%s；棋盤上記著 %s；轉送點 %s" % (
            "、".join("%s 疊在 B %s（%s 臂）" % (k, h["B"], h["arm"]) for k, h in sc.rec["piles"].items()),
            [(e["kind"], e["B"]) for e in sc.rec["board"]] or "沒有", sc.rec.get("hand")), flush=True)
        img, piles = sc.measure()
        for s in piles:
            if s["kind"] != "?":
                print("料疊 %s（看到的符號 %s）%d 顆 → B %s；頂面 Z %s" % (
                    s["kind"], s["seen"], s["n"], s["arm_xy"]["B"].round(1),
                    {n: round(z, 1) for n, z in s["z_top"].items()}), flush=True)
        print("棋盤：%s" % "".join(c if c != " " else "." for c in sc.read_board(img)), flush=True)
        if sc.pile(piles, kind) is None:
            raise SystemExit("%s 疊的固定位置附近沒有料疊" % kind)
        if not go:
            print("（只量不動；加 --go 才會動）")
            return
        moved = True
        carrier.begin(M)
        if "--clear" in sys.argv:
            rec["clear"] = carrier.clear(sc)
        elif "--tidy" in sys.argv:
            rec["tidy"] = carrier.tidy(sc, kind)
        for cell in cells or []:
            print("\n格 %d：" % cell, flush=True)
            r = carrier.place(sc, kind, cell)
            rec.setdefault("placed", []).append(r)
            if not r["ok"]:
                break
        if cells:
            path = os.path.join(carrier.OUT, "place_%s.jpg" % time.strftime("%Y%m%d_%H%M%S"))
            cv2.imwrite(path, sc.last[0])
            print("照片 %s" % path, flush=True)
        print("\n完成", flush=True)
    except BaseException as e:
        rec["error"] = repr(e)
        print("停止：", repr(e), flush=True)
        raise
    finally:
        if moved:
            M.park_all()
        path = os.path.join(carrier.OUT, "grip_test_%s_%s%s.json" % (stamp, name, kind))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1, default=str)
        print("已記錄 %s" % path, flush=True)
        cam.close()
        M.close()


if __name__ == "__main__":
    main()
