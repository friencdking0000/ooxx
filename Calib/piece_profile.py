"""棋子外觀設定（Calib/piece_profile.json，由 piece_calib.py 量出來）。檔案不存在＝用程式裡原本那批棋子的值。

    load()  -> dict 或 None（"_dir"＝設定檔所在資料夾，樣板檔跟它放一起）
內容：disc_r_mm（霍夫量到的外緣半徑）、piece_r_mm、piece_mm（厚度）、
      piece_hsv（棋子本體＋符號＋背面的 HSV 範圍，[[lo, hi], …]）、symbol_hsv（圈／叉符號的 HSV 範圍）、
      use_model（現場這批棋子 ox_model 認得對才用）、templates（圈／叉符號平均樣子的 .npz 檔名）。
環境變數：PIECE_PROFILE=0 不用設定檔；PIECE_PROFILE_FILE 指定別的設定檔（piece_calib 存檔前驗證用）。
改了要重啟對弈伺服器才生效。"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HERE, "piece_profile.json")


def path():
    return os.environ.get("PIECE_PROFILE_FILE") or FILE


def load():
    p = path()
    if os.environ.get("PIECE_PROFILE") == "0" or not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        d["_dir"] = os.path.dirname(os.path.abspath(p))
        return d
    except Exception as e:
        print("[piece_profile] 讀不到 %s，用預設值：%s" % (p, str(e)[:80]), flush=True)
        return None
