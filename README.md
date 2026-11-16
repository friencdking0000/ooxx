# OOXX 井字棋機器人 v1.1

兩台 Dobot MG400 四軸手臂（真空吸盤）加一台俯視的 Orbbec Gemini 2 深度相機，在實體棋盤上下井字棋。

- **人機對弈**：人在棋盤上落子，機器看盤、用 minimax 選格，再由手臂把棋子搬上去。可以用網頁或語音操作。
- **機器對下**：兩台手臂輪流下，可以連續下好幾盤。下完會自動把棋盤收乾淨。
- **網頁介面**：即時盤面、相機畫面、開新局／重置／停止，也有給 OBS 直播用的頁面。

## 硬體

| 項目 | 規格 |
|---|---|
| 手臂 | Dobot MG400 ×2（A、B），末端裝真空吸盤 |
| 相機 | Orbbec Gemini 2，俯視整個桌面，建議接 USB 3 |
| 電腦 | Windows 11，Python 3.11 |
| 棋子 | 圓形 O／X 棋子，翻面後用背面吸 |

## 安裝

```bash
pip install -r requirements.txt
copy site.example.json site.json   # 填入兩台手臂的 IP
```

## 快速開始

各資料夾的腳本都用相對路徑，**要在那個資料夾裡執行**。

```bash
cd Arena
python server.py --dry      # 乾跑：不連手臂、不開相機，先確認程式正常
python server.py            # 實機：雙臂對弈伺服器（開局後手臂會動）
python ctl.py               # 看狀態；也可加 new_game / selfplay / stop / reset
```

網頁：

- 本機：`http://127.0.0.1:18080/?k=<存取碼>`
- 區網：`https://<電腦 IP>:8443/?k=<存取碼>`（自簽憑證）

存取碼在 `Arena/access.txt`，第一次啟動時自動產生，沒有放進版本庫。

現場電腦的桌面捷徑：`OOXX一鍵部署.bat`（停掉舊的、檢查手臂、啟動伺服器、開網頁）、`OOXX現況檢查.bat`、`手臂監控.bat`、`停止OOXX.bat`。

## 資料夾

| 資料夾 | 內容 |
|---|---|
| `Arena/` | 對弈伺服器 `server.py`、搬運邏輯 `carrier.py`、手臂動作 `mover.py`、相機子程序 `cam_worker.py`、網頁 `index.html`、語音服務 `services.py` |
| `Calib/` | 座標換算、手眼校正、讀盤、料區偵測、O／X 辨識模型（`ox_model.py`）、棋子校正 |
| `MG400/` | Dobot 官方 SDK 與雙連線包裝 `dual_sync.py`；協議說明見 `MG400/README.md` |
| `Vision/` | 辨識模型訓練（YOLOv8n → ONNX） |
| `Console/` | 雙臂手動控制台／唯讀監控 |
| `Twin/` | 網頁數位孿生 |
| `Gemini2/` | 相機測試工具 |
| `_setup_tools/` | 現場架設與除錯時用的零散腳本 |

## 運作流程

```
cam_worker.py（子程序）─→ 彩色＋深度影像
        │
server.py 主迴圈每 0.25 秒讀盤
        │  人的回合：偵測新落子
        │  機器回合：minimax 選格
        ▼
carrier.place：量料區 → 規劃 → 搬運（搆不到就經轉送點交給另一台）
        ▼
mover.Arm（移動、接觸偵測）→ dual_sync.Arm（TCP）→ MG400
```

## 搬到新場地或新電腦

照 [`MIGRATION.md`](MIGRATION.md) 做：安裝 → 網路 → 乾跑 → 現場校正（順序不能換）→ 檢查現場常數 → 上線前檢查表。

## 安全注意

- 帶 `--go` 的腳本、非乾跑的伺服器開局之後，**實體手臂會動**。
- 不要同時開兩個會連手臂的程式（對弈伺服器、控制台、校正腳本會互搶連線）。
- 手臂不可以撞到相機支架；真空只在吸盤碰到棋子後才開。

## 其他文件

- [`MIGRATION.md`](MIGRATION.md)：遷移步驟與所有現場常數
- [`HISTORY.md`](HISTORY.md)：逐日開發紀錄
- [`CLAUDE.md`](CLAUDE.md)：架構細節與已知的坑
- [`MG400/README.md`](MG400/README.md)：手臂 TCP 協議與網路踩坑
