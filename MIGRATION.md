# 遷移說明：把 OOXX 對弈機器人搬到另一組手臂／另一台電腦

凍結版本：`v1.1-20261115`（git tag；v1.0 加上棋子辨識模型）。這一版在原現場完成過三盤真機機器對下、整盤自動收棋盤。
詳細的踩坑紀錄在 `CLAUDE.md`；這份只講「換地方要做什麼、照什麼順序」。

## 0. 前提（這套程式假設的硬體）

| 項目 | 假設 | 不一樣的話 |
|---|---|---|
| 手臂 | Dobot MG400 ×2（四軸、TCP 29999/30003/30004、1440 位元組回饋）| 換別的型號要改 `MG400/dobot_api.py` 以上整層，不是遷移而是移植 |
| 末端 | 真空方盒＋伸縮吸盤，`DO1`＝真空 | DO 編號不同：搜尋 `DOExecute(1,` |
| 俯視相機 | Orbbec Gemini 2（彩色 1280x720＋深度），從正上方看 | 內參會由校正吸收；解析度不同要改 `cam_worker.py`、`common.OrbbecCamera` |
| 側拍 | IPEVO V4K Pro（只給 OBS 直播／人看，程式流程不依賴）| 可省略 |
| 棋子 | 直徑約 84 mm、厚約 3.9 mm、黑底綠色 O／X、頂面貼膠帶 | 改 `Calib/stocks.py` 的 `PIECE_R_MM`、`PIECE_MM`、`GREEN` |
| 擺位 | 兩臂面對面、棋盤在中間、B 在畫面上方、A 在下方 | 只用一台也可以（`--b-only`，舊路徑）|

只有一台手臂時：把它當 **B**（IP 寫在 `site.json` 的 `arm_ips.B`），人機對弈全部功能可用；機器對下也可以，只要 O、X 料疊都放在它搆得到的地方。

## 1. 新電腦安裝

```powershell
# Python 3.11
pip install -r requirements.txt
```

- Orbbec：**相機盡量接 USB 3.0**。原現場接在 USB 2.0，所以程式是「分時」切深度（平常只開彩色、要量料區才短暫開深度），USB 3 下一樣能跑。新電腦若深度出不來，先查 Orbbec 官方的 Windows UVC metadata 註冊步驟（原電腦上有沒有做過，沒有紀錄）。
- 金鑰不在版本庫裡，新電腦要自己設環境變數：
  - `ASR_GATEWAY_API_KEY`（語音辨識）
  - `QIANPRO_STUDIO_TOKEN`（語音合成）
  - `ARENA_LLM_BASE`、`ARENA_LLM_KEY`、`ARENA_LLM_MODEL`（語音意圖）
  - 沒設也能跑：只是語音功能會失敗，下棋不受影響。
- `Arena/access.txt`、`key.pem`、`cert.pem` 第一次啟動會自動產生。

## 2. 網路

1. 兩台手臂各接一張網卡、不同網段（原現場：A `192.168.1.6`、B `192.168.2.7`）。
2. 複製 `site.example.json` → `site.json`，填新手臂的 IP。
3. 驗證（唯讀，不會動）：`cd MG400; python read_state.py --ip=<IP>`。

## 3. 先乾跑，確認程式本身沒問題（不連手臂、不開相機）

```powershell
cd Arena
.\start_server.ps1 --dry
python dry_run.py
```

## 4. 現場校正（**順序不能換**；每一步都會產生後面要用的檔案）

以下會讓手臂動，旁邊要有人、急停在手邊。

| # | 做什麼 | 指令（在 `Calib/`）| 產出 |
|---|---|---|---|
| 1 | 方盒頂面模板。末端方盒同款、相機高度相近就沿用現有的；外觀不同要重做：手臂 J1=0°、Z=−40 停在畫面中間，從俯視畫面把方盒頂面裁成正方形存檔（**沒有現成腳本**，要手動裁）| — | `box_face.png`（B）、`box_face_A.png`（A）|
| 2 | 量棋盤面到相機的深度，改常數 | — | `calib_box.py` 的 `D_BOARD`；`stocks.py` `STD_TOP_DEPTH`、`find_pieces.py` `PIECE_TOP`（＝D_BOARD − 棋子厚）|
| 3 | 手眼校正（每台）| `python calib_box.py B --go`、`python calib_box.py A --go` | `calib_B.json`、`calib_A.json` |
| 4 | 外推驗證（應 < 4 mm）| `python verify.py B 4 --z=-90 --go` | — |
| 5 | 支架／障礙物禁入區（影像座標）| 改 `plan_cells.py` 的 `POSTS` | — |
| 6 | 每格由誰下 | `python plan_cells.py` | `cell_plan.json` |
| 7 | **吸盤碰到棋子頂面的 Z**：在格 4 放一顆棋子 | `python touch_test.py <格4的B座標x,y> <預估Z>` | `Arena/mover.py` `Z_TOUCH_B`、`stocks.py` `Z_TOUCH_BOARD`（兩處要一樣）|
| 8 | 棋盤不平修正（棋盤清空）| `python board_height.py --now` | `board_dz.json` |
| 9 | 校正基準照（兩臂都在停放點、棋盤清空）| 把 `Arena/cam_full.jpg` 複製成 `Calib/calib_ref_scene.png` | `calib_ref_scene.png` |
| 10 | 料區偵測看得對不對 | `python stocks.py`（看 `out/stocks.jpg`）| — |

## 5. 跟現場綁定的常數（校正檔以外，要人工檢查的）

| 檔案 | 常數 | 意義 | 原現場的值 |
|---|---|---|---|
| `Arena/mover.py` | `Z_TOUCH_B` | B 吸盤碰到格 4 棋子頂面的 Z | −154.82 |
| | `DZ_A` | A 的 Z 比 B 高多少（兩臂安裝高度差）| 3.0 |
| | `DI_J2` | 接觸偵測：J2 電流變化門檻（A）| 0.12 |
| | `Arm.COMPRESS` | 伸縮吸盤壓縮量 | 4.5 mm |
| | `Arm.park`（A）、`Arm.R_INS` | A 停放點、轉向半徑候選 | (42,236)；215/200/185/170 |
| `Calib/play.py` | `PARKS`、`Z_HOVER`、`R_IN`、`CLEAR_MM` | 停放點、移動高度、轉向半徑、兩臂淨空 | Z_HOVER −40 |
| `Calib/plan_cells.py` | `POSTS`、`BOX_HALF_PX` | 相機支架禁入區（影像像素）| 左 (0,330,380,470)、右 (990,360,1280,480) |
| `Calib/calib_box.py` | `D_BOARD`、`POSES`、`BASE_PX` | 棋盤面深度、校正姿態、底座在影像的位置 | 540 mm |
| `Calib/drift_check.py` | `BASES` | 兩個底座在影像中的框（校正漂移檢查用）| 上 (480,0,780,110)、下 (510,605,810,720) |
| `Calib/stocks.py` | `STD_TOP_DEPTH`、`PIECE_MM`、`PIECE_R_MM`、`GREEN`、`DARK_V_MAX` | 棋子頂面深度、厚度、半徑、顏色門檻 | 536.3／3.9／42 |
| `Arena/carrier.py` | `BOARD_Z` | 各臂吸盤碰到空棋盤的 Z（教點中心格）| A −152.26、B −158.27 |
| | `SEED_RAW` | 原始影像上格線四個交叉點的大概位置（搜尋起點）| c0 (544,284.5)… |
| | `PLACE_MARGIN`、`GRIP_OFF_MAX`、`TOL` | 放子最深＝頂面＋2 mm；偏吸上限；挪正門檻 | 2／25／4 mm |
| `Arena/ox_record.json` | `piles`、`hand` | 兩疊的固定位置（B 座標）與負責的臂；棋盤外轉送點 | X [439,−190] A、O [124,193] B；轉送點 [307.1,187.5] |
| `Calib/posts.py` | `POST_REF`、`POST_FOOT` | 支架柱子大概位置（相機座標 mm）、腳座前後範圍 | 左 (−214,0)、右 (286,5) |
| `Calib/teach_fit.json` | — | 手動教點擬合（每台的格心與交叉點）| `jog.py`＋`fit_teach.py` 重做 |
| `Arena/server.py` | 憑證用的 IP 清單（檔尾）| HTTPS 自簽憑證含的本機 IP | 改成新電腦的 IP |

換光線後若 O／X 判錯：先調 `stocks.GREEN`、`DARK_V_MAX`（`find_pieces.py` 也有一份）。

## 5b. 棋子辨識模型（v1.1 新增，換光線不用調門檻）

- `Vision/ox.onnx`：訓練好的 YOLOv8n，偵測 O／X 棋子；執行期由 `Calib/ox_model.py` 用 **OpenCV DNN** 跑（CPU 約 0.1 秒一張），**新電腦不用裝 PyTorch、不用 GPU**。
- 用途：判斷「是不是棋子、是圈還是叉」，並補上顏色門檻在新光線下漏抓的棋子。位置、高度、手眼換算仍是原本的幾何／深度方法。
- 模型檔不見或載入失敗會自動退回原本的顏色規則（伺服器會印 `[ox_model] 模型載不起來`）；`OX_MODEL=0` 可強制關掉。
- 驗證（沒參與訓練的 36 張、174 顆棋子）：模型對 171、錯 3、漏 0；原本的顏色規則對 132、錯 24、看不出來 18。三盤終局畫面在變暗／過亮／偏黃下，模型全對，顏色規則在變暗、過亮時讀錯。
- 現場快速確認：`cd Calib; python ox_model.py`（存 `out/ox_model.jpg`，每顆棋子畫圈＋信心值）。
- **模型不能解決的**：相機或手臂被移動（要重做第 4 節的校正）、棋子換款式（先跑下面的「棋子校正」；還是不行再重新訓練）。

### 棋子換了一批（大小、厚度、顏色、圖案不同）：棋子校正

桌面「棋子校正.bat」（＝`cd Calib; python piece_calib.py`，手臂不會動，會先停掉對弈伺服器）：
1. 棋盤清空 → Enter　2. 九格中央各放一顆、正面朝上：**四個角和正中間放圈，其他四格放叉** → Enter　3. 九顆翻面 → Enter（不做輸入 s）
它量出半徑、厚度、本體／符號／背面的顏色，試三種認法（原本的模型、校正樣板、顏色規則）各認對幾顆，再用新設定跑一次真正的讀盤驗證，
輸入 y 存成 `Calib/piece_profile.json`＋`piece_templates.npz`，重啟伺服器（一鍵部署）生效。模型認錯這批棋子時自動不用模型。
回到原本那批：`python piece_calib.py --reset`。樣本影像存在 `Calib/out/piece_calib/<時間>/`（之後重訓可用）。
棋子很薄（< 2.5 mm）或大小差很多時它會警告：吸取的接觸門檻（carrier `piece_spot` 的 `h_min`）、吸取偏移、轉送點間距要實機重看。
- 重新訓練（要有 GPU 版 PyTorch＋ultralytics 的環境，約 8 分鐘）：
  1. 把新現場的俯視畫面（1280x720）放進 `Calib/out/`
  2. `cd Vision; python make_labels.py`（用現有規則自動標註，輸出 `work/crops_*.jpg` 檢查用拼圖）
  3. 看拼圖，把標錯的編號填進 `build_dataset.py` 的修正表，`python build_dataset.py`
  4. `python train.py 60` → 產生 `ox.onnx`
- 已知不足：訓練影像只來自原現場一種桌面與燈光（靠合成的亮度／色溫／反光擴增），到差異很大的現場最好補拍幾十張重訓。

## 6. 啟動與操作

```powershell
cd Arena
.\start_server.ps1            # 人機對弈（只用 B）
.\start_server.ps1           # 雙臂（預設）
python ctl.py                 # 看狀態
python ctl.py selfplay        # 機器對下一盤
python ctl.py selfplay_loop   # 連續對下（直播）
python ctl.py stop
```

網頁：`https://<電腦IP>:8443/?k=<Arena/access.txt 的內容>`；OBS 瀏覽器來源用 `http://127.0.0.1:18080/?k=…&obs=1`。
不要同時開 `Console/server.py`（兩邊會搶手臂連線）。

## 7. 上線前檢查表

- [ ] `read_state.py` 兩台都連得上、無報警
- [ ] 乾跑 `dry_run.py` 全過
- [ ] `verify.py` 外推誤差 < 4 mm
- [ ] 伺服器啟動時印「校正正常」
- [ ] `python stocks.py` 每一疊的種類、顆數正確
- [ ] 單手測試：`python ctl.py selfplay` 第一手吸得起來、放得準
- [ ] 人站在旁邊、急停在手邊看完第一盤

## 8. 已知限制（這一版沒解決的）

- X 棋子頂面膠帶皺，吸取成功率比 O 低；程式會換吸取點重試、歪了不一定救得回來。
- A 在遠離校正範圍的一側定位誤差可到 15～20 mm。
- 收棋盤時偶爾有吸不起來的棋子被來回拖（2026-11-15 格 4 的 O，原因未查明）。
- 一盤約 13～17 分鐘。
- 網頁沒有「一鍵重新校正」，相機或手臂被移動後要照第 4 節重做。
- 花式交接（放到手掌）已停用（`--handover` 才開），未在真機驗證。
