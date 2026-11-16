# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 這是什麼

兩台 Dobot MG400（四軸桌上型手臂，真空吸盤）＋俯視的 Orbbec Gemini 2 深度相機，下實體井字棋（OOXX）的機器人：人機對弈（語音／網頁）與機器自己對下。沒有測試框架、沒有 lint；「測試」是乾跑模式或對實機執行。

相關文件：`MIGRATION.md`（搬到另一台電腦／另一組手臂的步驟與所有現場常數）、`HISTORY.md`（逐日紀錄，查某個數字或某個坑的來由時看）、`MG400/README.md`（手臂 TCP 協議與網路踩坑）。

## 常用指令

Python 3.11，`pip install -r requirements.txt`。各資料夾的腳本用相對路徑與同層 import，**要在該資料夾內執行**。

```bash
# 對弈伺服器（Arena/）
python server.py                 # 雙臂（預設）：人機對弈＋機器對下，搬運走 carrier.py
python server.py --b-only        # 舊的 B 單臂路徑（robot.py）
python server.py --dry           # 乾跑：模擬手臂與盤面，不連手臂、不開相機
python dry_run.py                # 對乾跑伺服器跑完整情境（約 5 分鐘；需語音金鑰的項目沒金鑰會失敗）
python ctl.py [new_game|selfplay|selfplay_loop|loop_off|reset|stop|resume]   # 看狀態／下指令（走本機 18080）；loop_off＝這盤下完收好就停；reset＝手上那一步做完就收棋盤、回到開局前（網頁「↺ 重置」）
.\start_server.ps1 [旗標]        # 以獨立程序啟動，輸出到 out\server_run.log／server_err.log
#   經 ssh 遠端啟動時 Start-Process 的子程序會跟著連線結束被關：改用 Invoke-CimMethod Win32_Process Create（同 _setup_tools/startjog.ps1）

# 現場部署（GIGABYTE 筆電桌面）：OOXX一鍵部署.bat（＝python deploy.py：停舊的→查手臂→起伺服器→開網頁→選模式）、
#   OOXX現況檢查.bat（deploy.py --check）、手臂監控.bat（Console --watch：只看不控，可以跟對弈伺服器同時開）、停止OOXX.bat

# 手臂（MG400/），都支援 --ip=
python read_state.py             # 唯讀：模式、報警（中文）、姿態、關節
python clear_error.py            # 只清報警
python safe_test.py --go         # 會動

# 校正與現場檢查（Calib/）
python calib_box.py B [--go]     # 手眼校正（不加 --go 只印計畫）
python verify.py B 4 --z=-90 --go
python plan_cells.py             # 重算 cell_plan.json
python board_height.py --now     # 棋盤高度圖（棋盤要清空）
# 現場手動教點（Arena/）：python jog.py → http://127.0.0.1:18090 記四個交叉點＋中心；Calib/fit_teach.py 算座標；
#   python hover_check.py [--go] [B:4 A:2 …]  停在格子上方 20 mm 拍照記錄（會動）
python grip_test.py --go --place=6,7,8 A X   # （Arena/）不開伺服器的實機放子：A 拿 X、B 拿 O，臂和種類都要寫；--tidy 收回料疊、--clear 收整盤；不加 --go 只量
python stocks.py                 # 料區偵測，存 out/stocks.jpg
python piece_calib.py            # 棋子校正（桌面「棋子校正.bat」）：換一批棋子時量大小／厚度／顏色／圈叉樣子 → piece_profile.json；--reset 還原
python ox_model.py [影像]        # 辨識模型，存 out/ox_model.jpg

# 重新訓練辨識模型（Vision/，要有 GPU 版 torch＋ultralytics 的環境）
python make_labels.py && python build_dataset.py && python train.py 60
```

網頁：`https://<IP>:8443/?k=<Arena/access.txt>`（自簽憑證）；本機與 OBS 用 `http://127.0.0.1:18080/?k=…`。
開好幾個網頁時只有一個出聲（伺服器的 `sound_owner`＝最後按「開聲音」的分頁；OBS 頁 `obs=1` 照樣自己播）。

帶 `--go` 或啟動非乾跑伺服器後開局，**實體手臂會動**。

## 架構

資料流：`cam_worker.py`（子程序）→ `cam_full.jpg`＋`cam_depth.png` → `server.py` 主迴圈讀盤（`carrier.Scene.read_board`）→ 機器回合 `carrier.place` 量料區、規劃、搬運 → `mover.Arm`（移動、接觸偵測）→ `dual_sync.Arm`（TCP）→ 手臂。網頁每 0.35 秒輪詢 `/api/state`。

- **`Arena/server.py`**：唯一的長駐程式。`Controller._loop` 每 0.25 秒讀盤；人的回合偵測新落子，機器回合 minimax 選格 → `_move_piece`。HTTPS 8443＋本機 HTTP 18080。`HUMAN`／`ROBOT` 是模組層變數，人第一手放圈會 `set_roles()` 對調。
- **`Arena/carrier.py`**（2026-11-15 晚）：`Scene`＝相機（server 用 `FileCam` 讀檔、grip_test 用 `DirectCam`）＋格線交叉點換算（`teach_fit.json`）＋O／X 紀錄（`ox_record.json`）＋相機座標支架（`Calib/posts.py`，量完會換掉 `mover.hits_post`）。動作：`restore`（人的回合盤面被移動／拿走：刪去法搬回、看不到就從料疊補，分不出的不動）、`place`（每一步先看現況：已在格上→挪正；剛掉在棋盤上→撿回；轉送點上有同種→另一台接手；否則從料疊拿，搆不到經轉送點）、`tidy`／`clear`（收回各自料疊）、`settle`（偏 >4 mm 挪正）、`can_place`（只算不動）。`begin(M)` 兩台上電回停放點。
- **`Arena/mover.py`**：只剩手臂基本動作：`Arm.goto`（縮回→轉向→伸出）、`ik`／`post`／`leg_ok`、`touch()`（J2 電流接觸偵測＋高度把關才開真空）、`lift`；`Mover.clear_of_other`、`park_all`。`tuning.json` 存現場調過的 `TRAVEL_SPEED`／`VAC_DWELL`／`PRESS_B`。
- **`Arena/cam_worker.py`**：相機獨立成子程序（Orbbec SDK 斷線會殺掉整個程序），由 server 監看重啟。彩色＋深度同時常開；每張影像先套 `cam_align.json` 的對齊變換再輸出。
- **`Calib/`**：座標與視覺。`world.py`（影像 px ↔ 手臂 XY，用 `calib_A/B.json`）、`find_pieces.py`（讀棋盤）、`stocks.py`（料區：位置用彩色、顆數與高度用深度）、`ox_model.py`（YOLOv8n ONNX，OpenCV DNN 執行）、`align.py`（相機被碰歪時用棋盤格線求對齊變換）、`plan_cells.py`（支架禁入區 `hits_post`）、`common.py`（相機與手臂的共用包裝）。
- **`MG400/dobot_api.py`**：官方 SDK，不要大改。`dual_sync.Arm` 包三條連線：29999 指令、30003 運動、30004 即時回饋（1440 位元組封包）。
- **`Arena/services.py`**：語音辨識、語音合成、語音意圖，全是外部服務；下棋本身不依賴它們。
- **`Console/`**（雙臂手動控制台；`--watch` 只看不控，對弈時當手臂監控用）、**`Twin/`**（網頁數位孿生）是獨立工具，不在對弈流程裡。`Arena/robot.py` 是舊的 B 單臂路徑，只有 `--b-only` 會用到。

## 兩個座標系

紀錄與溝通以 **B 臂座標**為主（`ox_record.json`、料區的 `arm_xy["B"]`）；carrier 對每一台都直接用「影像→該臂」的換算（交叉點擬合），不經 A→B 轉換（`a_to_b_transform()` 只剩兩臂淨空檢查在用）。畫面上 B 的底座在上、A 在下，B 的 +X 朝下（往棋盤）、+Y 朝右。Z 是手臂自己的高度：`Z_HOVER=-40` 是移動高度，carrier 的棋盤面高度用各臂教點中心格的 Z（`carrier.BOARD_Z`：A −152.26、B −158.27），棋盤外用深度量；`Z_TOUCH_B≈-154.8`／`DZ_A` 是舊校正的值，只剩 robot.py 與幾支測試腳本在用。

## 不明顯但必須知道的事

**現場鐵律（使用者定的）**
- 不依賴人手：搬棋子、收棋盤一律由手臂做；吸不起來就換點、拖近、或換另一台。
- 不可以撞到相機支架。
- 真空**碰到才開**：`touch()` 量到的接觸高度若只比底下的桌面／棋盤面高不到門檻，就不開真空。曾經開著真空往下探，把整塊棋盤木板吸起來。
- 棋子目前**翻面吸背面**（使用者 2026-11-15：正面有概率吸不住；背面平，偏著吸也密合）。格子中心搆不到時：吸「棋子中心＋d」、吸盤放到「格子中心＋d」，棋子中心照樣落在格心（`carrier.reach_shift`，|d| ≤ 25 mm）；還不行就放到棋盤外的轉送點給另一台，**不可以借用棋盤上的格子**（對弈時會被當成一步）。
- 翻面後看不到 O／X，**不要自己搞混**：種類只靠 `Arena/ox_record.json`——`piles` 料疊固定位置＋負責的臂（X 疊在 A 側只有 A 拿、O 疊在 B 側只有 B 拿；人把料疊擺到別處時 `Scene.relearn_piles` 自動改位置：記錄位置 60 mm 內沒疊、負責那台那一側剛好一疊 ≥2 顆、搆得到、不在轉送點）；`board` 棋盤上每顆（放的時候記、收的時候刪、連續 6 次量不到才刪（開新局時對不到的直接刪））；`hand` 轉送點與上面那顆。沒有紀錄的棋子不猜、不收；人下的新棋子由讀盤記成人的種類。相機看到的符號只當參考（翻面的疊會被看錯）。

**手臂**
- 四軸：`MovJ/MovL(X,Y,Z,R)` 一律 4 個值。指令回傳 `"ErrorID,{值},指令();"`，`ErrorID==0` 才成功。
- 到位用回饋判斷（`common.sync_move`），不要用 `Sync()`（會提早回傳）。DO 用 `DOExecute`（立即），不要用 `DO`（排進運動佇列）。
- 移動走「縮回到小半徑 → 分段轉向 → 直線伸出」（`Arm.goto`）；直接遠距離 `MovL` 會報警 23。
- `RobotMode`：4 未使能、5 空閒、6 拖曳、7 運行、9 報警。報警 67／73＝手臂沒上電時被推到關節限位外，**軟體清不掉**，要人把大臂往前推；85＝急停。
- 手臂被重開電源後，舊 socket 不丟例外而是一直回空字串。`Arm.ik` 收到空回覆會丟 `ConnectionError`，server 會重連。
- 30004 回饋偶爾會卡死（連得上但沒資料，多半在網卡被重插之後），`dual_sync.Arm` 會自動改用 30005（200 ms），重開手臂電源才恢復。
- 不要同時開兩個會連手臂的程式（`Console/server.py`、`Arena/server.py`、校正腳本）：29999 互搶會回空字串。

**相機**
- Gemini 2 接在 USB 2.0 時撐不住 1280x720 彩色（會停 3～15 秒）。`cam_worker` 依連線類型自動選：USB 3 用 1280x720，USB 2 用 640x360 放大；環境變數 `CAM_HIRES` 可強制。
- 不要做彩色／深度的分時切換（會讓相機卡到出幾張就停）。真的卡住時 `device.reboot()` 可救回，`cam_worker` 內建。
- 相機被碰歪：`calib_check()` 在**棋盤清空時**用格線自動對齊（`align.py`）。棋盤上有棋子時格線被蓋住，估計會錯，所以只粗看不修正。棋盤本身若相對手臂移動，自動對齊會跟著錯，要重做手眼校正。

**視覺**
- 圈／叉由 `ox_model` 判斷，載入失敗自動退回顏色門檻（`stocks._kind_at`）。模型只解決光線變化，不解決相機或手臂移位。
- `Calib/piece_profile.json`（棋子校正產生；不存在＝原本那批棋子）覆蓋 `stocks` 的 `DISC_R_MM`／`PIECE_MM`／顏色範圍（`piece_mask`／`symbol_mask`）、px 門檻照 `SCALE` 縮放，`use_model:false` 關掉模型、改用樣板（`tpl_kind`）＋顏色規則。沒有這個檔時讀盤結果跟原本逐顆相同（13 張舊畫面比對過）。
- 料疊最上面那顆反光或被手遮住時會判不出種類：server 會沿用同一位置幾分鐘內看過的種類（`pile_mem`），並重看約 30 秒才放棄。
- 讀盤、量料區時手或手臂擋在上方就會失敗；程式的處理原則是「講清楚原因、重看、不要直接結束這盤」。

**改程式時**
- 現場常數散在多個檔案，完整清單與所在位置在 `MIGRATION.md` 第 5 節。對弈用的在 `carrier.py`（`BOARD_Z`、`SEED_RAW`、`PLACE_MARGIN`、`GRIP_OFF_MAX`…）、`ox_record.json`（料疊位置、轉送點）、`Calib/posts.py`（`POST_REF`／`POST_FOOT`）、`Calib/teach_fit.json`（教點）；`Z_TOUCH_B` 在 `mover.py` 與 `stocks.py` 各有一份（舊路徑用），要一起改。
- 手臂 IP 用專案根目錄的 `site.json` 覆蓋（範例 `site.example.json`），不要改程式裡的預設值。
- 金鑰不在版本庫：`services.py` 從環境變數或 `~/.claude/skill-secrets.env` 讀。`*.qianpro.shop` 的請求要帶一般的 User-Agent。
- PowerShell 5.1 把沒有 BOM 的 UTF-8 腳本當成 ANSI 讀，含中文的 `.ps1` 要存成 UTF-8 with BOM。
- 改完 `server.py`／`carrier.py`／`mover.py`／`Calib/*.py` 要重啟伺服器才生效；`index.html` 每次請求都重讀。重啟會丟掉進行中的那一盤。
