# MG400 實機操作紀錄與工具

Dobot MG400（四軸桌上型機械手臂）以 TCP/IP 從電腦直接控制的完整流程。
本資料夾可獨立執行：SDK、腳本、報警碼表、官方協議文件都在裡面。

> 實測日期：2026-11-10。結果：連線、清報警、使能、MovJ／MovL 全部成功，到位誤差 < 0.0001 mm。

## 資料夾內容

| 檔案 | 用途 | 會不會讓手臂動 |
|---|---|---|
| `read_state.py` | 讀模式、報警（含中文說明）、姿態、關節角 | 不會 |
| `clear_error.py` | 清除報警 | 不會 |
| `safe_test.py` | 清報警 → 使能 → MovJ 到安全點 → MovL 小範圍來回 3 次 → 下使能 | 加 `--go` 才會 |
| `dual_sync.py` | 兩台同時運動並量測同步誤差（見第 7 節）；單台腳本也都支援 `--ip=` | 加 `--go` 才會 |
| `main.py` | 越疆官方範例（**不建議直接用**，原因見下方） | 會 |
| `dobot_api.py` | 越疆官方四軸 SDK（`Dobot-Arm/TCP-IP-4Axis-Python`），授權見 `LICENSE-dobot-sdk` | — |
| `files/alarm_*.json` | 控制器／伺服報警碼對照表 | — |
| `docs/*.pdf` | 官方四軸 TCP/IP 協議文件（中／英），指令格式以此為準 | — |

## 1. 環境

```bash
pip install numpy
```

Python 3.6+。**所有腳本都要在本資料夾內執行**（`cd` 進來），因為 `from dobot_api import ...` 是同層 import。

## 2. 網路設定（踩過的坑）

| 項目 | 值 |
|---|---|
| 手臂接口 | 底座 **LAN1** |
| 手臂 IP | `192.168.1.6`（LAN1 出廠預設） |
| 電腦乙太網路 | 手動設 `192.168.1.7` / `255.255.255.0`（同網段、不可與手臂相同） |
| 使用的埠 | 29999（Dashboard：控制／狀態）、30003（運動指令）、30004（即時回饋） |

**症狀：網路線接上但 ping 不通、DobotStudio 也搜不到設備。**
實測原因有兩個：

1. 電腦的乙太網路卡沒有 `192.168.1.x` 的位址（原本設的是別的網段）。
2. 電腦的 **Wi-Fi 也在 `192.168.1.0/24`**（家用路由器常見），封包被送去 Wi-Fi，
   ping 會出現「`Reply from 192.168.1.102: Destination host unreachable`」——是本機 Wi-Fi 自己回的。

**解法**：把乙太網路卡加上 `192.168.1.7`。若 Wi-Fi 也是 192.168.1.x 而仍走錯網卡，
可用系統管理員 PowerShell 加一條只給手臂的路由（不需關 Wi-Fi）：

```powershell
New-NetIPAddress -InterfaceAlias '乙太網路' -IPAddress 192.168.1.7 -PrefixLength 24
New-NetRoute -DestinationPrefix 192.168.1.6/32 -InterfaceAlias '乙太網路' -NextHop 0.0.0.0 -RouteMetric 1
```

**檢查方式**：

```powershell
Find-NetRoute -RemoteIPAddress 192.168.1.6 | Select-Object -Last 1   # 應該是「乙太網路」
ping 192.168.1.6                                                     # 應有回應、<1ms
foreach($p in 29999,30003,30004){ $c=New-Object Net.Sockets.TcpClient; "$p : " + $c.ConnectAsync('192.168.1.6',$p).Wait(2000); $c.Close() }
```

三個埠都要是 `True`。另外手臂要在 **TCP/IP 控制模式**（在 DobotStudio Pro 切換）。

## 3. 操作流程

```bash
python read_state.py          # 1. 先看狀態
python clear_error.py         # 2. 有報警就清（急停要先實體旋開）
python safe_test.py           # 3. 只印計畫、不動
python safe_test.py --go      # 4. 實際運動（周圍淨空、急停在手邊）
```

`safe_test.py --go` 實測輸出：

```
目前模式 9（報警），錯誤 [23]，姿態 X274.5 Y-34.9 Z120.9 R127.4
計畫：速度 20%，MovJ → [270.0, 0.0, 100.0, 127.4]，再 MovL [270,-40,100] ↔ [270,40,100] 共 3 次
清除報警 [23]
到位 [270.0, 0.0, 100.0, 127.419456]
第 1 次到位 [270.000018, -40.000001, 100.0, 127.419449]
...
回安全點 [270.0, 0.0, 100.0, 127.419456]
完成 ✓
```

要改動作範圍，調 `safe_test.py` 開頭的 `SAFE_X`、`SAFE_Z`、`SWING_Y`、`CYCLES`、`SPEED`。
程式任何一步失敗（回傳碼非 0、運動中報警、30 秒未到位）都會 `ResetRobot()` 停止並印原因；
結束或 Ctrl+C 一律 `DisableRobot()` 下使能。

## 4. 實測遇到的報警

| 碼 | 說明 | 原因 | 處理 |
|---|---|---|---|
| **85** | 硬體急停按鈕按下或軟體急停觸發 | 急停按鈕是按下狀態；此時 `EnableRobot()` 回 `-1` | 到現場旋開急停 → `clear_error.py`（模式 9 → 4） |
| **23** | 直線運動過程中規劃點超出工作空間 | 官方 `main.py` 從目前位置直接 `MovL` 到 `(20,280,-60)`，直線路徑中段出界，規劃階段即被拒 | 先用 `MovJ`（關節插補，不檢查直線路徑）到安全點，再做小範圍 `MovL` |

## 5. 官方 `main.py` 的問題

- `EnableRobot()` 回 `-1`（失敗）照樣往下送運動指令，並印「完成使能」。
- 報警時用 `input()` 問是否清除；沒有互動終端時該執行緒拋例外，**且當時握著共用鎖沒釋放**，
  等待到位的迴圈就永遠卡住。
- `point_a`／`point_b` 是範例值，不是針對這台手臂的位置，第一步就報 23。
- 無限迴圈，沒有結束條件。

所以實機測試請用 `safe_test.py`。

## 6. 協議重點（四軸）

- 回傳都是字串 `"ErrorID,{值,...},指令();"`，`ErrorID` 為 0 才是成功。
- 點到點 `MovJ(X,Y,Z,R)`、直線 `MovL(X,Y,Z,R)`、關節 `JointMovJ(J1,J2,J3,J4)`——**都是 4 個值**。
  別拿六軸（CR 系列）SDK 來控 MG400：六軸送 6 個值的 pose，四軸控制器不認，結果是關節點動會動、笛卡爾不動。
- `GetPose()`／`GetAngle()` 回 6 個值，後兩個固定是 0（協議補位，四軸用不到）。
- `RobotMode()` 常見值：4 未使能、5 使能空閒、7 運行中、9 報警。
- 速度：`SpeedFactor(1~100)` 全域比例；`SpeedJ`／`AccJ` 只影響 MovJ 類，`SpeedL`／`AccL` 只影響 MovL 類。

## 7. 雙臂（兩台 MG400 同時使用）

實測日期：2026-11-10。

| 手臂 | 手臂 IP | 接口 | 電腦網卡 | 電腦 IP |
|---|---|---|---|---|
| A | `192.168.1.6` | LAN1 | `乙太網路 11`（USB 網卡） | `192.168.1.3` |
| B | `192.168.2.7` | LAN2 | `乙太網路 10`（USB 網卡） | `192.168.2.3` |

- 兩台出廠 LAN1 IP 相同，不能並接同一網段；這裡一台走 LAN1、一台走 LAN2，各接一張網卡、不同網段。電腦端 IP 在 GUI（`ncpa.cpl`）設，需管理員權限。
- `read_state.py`／`clear_error.py`／`safe_test.py` 用 `--ip=192.168.2.7` 指定手臂，預設 `192.168.1.6`。
- 偶發：同時連線時 A 曾在 29999 逾時一次，重試即正常（`dual_sync.py` 內建 3 次重試）。

### 同步運動：`dual_sync.py`

```bash
python dual_sync.py                                  # 只讀、印計畫
python dual_sync.py --go --speed=50 --cycles=3 --accl=100 --movl-b=AccL=50
```

做法：每台一條執行緒，`Barrier` 讓兩台的指令同一瞬間送出；用 `Sync()` 阻塞等到位（不輪詢）；
各連 30004 回饋埠（約 125 Hz）以實際位置算起動／到位時間差。

| 設定 | 送出差 | 起動差 | 到位差（平均／最大） | 每段行程 |
|---|---|---|---|---|
| 兩個行程各跑 `safe_test.py`（參考） | 不可控 | — | — | — |
| `dual_sync.py` 預設參數，20% | 0.1 ms | 約 11 ms | **111／127 ms** | A 0.71 s、B 0.60 s |
| B 的 MovL 加 `AccL=25`，50% | 0.1 ms | 約 15 ms | 13.9／23.0 ms | 兩台 0.72 s |
| A `--accl=100`、B `AccL=50`，50% | 0.1 ms | 約 15 ms | **9.7／15.9 ms** | 兩台 0.63 s |

重點發現：
- 兩台在同樣設定下，同一段軌跡時間不同（B 較快約 0.1 s），統一全域 `SpeedL`／`AccL` **無效**；要在每個 MovL 後面帶 `AccL=…` 才能調，B 約需 A 的一半。
- 80 mm 的短行程是**加速度受限**，`SpeedFactor` 從 20% 拉到 50% 行程時間不變；要更快要提高 `AccL`。
- 到位差仍有約 10 ms，是回饋封包時間抖動加上兩台個別差異，若需要更緊要改用控制器端的時間戳（`controller_timer`）。
