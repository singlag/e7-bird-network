# PAGED — GUI 逐場 storyboard v2（對齊已實裝嘅版面）

單視窗 Tkinter，title 固定為 **`E7 object detection simulator - GUI`**（中／英都係呢個名），
預設 1500×840，可放大，min 1040×660。版面：

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ 左欄 (控制)                    │  預覽 canvas          │ 5 · 偵測框 list box     │
│  語言 [中文 ▾]                 │   原圖 / 當前影格     │  序 類別 等級 機率       │
│  1 · 輸入   [選擇輸入 …]       │   ┌ ─ ─ 裁剪虛線 ─ ┐  │  token slot 框          │
│           filename · 1200×900  │   │ ▭ 框 + 標籤      │  ──────────────────      │
│  2 · Model  [E7 Bird ▾]        │   │   選咗嘅框 100%  │  ▸ 0  0  974 0.874       │
│  3 · 設定   取樣/秒, 最大幀數   │   │   其他位置轉淡   │    1  4  991 0.612       │
│           執行緒, Decoder       │   └ ─ ─ ─ ─ ─ ─ ─ ┘  │    2  2  898 0.431       │
│           ☑ 每幀PNG ☑ heads     │                        │  [清除選擇]            │
│           ☑ 標註影片 .mp4       │                        │  已選 #1：框內 100%…   │
│           輸出目錄 [____]       │                        │                        │
│  4 · 執行   [執行] [取消]       │                        │                        │
│           進度 ▓▓▓▓░░░░         │                        │                        │
│           status 一行           │                        │                        │
│           log (9 行)            │                        │                        │
├──────────────────────────────────┴───────────────────────────────────────────────┤
│ caption：預覽：影片第 0 幀。取樣計劃：每秒取 2 幀（stride 12），最多 3 幀        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## S0 剛啟動（未選檔）
- canvas 空白、caption「準備好 / ready — 選擇一個圖片或影片檔案」；`[取消]` disabled
- log：未執行前係空白（版本 / decoder 狀態喺按「執行」先 log，因為 model 係 lazy 載入）
- 「語言」下拉：中文 / English

## S1 已選圖片
- canvas 顯示**原圖**（fit canvas）+ 裁剪虛線框；caption「預覽：選了的圖片」
- input label：檔名 · 靜止圖片 · 尺寸
- status 行顯示裁剪描述（削高度／削寬度、裁剪區、送模型 384×256）
- 輸出目錄自動填 `<input stem>-e7-out`

## S2 已選影片
- canvas 顯示**第一幀**；caption 寫明取樣計劃（每秒取幾多幀、stride、最多幾多幀、cap = min(60, fps)）
- 改「取樣/秒」或「最大幀數」會即刻更新 caption 嘅預計

## S3 執行中
- worker thread 跑推論，UI 唔凍結；進度条 maximum = 將處理幀數
- 每幀完成：canvas 換成嗰幀 + 框 + 標籤（`cls · rank · p · tok`）、status 行顯示
  `幀 i · t=秒 · done/total · 推論秒 · 偵測 n 框 · crop mode`、list box 填返嗰幀嘅 row
- `[取消]` 喺當前幀跑完之後停，log「已取消 / cancelled」

## S4 完成（圖片／影片）
- status：`{processed}/{planned} 幀 · {detections} 框 · 每幀 {s}s · 輸出 {dir}`
  影片加 `標註影片：path`；被 cap 截斷會註明
- log：`完成：N 幀，M 個偵測框，T 秒` + `輸出：detections.csv …`
- list box 每 row：`序 (行序) | 類別 | 等級(0–1000 rank) | 機率(0–1 combined) | token | slot | 框座標`

## S5 選偵測框（重疊框逐個睇）
- click list box row（或 ↑/↓）：嗰個框變黃色加粗、**框內畫面 100% 原樣**、
  **其他位置全部混合向灰色（保留 35% 原圖）**；其他框變淡紅
- 「清除選擇」：全部框恢复正常顯示、冇淡化
- 冇選框時新影格唔會自動選（保持你原本嘅選取，直到超出框數先清）

## S6 語言切換
- 「語言」改 English → 全部 label、button、list 頭、status、caption、log 後續行都轉英文
- 已經選咗嘅偵測框、canvas、設定、舊 log 內容全部保留（重建 widget，唔清 state）
- 再轉返中文一樣

## S7 錯誤
- 讀取輸入失敗 → status「無法讀取輸入：…」；預覽失敗 → status 顯示原因
- 執行失敗 → status「失敗：…」+ log 完整 traceback
- 冇選檔按執行 → status「請先選擇輸入檔案」

## S8 匯出
- 圖片：`annotated.png`、`heads-NNNNN.npz`（可選）、`detections.csv`、`result.json`
- 影片：每幀 PNG（可選）、`annotated.mp4`（可選）、`detections.csv`、`result.json`
- `detections.csv` 頭 13 行係 `#` 註解，講清楚：`rank_score` 係 0–1000 校準**等級**（唔係機率，
  只能喺同一 slot 之內比較）、`rank_order` 係行序、`token/slot` 話畤你邊個 anchor 家族＋邊條曲線、
  `combined_probability` = sigmoid(class)*objectness（先係可比較嘅 0–1 可信度）

## 逐場驗收準則（已跑嘅自動化驗證）
- 裁剪數學：`app/tests/test_crop.py` ✓
- 取樣 stride／cap：`app/tests/test_sampling.py` ✓
- 等級／token／slot／機率 replay：`app/tests/test_score.py` ✓（用 crafted head 逐 cell 對返 decoder 自己嘅 candidate）
- title、語言切換、list box、選框 100%／其他轉淡：`app/tests/test_gui.py` ✓ + `app/gui.py --selftest` ✓
- port vs native decoder：`app/tests/test_decoder_parity.py`
- 原 package `verify.py` / `verify_integrity.py` 依然 pass（冇改原檔）
