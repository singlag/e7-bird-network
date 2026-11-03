# PLAN — e7-bird-network GUI fork

Fork 已建：`singlag/e7-bird-network`（https://github.com/singlag/e7-bird-network）
本地：`~/Developer/e7-bird-network`，remote `origin` = fork，`upstream` = 原版，工作 branch = `gui`
原 package 檔案（model body、core.py、network.py、run.py、verify.py、manifest hash）**一律不改**，GUI 全部放喺新目錄 `app/`，所以 `verify_integrity.py` / `verify.py` 繼續 pass。

## 需求對應

| 需求 | 做法 |
|---|---|
| 1 GUI，Python，Windows/Linux 通用 | Tkinter（stdlib）+ Pillow + numpy；無 Qt、無平台特定 code |
| 2 揀 input photo 或 video | 檔案對話框 + 副檔名過濾；支援拖入路徑 |
| 3 顯示所選 input | 右側大 canvas：靜圖顯示原圖；影片顯示當前幀 |
| 4 Model 選擇 | 下拉選單， registry 目前只有 `E7 Bird Network (resource744)`，預留擴充介面 |
| 5 執行 | 背景 thread 跑 inference + decode，進度列 + 可取消 |
| 6 顯示 model output + 識別框 | 同一個 canvas 疊加 detection 框（映射返原圖像素）+ 裁剪虛線框 |
| 7 3:2 自動裁剪 | center crop 到 1.5 aspect（優先削高度），再 bilinear resize 到 384×256 |
| 8 可信度唔再俾 0–1000 誤導 | `score.py` 由 head buffer replay 返 decoder 自己嘅 per-cell gates：行序 `rank_order`、`token`、`slot`、`combined_probability`（sigmoid(cls)·objectness）、`objectness`。CSV、list box、框標籤三處都帶；README 註明 `rank_score` 係 24 條曲線校準出嘅 0–1000 **等級**，只能同 slot 之內比較 |
| 9 重疊偵測框逐個睇 | canvas 右側 `ttk.Treeview`（序／類別／等級／機率／token／slot／框）；揀咗嘅 row：該框 100% + 框內像素原樣、其他位置混合向灰（留 35%），其他框變淡；「清除選擇」恢复 |
| 10 英文選擇 | `gui.py` 內 `I18N_EN` / `t()`（繁中＋英文兩套 key）；「語言」下拉重建 widget，state／選框／log 保留 |
| 11 視窗 title | `E7 object detection simulator - GUI`（中／英一致） |

## 新檔案

```
app/
  pipeline.py        # 幾何：3:2 center crop → 384×256 → u8 CHW[3,256,384]；框座標映射回原圖
  model_registry.py  # 模型註冊表（目前只有 E7 bird；介面目標：load()/infer()/decode()/score_table()）
  e7model.py         # E7 bird 模型 adapter：載入原 package network.py + 解碼器 + score_table accessor
  native_decoder.py  # Windows 用 bundled wide-reference.dll；Linux/macOS 用本地 build 嘅 .so/.dylib
  decoder_port.py    # 純 Python(numpy float32) 移植版，跨平台 fallback
  score.py           # 由 head buffer replay per-cell gates：rank_order / token / slot / combined_probability
  draw.py            # PIL 畫框、裁剪虛線框、dim_outside（選框之外混合向灰）
  frames.py          # 影格來源：靜圖 / 影片（imageio-ffmpeg）；取樣策略 plan_sampling()
  gui.py             # Tkinter GUI（主檔，含 I18N_EN / t() 語言切換、偵測框 list box、`--selftest`）
  job.py             # 一次 run 嘅編排：infer → decode → score → CSV/PNG/mp4
  cli.py             # 無 GUI 同款 pipeline，用嚟 test / batch
  build_decoder.sh   # Linux/macOS 由 wide_reference.c + qsort-reference.c build 共享庫
  build_decoder.ps1  # Windows 用 gcc/MinGW build DLL
  setup_windows.ps1  # Windows 環境一次性 setup（venv + requirements + DLL）
  README.md
  requirements.txt
  tests/             # test_crop / test_sampling / test_score / test_gui / test_decoder_parity / test_e2e_fixture
```

## 實作要點（決定咗嘅技術細節）

1. **裁剪規則**：`aspect = w/h`，目標 1.5。
   - `aspect ≤ 1.5`（例如 4:3、3:4）→ **削高度**（使用者要求）：`h' = w/1.5`，center 垂直裁剪。
   - `aspect > 1.5`（例如 16:9 超闊）→ 削高度物理上做不到（要「加」高度），此時削寬度 `w' = h*1.5`，UI 會顯示實際用了邊種。
   - 裁剪後 bilinear resize 到 384×256。**呢度同原版 `run.py` 唔同**：原版係直接 stretch（冇裁剪），我哋呢邊係保持比例嘅 crop + resize，UI 會註明。
2. **框座標映射**：模型輸出係 384×256 座標 → 原圖像素：`x*crop_w/384 + crop_x`、`y*crop_h/256 + crop_y`。所以可以喺**原圖**上面畫框，同時畫出裁剪範圍。
3. **解碼器跨平台**：bundled `wide-reference.dll` 係 Windows x64。方案：Windows 用 DLL；Linux/macOS 用 `build_decoder.*` 由原 C 源碼 build；如果兩者都冇 → 用 `decoder_port.py`（純 Python）。移植版會用現成嘅 Linux `.so` 做 bit 級對齊測試（隨機造 head 比對 records）。注意：原版 qsort 對同分 record 嘅次序係實作依賴，Python 版用穩定排序，同分情況可能唔同次序（會寫喺 README）。
4. **影片策略**：網路單幀喺呢部機 111 s（佢 Windows 機報 13.7 s），全 fps 跑唔到。預設「每秒取樣 1 幀」（`N = round(fps)`）。
   **處理幀數上限 = `min(60, round(fps))`**（使用者指定：正常上限 60 幀，但如果影片 fps < 60，就用影片嘅 fps 做上限）。UI 顯示呢個預設值並且可以改；如果影片長度超過上限（即只處理首 `上限` 秒），UI 會寫明「只處理咗首 N 秒，想全片請加大上限」。完成後可匯出標註 mp4 + CSV。播放用 timeline 逐幀睇，唔做即時推論。
5. **Thread 模型**：主 thread 只做 UI；推論行徑 worker thread，用 queue 傳進度；Cancel 旗幟；同一時間只允許一個 job（因為陀螺/晶片無關，但 torch CPU 多線程已用 `threads=` 參數）。
6. **Privacy**：GUI 產出（標註圖、mp4、CSV、log）一律寫去用戶揀嘅輸出目錄，唔寫入 package 目錄；`tools/audit_publication.py` 嘅 allowlist 會更新到包含 `app/`（保留 credential/私人路徑掃描）。

## 步驟

1. Fork + branch（已完成）
2. `app/pipeline.py` + 單元測試（4:3、3:4、16:9、1:1、已有 3:2 邊界 case）
3. `app/decoder_port.py` + `native_decoder.py` + 對齊測試（vs 現成 Linux `.so`，隨機造 head）
4. `app/frames.py`（imageio-ffmpeg 解幀、取樣、mp4 匯出）
5. `app/model_registry.py` + `app/e7model.py`
6. `app/cli.py`（先測通 pipeline，再上 GUI）
7. `app/e7gui.py`（Tkinter，按 PAGED.md 嘅画面狀態）
8. 驗證：(a) `verify.py --case all --heads-only` 仍 pass；(b) 端到端用真實鳥類照片出框；(c) 端到端用短片出標註 mp4；(d) Tk 喺 `DISPLAY=:0` 起得、截圖攞 UI 狀態
9. README / requirements-app.txt / build script；push 去 fork

## 風險 / 要留意

- **推論速度**：呢部機 111 s/幀；影片取樣幀數要克制（預設 ≤ 60 幀）。
- **偵測結果唔保證出框**：公開 fixture 係合成輸入，正確行為係 0 條 record。真實照片會唔會出框要實測；如果出唔到框，我会老实講，並用造嘅 head 證明解碼/畫框路徑係通。
- **Tk 喺 Linux 要 python3-tk**（呢部機有；Windows 內建）。
- **imageio-ffmpeg 需要 ffmpeg 二進位**（wheel 自帶；呢部機亦有系統 ffmpeg）。
- 原版 `run.py` 嘅 stretch 行為我唔會改，GUI 用自己嘅 crop+resize，兩者輸出會唔同（UI 會註明）。
