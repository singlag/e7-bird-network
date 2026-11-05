# NEF golden vs NNIN 報告 — AF enrichment、0-detection root cause、class binding

日期：2026-11-05 · 樣本：10 張 Z8 NEF（`~/Documents/Z8_raw`）· model：BIRD (resource744) + NET4/6/7/9（Z8 body，由 emulator replay）
方法全部可複現：腳本喺 `~/.hermes/cache/scratch/`（`nef0_diag.py`、`obj_dist.py`、`gt_enrich.py`、`class_bind.py`、`test_e2e_fixture.py` log）。

## 1. 設計前提（user 確認過）

- NEF raw 嘅用途：攞機身記錄嘅 AF 框（NEF 得一個）同 model detection（可以多個）比較。
- 因此 1:1 peak matching 唔成立；有意義嘅統計量係 **golden box 附近嘅 activation enrichment**（box 對全 plane 基線）。
- 所有 10 張都係 `AFAreaMode = 3D-tracking`：機身揀邊個主體依賴 tracking context，單張 NEF 重構唔到。

## 2. Protocol（全部分析共用）

- 輸入：NEF 內 `SubIFD3` 384×256 NNIN strip（8256:5504 = 384:256，唔 crop，直接餵）。
- stage-0 activation maps 存檔：`gt_maps/<file>-<net>-s0.npy`（50 個 map，10 files × 5 nets）。
- 相機 AF 框（`Nikon:AFAreaX/Y/W/H` ÷ `AFImage 8256×5504`）映射到 feature grid；4 個方向變體（id/rot90cw/rot180/rot270cw），per-plane 攞 best variant（方向會改變邊個 cell 對應 golden box — 逐檔實測有分別）。
- enrichment ratio = relu(box mean) ÷ (relu plane mean + 1e-6)；null =  exhaustive/permutation（檔 ↔ 框 shuffle）。

## 3. Enrichment vs permutation null（per net）

| net | obs_mean | null_mean | null_p95 | 超過 null_p95 嘅檔 |
|---|---|---|---|---|
| BIRD | 11.15 | 13.10 | 21.0 | `00006` rot180 = 24.0 |
| NET4 | 19.43 | 22.20 | 27.0 | `00006` id = 30.2 |
| NET6 | 15.64 | 16.10 | 24.0 | `1783024282` rot180 = 31.9；`00014` id = 24.0 |
| NET7 | 17.15 | 17.30 | 25.0 | `00006` id = **42.7** |
| NET9 | 6.61 | 17.60 | 23.0 | 無；p=0.900，null-like |

判讀：整體 obs ≈ null baseline（無全局 enrichment），但有 **檔 × net 級 outliers**：`nikon-z8-raw-00006`（Animals）喺 BIRD/NET4/NET7 三個 net 都超 p95。NET9 handler candidate 唔算證據（null-like）。

## 4. 0-detection root cause（`4351185987.nef`，完整證據鏈）

1. `heads-00000.npz` replay：fine (22,16,24) + coarse (33,8,12) 有效，0 half-pattern errors；decoder visited 1056 cells、accepted 0。
2. gate 實測：`objectness_gate = score_gate = 0.10`；**全部 1056 cells objectness < 0.10**；golden 框（normalized center cx .449 / cy .592）映射到 fine `r9c10` / coarse `r4c5`，全部 anchor 低於 gate。→ 0 boxes 係 decoder 正確行為，唔係 bug。
3. A/B discriminator：同一張嘅 embedded preview PNG 經正常 pipeline 都係 0 accepted → 問題唔係 NEF 專屬。
4. Objectness 分布（全 1056 cells，sigmoid field 4）：

| 輸入 | max | p99 | mean | ≥0.10 |
|---|---|---|---|---|
| NEF NNIN strip | 0.0504 | 0.0087 | 0.00068 | 0 |
| preview PNG | 0.0048 | 0.0023 | 0.00009 | 0 |
| fixture_stripe | 0.0077 | 0.0011 | 0.00005 | 0 |
| fixture_random | 0.0012 | 0.0007 | 0.00007 | 0 |

NNIN strip 訊號最強（10× PNG、6.5× synthetic fixture），但仍然全部唔夠 gate。
5. Bit-exact anchor：`test_e2e_fixture.py` → `heads exact=True`（兩個 fixture reproduction 完全 bit-exact）→ torch reproduction 冇問題；flatness 係 artifact/context 限制（單 frame、無 tracking history/PDAF/NMS context）。

## 5. Class binding（#3：SubjectDetection vs category planes）

- 分組：`Nikon:SubjectDetection` Animals×5、People×2、Auto×3（Auto 唔入對比；SubjectDetection 可能係 menu 設定而唔係偵測結果 — 限制）。
- BIRD planes 解碼：`plane = anchor*11 + field`；field 4 = objectness，fields 5–10 = 六個 category scores（`decoder_port.py:362` 實證）。NET planes 無公開 field map，只按 index 報告。
- 統計：Animals vs People 絕對平均差，per-net 取 max-over-planes 做 statistic，**exhaustive permutation（C(7,2)=21，無取樣誤差）** already corrects multiple planes。

| net | top plane/category | diff | exhaustive p |
|---|---|---|---|
| BIRD | obj(a1) | 57.5 | 0.773 |
| NET4 | plane 8 | 43.7 | 0.545 |
| NET6 | plane 19 | 22.7 | 0.364 |
| NET7 | plane 25 | 10.4 | 0.727 |
| NET9 | plane 10 | 11.2 | 0.500 |

判讀：
- **冇 statistically supportable 嘅 category ↔ SubjectDetection binding**（全部 p ≥ 0.36；n=2 People 根本唔夠）。
- 值得記錄嘅訊號：BIRD **objectness(a1) 喺 2/5 張 Animals 相 enrichment 96× / 192×**（`00006`、`00014`），People 兩張都係 0.0 —— 方向同 net 假設（bird detector 對動物主體）一致但 n 太小，只算 anecdote。
- BIRD category plane 最高差異得 `cat1(a1)` diff 5.1 —— 無 binding 證據。

## 6. 外部樣本（raw.pixls.us，2026-11-05）

- 落咗 5 個 NEF 去 `~/Documents/Z8_raw`（`pixls-` 前綴，唔混入你自己嘅 shots）：
  Z8 lossless/HE、Z9 lossless/HE/HE*。五個都 parse 到 98,304 bytes NNIN strip。
- 兩個 Z8 樣本冇可用 AFInfo2（NNIN-only）— 只能做 parse/NNIN coverage 測試，唔做 golden。
- 三個 Z9 檔係同一 shot、同一 AF box（4905,2461；291×323 @ 8256×5504；mode 193 =
  **Single**，即 strong-subset 級）。但 NNIN strip 喺三種壓縮模式之間**逐 byte 唔同**
  （sha256 唔同）— camera 每次存檔會重寫 NNIN，所以 enrichment 只用 lossless 嗰個，
  統計上算 1 個獨立 sample。
- Enrichment vs permutation null（同一 protocol，golden 數 10→11，null N=2200，seed 11）：
  pixls-z9-lossless best-variant ratio — BIRD 10.0（null mean 13.45）、NET4 9.9（22.09）、
  NET6 15.8（15.45）、NET7 18.1（16.82）、NET9 7.7（18.09）。**沒一個 net 超過 null p95**
  → 呢個 external golden 無顯著 enrichment。原有 outliers 不變：00006（BIRD/NET4/NET7）、
  1783024282 + 00014（NET6）。11-file 版 aggregate：BIRD obs 11.05 / null 13.45 / p95 21；
  NET4 18.56 / 22.09 / 27；NET6 15.66 / 15.45 / 24；NET7 17.24 / 16.82 / 25；
  NET9 6.70 / 18.09 / 23（均 p 無變）。
- 解讀：Single-area AF 唔保證 NNIN 個 box 入面有 model 認定嘅主體（pixls scene 無 label，
  主體不明）；即 external golden 對 membership 冇加分。外部樣本嘅價值係 parse/NNIN 覆蓋，
  唔係 golden。

## 6a. Windows regression gate（已收，2026-11-05）

- User 喺 Windows 跑 `verify.py --case all`（Win10, Python 3.14.7, torch 2.9.1+cpu，
  GitHub `main` copy = commit `27b8507`）：**passed** — fine/coarse head diffs 0
  （11,616 halves/case）、defined_workspace diffs 0（122,880 halves/case）、
  `decoder_equal` true、records [] 對晒 fixtures、integer add/multiply 119,687 checks
  0 differences。torch reproduction 喺 Windows 都係 bit-exact。
- GUI 實測（同一機、同一 copy）：user 確認全流程 OK。

## 7. 限制（誠實列）

1. n=10、People=2 —— 任何 binding claim 都唔成立；要更多有標註嘅相先再做。
2. `SubjectDetection` tag 語義（設定 vs 偵測結果）未官方確認。
3. 方向變體係几何近似（input rotation ≈ map rotation），非嚴格等變。
4. NET4/6/7/9 roles 全部 hypothesis-level（Z8 firmware 無 model IDs）；NET planes 無 published field map。
5. 完整模型 regression gate（`verify.py --case all`）已喺 Windows 驗證通過（見 §6a）；
   Linux 端繼續用 `app/tests/*` + port decoder。

## 8. 下一步（體外決策）

- 收更多有明確主體類別嘅 NEF（尤其 People）先再跑 class binding。
- ~~Windows GUI 實測~~ ✅ 已喺 Windows 行過 GUI，全流程 OK（2026-11-05）。
- NET decode contract 如将来发表，先至可以俾 NET boxes。
