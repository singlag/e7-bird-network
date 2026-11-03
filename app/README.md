# app/ — GUI and CLI on top of the E7 bird network

Everything here is new.  The original package (`network.py`, `core.py`,
`resource744.body.bin`, `commands.json`, `consumer.py`, `verify.py`,
`wide_reference.c`, `wide-reference.dll`, fixtures, manifest hashes) is
untouched, so `verify_integrity.py` and `verify.py` still pass.

| Brief | Where |
| --- | --- |
| 1. GUI 介面（Windows / Linux 通用） | `gui.py` — Tkinter, stdlib, no extra GUI dependency |
| 2. 選擇 input photo or video | 「選擇輸入 …」 button (`Ctrl+O`), photo + video file filter |
| 3. 顯示選了的 input | right-hand canvas: image, or the first sampled video frame |
| 4. 選擇 Model | Model dropdown fed by `model_registry.py` (E7 bird model today) |
| 5. 執行 | 「執行」/「取消」 (`Ctrl+R`), worker thread, progress bar, log |
| 6. 顯示 model output + 識別目標框 | canvas redraws each processed frame with detection boxes plus the dashed 3:2 crop rectangle |
| 加：偵測框 list box | `ttk.Treeview` beside the preview — 序 / 類別 / 等級 / 機率 / token / slot / 框.  Boxes overlap, so one row can be selected: that box and the pixels inside it stay at 100%, every other box is not drawn at all, and the rest of the frame fades toward grey (`draw.dim_outside`).  「清除選擇」 shows all boxes normally again.  Clicking a column heading sorts the list by that column (numeric, not text); clicking it again reverses, and the heading shows ▲/▼.  Row order never moves the selection: item ids stay tied to the detection. |
| 加：token / combined probability / 行序 | `score.py` replays the decoder's per-cell gates off the head buffer, so every row carries `rank_order` (行序), `token`, `slot`, `combined_probability` and `objectness` — in the list box and in `detections.csv`.  The box label on the preview keeps only `#序 · p 機率`; class / rank / token / slot live in the list.  The 0–1000 `rank_score` is a calibrated rank off one of 24 curves, not a probability, and is only comparable inside one slot. |
| 加：語言 / Language | 「語言」 dropdown (中文 / English) relabels the whole window in place; settings, boxes and log survive the switch.  Window title is `E7 object detection simulator - GUI` in both languages. |

Non-3:2 input is centre-cropped to 3:2 **before** the 384×256 resize: height is
trimmed when the aspect is ≤ 1.5, otherwise width is trimmed (you cannot add
height); the crop actually used is printed in the log and stored in
`result.json`.

## Windows

```powershell
powershell -ExecutionPolicy Bypass -File app\setup_windows.ps1
..\e7-gui-env\Scripts\python.exe app\gui.py
```

The script creates `..\e7-gui-env`, installs `requirements.txt` (model) plus
`app/requirements.txt` (video I/O), and checks that `torch`, `numpy`, `Pillow`,
`imageio_ffmpeg` and `tkinter` import.  Offline install instead: use the
upstream wheelhouse exactly as the root `README.txt` describes.

On Windows the wide decoder defaults to the **bundled `wide-reference.dll`**
(the upstream x64 build), so nothing has to be compiled.  `app\build_decoder.ps1`
is only for building your own copy into `app\vendor\`.

Tkinter ships with the python.org installer; if 「執行」 fails with
`ModuleNotFoundError: tkinter`, reinstall Python with "tcl/tk and IDLE" checked.

## Linux / macOS

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r app/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
sudo apt install python3-tk          # Debian/Ubuntu only
./app/build_decoder.sh               # optional: native decoder (needs a C compiler)
.venv/bin/python app/gui.py
```

Without a built library the decoder falls back to `decoder_port.py`, a Python
port of `wide_reference.c` + `qsort-reference.c` that is asserted bit-equal to
the native library (see Tests).  `E7_DECODER_LIBRARY=/path/to/lib` overrides
which library is used.

## GUI controls

- 取樣/秒 — video frames sampled per second (default 1.0).  Model inference on
  CPU is ~10 s to ~170 s per frame, so full-frame-rate video is not realistic.
- 最大幀數 (0=auto) — auto cap is `min(60, fps)`: a 25 fps clip processes at
  most 25 sampled frames, a 60+ fps clip 60.
- 執行緒 — torch CPU threads.
- Decoder — `auto` (native library if available, else port), `native`, `port`.
- 儲存每幀 PNG / 儲存 heads (.npz) / 輸出標註影片 .mp4.
- 輸出目錄 — defaults to `<input-name>-e7-out` next to the input file.
- 語言 — 中文 / English; the window is rebuilt with the new labels and the
  current settings, boxes and log are kept.
- 5 · 偵測框 — click a row to spotlight that box: its pixels stay at 100%, every
  other box disappears, and the rest of the frame fades.  The row shows rank /
  class / rank_score / combined probability / token / slot / box geometry, and
  clicking a column heading sorts the list by that column (again = reverse).
  The box label itself only carries `#序 · p 機率`.  Keyboard: ↑/↓ through rows,
  「清除選擇」 to show all boxes at full strength.

Outputs: `annotated.png` (image input), per-frame PNGs (video, optional),
`annotated.mp4` (video, optional, drawn boxes baked in), `detections.csv`
(`rank_order`, `cx/cy/w/h` in 384×256 model space, `rank_score`, `cls`,
`token`, `slot`, `combined_probability`, `objectness`, the same box mapped to
source pixels, `crop_mode`, `decoder_status`), `heads-NNNNN.npz` (optional),
`result.json` (crop, sampling plan, timings, decoder status).  The CSV starts
with `#` comment lines explaining each column — readers should skip rows whose
first field begins with `#`.

## CLI

Same pipeline without the window — handy for scripting or checking a headless box:

```bash
python app/cli.py --input photo.jpg
python app/cli.py --input clip.mp4 --per-second 1 --max-frames 30 --save-heads --no-video
python app/cli.py --input clip.mp4 --decoder port --output out/
```

## Tests

```bash
python app/tests/test_crop.py            # 3:2 crop rules, fast
python app/tests/test_sampling.py        # stride / cap rules for video sampling
python app/tests/test_score.py           # rank / token / slot / combined probability replay
python app/tests/test_gui.py             # window title, language switch, box list, selected-box fade
python app/tests/test_decoder_parity.py  # port vs native library, needs a built/ bundled library
python app/tests/test_e2e_fixture.py     # fixture input -> network heads -> decoder (~3 min/fixture on CPU)
python app/gui.py --selftest path/to/file   # builds the window and loads the file, then exits
```

`test_gui.py` needs a display; it skips cleanly when Tkinter cannot open one.

`test_decoder_parity.py` finds the library automatically (bundled DLL on
Windows, `app/vendor/wide-reference.so|dylib` otherwise) or take
`E7_DECODER_LIBRARY=`.  It covers ties, suppression, blank heads, non-finite
heads and random head buffers — including the archived qsort's tie order.
