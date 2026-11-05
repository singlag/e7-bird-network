"""E7 object detection simulator — GUI (Tkinter, so it runs the same way on Windows and Linux).

    python app/gui.py                 (Windows:  py -3 app\\gui.py)

Requirements from the brief:
  1. GUI 介面                    — this window (Tkinter, stdlib)
  2. 選擇 input photo or video   — 「選擇輸入」 button, photo/video/NEF RAW file filter
  3. 顯示選擇了的 input          — canvas shows the image / sampled video frames
  4. 選擇 Model                  — dropdown fed by `model_registry` (E7 bird model today)
  5. 執行                        — run / cancel buttons, worker thread, progress bar, log
  6. 顯示 model output + 目標框  — the canvas redraws each frame with detection boxes and
                                   the dashed 3:2 crop rectangle drawn on the original frame

Added on top of the brief:
  * Detection list beside the preview.  Boxes overlap, so one row can be selected;
    the selected box and the pixels inside it stay at 100% while everything else is
    faded (`draw.dim_outside`).
  * Per-box `rank` / `combined probability` / `token` / `slot` in the list, on the
    boxes and in `detections.csv` — the decoder's 0–1000 score is a calibrated rank
    read off one of 24 curves, so it is only comparable within the same slot.
  * 語言 / Language switch (中文 / English) — rebuilds every label in place.

Input that is not 3:2 is centre-cropped to 3:2 first (trimming height where
possible, width otherwise) and then resized to 384x256 — see `pipeline.py`.
NEF raw files bypass the crop: the camera's own 384x256 NNIN strip is fed to
the model as-is, and the maker-note AF box is shown in green on the preview
plus as an `AF` row in the detection list — see `nef.py`.
"""

import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

APP = Path(__file__).resolve().parent
for path in (APP, APP.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, scrolledtext, ttk

from PIL import Image, ImageTk

import draw
import frames as frames_module
import job
import model_registry
import nef
import pipeline

WINDOW_TITLE = "E7 object detection simulator - GUI"
IMAGE_GLOBS = "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.gif"
VIDEO_GLOBS = "*.mp4 *.mov *.avi *.mkv *.webm *.m4v *.wmv *.mpg *.mpeg *.3gp"
RAW_GLOBS = "*.nef *.NEF *.nrw *.NRW"

STRINGS = {
    "zh": {
        "section_input": "1 · 輸入",
        "choose": "選擇輸入 …",
        "no_input": "(未選擇)",
        "section_model": "2 · Model",
        "section_settings": "3 · 設定",
        "per_second": "取樣/秒",
        "max_frames": "最大幀數 (0=auto)",
        "threads": "執行緒",
        "decoder": "Decoder",
        "save_frames": "儲存每幀 PNG",
        "save_heads": "儲存 heads (.npz)",
        "export_video": "輸出標註影片 .mp4",
        "output_dir": "輸出目錄",
        "section_run": "4 · 執行",
        "run": "執行",
        "cancel": "取消",
        "language": "語言",
        "ready": "準備好 / ready — 選擇一個圖片或影片檔案",
        "pick_title": "選擇圖片、影片或 NEF RAW",
        "file_all": "圖片、影片及 RAW",
        "file_image": "圖片",
        "file_video": "影片",
        "file_raw": "NEF RAW",
        "file_every": "所有檔案",
        "no_source": "(未有輸入)",
        "preview_image": "預覽：選了的圖片",
        "preview_video": "預覽：影片第 0 幀。取樣計劃：每秒取 {per_second} 幀（stride {stride}），最多處理 {planned} 幀（cap = min(60, fps)）",
        "select_input_first": "請先選擇輸入檔案",
        "running": "執行中 …",
        "model_loaded": "模型已載入：{name}",
        "model_load_failed": "模型載入失敗：{error}",
        "model_not_runnable": "呢個 model 唔可以喺呢個 program 執行 — {note}",
        "model_details": "模型資料",
        "md_name": "名稱", "md_file": "檔案", "md_location": "位置",
        "md_role": "類型／角色假設", "md_blocks": "Layer blocks", "md_input": "輸入",
        "md_head": "Head 輸出", "md_bytes": "Body bytes", "md_params": "fp16 參數 words",
        "md_sha": "SHA-256", "md_runnable": "可執行", "md_boxes": "可出框",
        "md_yes": "係", "md_no": "冇", "md_source": "來源",
        "md_data": "Model data 來源",
        "run_header": "=== 執行 {name} → {output} ===",
        "cancelled": "要求取消 …",
        "job_cancelled": "已取消 / cancelled",
        "sampling": "取樣：stride {stride} 幀，處理 {planned} 幀{truncated}",
        "truncated": "（已按 cap 截斷，只處理首 {planned} 幀）",
        "frame_status": "幀 {frame_index} · t={seconds}s · {done}/{total} · 推論 {frame_seconds}s · 偵測 {detections} 框 · crop {crop_mode}",
        "done_log": "完成：{processed} 幀，{detections_total} 個偵測框，{total_seconds} 秒",
        "finish_summary": "{processed}/{planned} 幀 · {detections} 框 · 每幀 {per_frame}s · 輸出 {output}",
        "finish_video": "\n標註影片：{path}",
        "finish_cap": "\n（cap = min(60, fps)，可用「最大幀數」放大）",
        "exports": "輸出：{names}",
        "failed": "失敗：{error}",
        "unreadable": "無法讀取輸入：{error}",
        "preview_failed": "預覽失敗：{error}",
        "section_boxes": "5 · 偵測框",
        "clear_selection": "清除選擇",
        "no_boxes": "(未有偵測框 — 未執行或冇偵測到)",
        "box_selected": "已選 #{index}：框內 100% 顯示，其他位置轉淡",
        "box_none": "冇選偵測框：全部框正常顯示",
        "kind_video": "影片",
        "kind_image": "靜止圖片",
        "kind_nef": "NEF RAW",
        "af_log": "機身 AF 框：{box}（{mode}）— 綠色顯示喺預覽",
        "col_rank": "序",
        "col_cls": "類別",
        "col_score": "等級",
        "col_p": "機率",
        "col_token": "token",
        "col_slot": "slot",
        "col_box": "框 (384x256)",
    },
    "en": {
        "section_input": "1 · Input",
        "choose": "Choose input …",
        "no_input": "(none selected)",
        "section_model": "2 · Model",
        "section_settings": "3 · Settings",
        "per_second": "Samples/sec",
        "max_frames": "Max frames (0=auto)",
        "threads": "Threads",
        "decoder": "Decoder",
        "save_frames": "Save every frame as PNG",
        "save_heads": "Save heads (.npz)",
        "export_video": "Export annotated .mp4",
        "output_dir": "Output folder",
        "section_run": "4 · Run",
        "run": "Run",
        "cancel": "Cancel",
        "language": "Language",
        "ready": "Ready — pick an image or video file",
        "pick_title": "Choose an image, video or NEF raw",
        "file_all": "Images, videos and RAW",
        "file_image": "Images",
        "file_video": "Videos",
        "file_raw": "RAW (NEF)",
        "file_every": "All files",
        "no_source": "(no input)",
        "preview_image": "Preview: the selected image",
        "preview_video": "Preview: video frame 0. Sampling: {per_second} frames/sec (stride {stride}), at most {planned} frames (cap = min(60, fps))",
        "select_input_first": "Pick an input file first",
        "running": "Running …",
        "model_loaded": "Model loaded: {name}",
        "model_load_failed": "Model load failed: {error}",
        "model_not_runnable": "This model cannot run in this program — {note}",
        "model_details": "Model data",
        "md_name": "Name", "md_file": "File", "md_location": "Location",
        "md_role": "Type / role hypothesis", "md_blocks": "Layer blocks", "md_input": "Input",
        "md_head": "Head output", "md_bytes": "Body bytes", "md_params": "fp16 param words",
        "md_sha": "SHA-256", "md_runnable": "Runnable", "md_boxes": "Detection boxes",
        "md_yes": "yes", "md_no": "no", "md_source": "Source",
        "md_data": "Model data taken from",
        "run_header": "=== run {name} → {output} ===",
        "cancelled": "Cancellation requested …",
        "job_cancelled": "Cancelled",
        "sampling": "Sampling: stride {stride} frames, processing {planned} frames{truncated}",
        "truncated": " (truncated to the first {planned} frames by the cap)",
        "frame_status": "frame {frame_index} · t={seconds}s · {done}/{total} · inference {frame_seconds}s · {detections} boxes · crop {crop_mode}",
        "done_log": "Done: {processed} frames, {detections_total} detection boxes, {total_seconds} s",
        "finish_summary": "{processed}/{planned} frames · {detections} boxes · {per_frame}s/frame · output {output}",
        "finish_video": "\nAnnotated video: {path}",
        "finish_cap": "\n(cap = min(60, fps) — raise \"Max frames\" to process more)",
        "exports": "Exports: {names}",
        "failed": "Failed: {error}",
        "unreadable": "Cannot read input: {error}",
        "preview_failed": "Preview failed: {error}",
        "section_boxes": "5 · Detections",
        "clear_selection": "Clear selection",
        "no_boxes": "(no detections — not run yet, or none found)",
        "box_selected": "Selected #{index}: box interior at 100%, everything else faded",
        "box_none": "No box selected: all boxes shown normally",
        "kind_video": "video",
        "kind_image": "still image",
        "kind_nef": "NEF raw",
        "af_log": "camera AF box: {box} ({mode}) — green rectangle on the preview",
        "col_rank": "#",
        "col_cls": "Class",
        "col_score": "Rank",
        "col_p": "Prob",
        "col_token": "Token",
        "col_slot": "Slot",
        "col_box": "Box (384x256)",
    },
}


class App(tk.Tk):
    def __init__(self, lang: str = "zh"):
        super().__init__()
        self.lang = lang if lang in STRINGS else "zh"

        self.source: Path | None = None
        self.info: frames_module.SourceInfo | None = None
        self.model = None
        self.model_key = model_registry.default_key()
        self.events: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.running = False
        self._photo: ImageTk.PhotoImage | None = None
        self._latest_result = None

        # what the canvas composes from: raw frame + boxes + crop + selection
        self._raw = None
        self._mapped: list = []
        self._scored: list = []
        self._crop = None
        self._selected: int | None = None

        self._status_text = ""
        self._caption_text = ""
        self._log_lines: list = []

        # Settings live in Tk variables created on the main thread; the worker
        # never reads them (see _run), they are snapshotted before it starts.
        self.threads_var = tk.IntVar(value=max(1, min(4, os.cpu_count() or 1)))
        self.decoder_var = tk.StringVar(value="auto")
        self.per_second_var = tk.DoubleVar(value=1.0)
        self.max_frames_var = tk.IntVar(value=0)      # 0 = auto (min(60, fps))
        self.save_heads_var = tk.BooleanVar(value=False)
        self.save_frames_var = tk.BooleanVar(value=False)
        self.export_video_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._set_status(self._t("ready"))
        self.after(60, self._drain_events)

    # ------------------------------------------------------------------- i18n
    def _t(self, key: str, **values) -> str:
        text = STRINGS[self.lang].get(key, STRINGS["zh"].get(key, key))
        return text.format(**values) if values else text

    def set_language(self, lang: str):
        """Re-label the whole window.  State (source, boxes, log) is preserved."""
        target = "en" if lang.lower().startswith("en") else "zh"
        if target == self.lang:
            return
        self.lang = target
        self._rebuild_ui()

    # ------------------------------------------------------------------- UI
    def _build_ui(self):
        self.title(WINDOW_TITLE)
        self.geometry("1760x990")
        self.minsize(1280, 720)

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        heading = tkfont.nametofont("TkHeadingFont")
        bold_font = (heading.actual("family"), heading.actual("size"), "bold")
        style.configure("Heading.TLabel", font=bold_font)
        style.configure("Run.TButton", font=bold_font)

        controls = ttk.Frame(self, padding=(14, 12))
        controls.pack(side="left", fill="y")
        stage = ttk.Frame(self, padding=(6, 12, 14, 12))
        stage.pack(side="left", fill="both", expand=True)

        # language switch (top of the control column)
        lang_row = ttk.Frame(controls)
        lang_row.pack(fill="x", pady=(0, 8))
        ttk.Label(lang_row, text=self._t("language")).pack(side="left")
        self.lang_combo = ttk.Combobox(lang_row, state="readonly", width=8, values=["中文", "English"])
        self.lang_combo.set("中文" if self.lang == "zh" else "English")
        self.lang_combo.pack(side="right")
        self.lang_combo.bind("<<ComboboxSelected>>", self._language_selected)

        # 2 + 3: input selection and preview
        ttk.Label(controls, text=self._t("section_input"), style="Heading.TLabel").pack(anchor="w")
        row = ttk.Frame(controls)
        row.pack(fill="x", pady=(4, 2))
        ttk.Button(row, text=self._t("choose"), command=self._choose_input).pack(side="left")
        self.input_label = ttk.Label(controls, text=self._t("no_input"), wraplength=300, foreground="#666")
        self.input_label.pack(anchor="w", pady=(2, 8))
        if self.info is not None:
            self.input_label.config(text=self._input_text())

        # 4: model selection
        ttk.Label(controls, text=self._t("section_model"), style="Heading.TLabel").pack(anchor="w")
        self.model_combo = ttk.Combobox(
            controls, state="readonly",
            values=[label for _, label in model_registry.labels()],
        )
        self.model_combo.current(0)
        self.model_combo.bind("<<ComboboxSelected>>", self._model_changed)
        self.model_combo.pack(fill="x", pady=(4, 2))
        self.model_note = ttk.Label(controls, text=model_registry.find(self.model_key).note,
                                    wraplength=300, foreground="#666")
        self.model_note.pack(anchor="w", pady=(0, 4))
        # full model data: identity by hash first, then everything the verified
        # table knows about this body (blocks, input, head, provenance, flags)
        ttk.Label(controls, text=self._t("model_details"),
                  style="Heading.TLabel").pack(anchor="w")
        self.model_details = tk.Text(controls, height=18, width=44, wrap="word",
                                     state="disabled", relief="solid", bd=1,
                                     background="#fbfbfb", font=("TkFixedFont", 8))
        self.model_details.pack(fill="x", pady=(2, 8))
        self._show_model_details(self.model_key)

        # engine settings
        ttk.Label(controls, text=self._t("section_settings"), style="Heading.TLabel").pack(anchor="w")
        if self.threads_var is None:
            self.threads_var = tk.IntVar(value=max(1, min(4, os.cpu_count() or 1)))
            self.decoder_var = tk.StringVar(value="auto")
            self.per_second_var = tk.DoubleVar(value=1.0)
            self.max_frames_var = tk.IntVar(value=0)      # 0 = auto (min(60, fps))
            self.save_heads_var = tk.BooleanVar(value=False)
            self.save_frames_var = tk.BooleanVar(value=False)
            self.export_video_var = tk.BooleanVar(value=True)

        grid = ttk.Frame(controls)
        grid.pack(fill="x", pady=(4, 6))
        ttk.Label(grid, text=self._t("per_second")).grid(row=0, column=0, sticky="w")
        ttk.Spinbox(grid, from_=0.25, to=10.0, increment=0.25, width=7,
                    textvariable=self.per_second_var).grid(row=0, column=1, sticky="e")
        ttk.Label(grid, text=self._t("max_frames")).grid(row=1, column=0, sticky="w")
        ttk.Spinbox(grid, from_=0, to=100000, increment=5, width=7,
                    textvariable=self.max_frames_var).grid(row=1, column=1, sticky="e")
        ttk.Label(grid, text=self._t("threads")).grid(row=2, column=0, sticky="w")
        ttk.Spinbox(grid, from_=1, to=64, width=7, textvariable=self.threads_var).grid(row=2, column=1, sticky="e")
        ttk.Label(grid, text=self._t("decoder")).grid(row=3, column=0, sticky="w")
        ttk.Combobox(grid, values=["auto", "native", "port"], width=7, state="readonly",
                     textvariable=self.decoder_var).grid(row=3, column=1, sticky="e")

        for text, var in ((self._t("save_frames"), self.save_frames_var),
                          (self._t("save_heads"), self.save_heads_var),
                          (self._t("export_video"), self.export_video_var)):
            ttk.Checkbutton(controls, text=text, variable=var).pack(anchor="w")

        ttk.Label(controls, text=self._t("output_dir")).pack(anchor="w", pady=(8, 0))
        self.output_entry = ttk.Entry(controls, width=34)
        if self.source is not None:
            self.output_entry.insert(0, str(self.source.with_name(f"{self.source.stem}-e7-out")))
        elif self._output_dir_text:
            self.output_entry.insert(0, self._output_dir_text)
        self.output_entry.pack(fill="x", pady=(2, 8))

        # 5: execution
        ttk.Label(controls, text=self._t("section_run"), style="Heading.TLabel").pack(anchor="w")
        buttons = ttk.Frame(controls)
        buttons.pack(fill="x", pady=(4, 4))
        self.run_button = ttk.Button(buttons, text=self._t("run"), style="Run.TButton", command=self._run)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(buttons, text=self._t("cancel"), command=self._cancel,
                                        state="disabled" if not self.running else "normal")
        self.cancel_button.pack(side="left", padx=(6, 0))
        self.progress = ttk.Progressbar(controls, length=300, mode="determinate", maximum=1)
        self.progress.pack(fill="x", pady=(4, 2))
        self.status_label = ttk.Label(controls, text=self._status_text, wraplength=300)
        self.status_label.pack(anchor="w")

        self.log = scrolledtext.ScrolledText(controls, height=9, width=40, state="disabled",
                                             font=tkfont.nametofont("TkTextFont"))
        self.log.pack(fill="x", pady=(8, 0))
        if self._log_lines:
            self._replay_log()

        # 3 + 6 + box selection: canvas on the left, detection list on the right
        right = ttk.Frame(stage)
        right.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(right, background="#141414", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self._redraw())

        box_panel = ttk.Frame(right, padding=(10, 0, 0, 0))
        box_panel.pack(side="right", fill="y")
        ttk.Label(box_panel, text=self._t("section_boxes"), style="Heading.TLabel").pack(anchor="w")
        columns = ("rank", "cls", "score", "p", "token", "slot", "box")
        self.tree = ttk.Treeview(box_panel, columns=columns, show="headings", height=14, selectmode="browse")
        widths = {"rank": 30, "cls": 46, "score": 56, "p": 62, "token": 52, "slot": 40, "box": 116}
        for column in columns:
            # clicking a heading sorts the list by that column (again = reverse)
            self.tree.heading(column, text=self._heading_text(column),
                              command=lambda column=column: self._sort_by_heading(column))
            self.tree.column(column, width=widths[column], anchor="center" if column != "box" else "w")
        self.tree.pack(fill="both", expand=False, pady=(4, 4))
        self.tree.bind("<<TreeviewSelect>>", self._tree_selected)
        ttk.Button(box_panel, text=self._t("clear_selection"), command=self._clear_selection).pack(anchor="w")
        self.selection_label = ttk.Label(box_panel, text=self._t("no_boxes"), wraplength=250, foreground="#666")
        self.selection_label.pack(anchor="w", pady=(6, 0))

        self.stage_caption = ttk.Label(stage, text=self._caption_text or self._t("no_source"),
                                       wraplength=900, foreground="#666")
        self.stage_caption.pack(anchor="w", pady=(6, 0), side="bottom")

        self.bind("<Control-o>", lambda event: self._choose_input())
        self.bind("<Control-r>", lambda event: self._run())

        # widgets that hold values are created once and reused across language rebuilds
        self._redraw()

    # widget holders created lazily so `_rebuild_ui()` keeps user settings
    threads_var: tk.IntVar
    decoder_var: tk.StringVar
    per_second_var: tk.DoubleVar
    max_frames_var: tk.IntVar
    save_heads_var: tk.BooleanVar
    save_frames_var: tk.BooleanVar
    export_video_var: tk.BooleanVar
    output_entry = None
    _output_dir_text = ""
    _af_box: dict | None = None  # camera AF rectangle in display pixels (NEF sources only)
    _after_id = None
    _sort_column = None
    _sort_descending = False
    # columns whose first click should show the biggest values first
    _SORT_FIRST_DESCENDING = {"rank": True, "score": True, "p": True, "box": True}

    def _rebuild_ui(self):
        if self.output_entry is not None:
            try:
                self._output_dir_text = self.output_entry.get()
            except tk.TclError:
                pass
        for child in self.winfo_children():
            child.destroy()
        self._build_ui()
        self._fill_tree()   # the detection list is a fresh widget, so repopulate it
        self._redraw()

    def _language_selected(self, event=None):
        self.set_language(self.lang_combo.get())

    def _replay_log(self):
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        for line in self._log_lines:
            self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    # ----------------------------------------------------------------- input
    def _file_types(self):
        return [
            (self._t("file_all"), f"{IMAGE_GLOBS} {VIDEO_GLOBS} {RAW_GLOBS}"),
            (self._t("file_image"), IMAGE_GLOBS),
            (self._t("file_video"), VIDEO_GLOBS),
            (self._t("file_raw"), RAW_GLOBS),
            (self._t("file_every"), "*.*"),
        ]

    def _choose_input(self):
        path = filedialog.askopenfilename(title=self._t("pick_title"), filetypes=self._file_types())
        if not path:
            return
        self.set_source(Path(path))

    def _input_text(self) -> str:
        info = self.info
        size = f"{info.width}x{info.height}"
        if info.kind == "video":
            detail = f"{info.fps:.2f} fps · {info.duration:.2f} s · {info.total_frames or '?'} frames"
        elif info.kind == "nef":
            detail = info.note or self._t("kind_nef")
        else:
            detail = self._t("kind_image")
        kind_name = {"video": self._t("kind_video"), "nef": self._t("kind_nef")}.get(info.kind, info.kind)
        return f"{self.source.name}\n{kind_name} · {size} · {detail}"

    def set_source(self, path: Path):
        try:
            info = frames_module.probe(path)
        except Exception as error:  # unreadable / unsupported file
            self._set_status(self._t("unreadable", error=error))
            return
        self.source, self.info = path, info
        self._clear_boxes()
        self.input_label.config(text=self._input_text())
        self.output_entry.delete(0, "end")
        self.output_entry.insert(0, str(path.with_name(f"{path.stem}-e7-out")))
        self._refresh_sampling_hint()
        self._preview_first_frame()

    def _refresh_sampling_hint(self):
        if self.info is None or self.info.kind != "video":
            self._set_caption(self._t("preview_image"))
            return
        per_second = float(self.per_second_var.get() or 1.0)
        cap = self.max_frames_var.get() or None
        stride, planned = frames_module.sampling_plan(self.info, per_second, cap)
        self._set_caption(self._t("preview_video", per_second=per_second, stride=stride, planned=planned))

    def _preview_first_frame(self):
        if self.info is None:
            return
        try:
            stride, planned = frames_module.sampling_plan(
                self.info, float(self.per_second_var.get() or 1.0), self.max_frames_var.get() or None
            )
            for _, _, image in frames_module.iter_frames(self.source, self.info, stride, 1):
                if self.info.kind == "nef":
                    # Raw: the camera already framed 3:2 and fed the NN the
                    # NNIN strip — show its own AF box in green instead of a crop.
                    crop = pipeline.Crop(0, 0, image.width, image.height, "raw_nnin")
                    self._af_box = nef.af_display_box((self.info.extra or {})["meta"], image.size)
                else:
                    crop = pipeline.crop_to_target(image.width, image.height)
                self._set_frame(image, [], [], crop)
                self._set_status(pipeline.describe_crop(image.width, image.height, crop))
                break
        except Exception as error:
            self._set_status(self._t("preview_failed", error=error))

    def _model_changed(self, event=None):
        label = self.model_combo.get()
        self.model_key = next(key for key, name in model_registry.labels() if name == label)
        spec = model_registry.find(self.model_key)
        self.model_note.config(text=spec.note)
        self._show_model_details(self.model_key)
        self.model = None
        # Selecting a model loads the exact file it names, right now — the
        # file name shown in the list is the file that gets loaded (identity
        # comes from the content hash, never from the name).
        can_run = spec.runnable
        self.run_button.config(state="normal" if can_run else "disabled")
        if not spec.runnable:
            self._set_status(self._t("model_not_runnable", note=spec.note))
            return
        # Tk variables may only be read on the main thread: snapshot now.
        threads = int(self.threads_var.get())
        decoder_mode = self.decoder_var.get()
        threading.Thread(target=self._load_selected_model,
                         args=(spec, threads, decoder_mode),
                         name="model-load", daemon=True).start()

    def _show_model_details(self, key: str) -> None:
        """All known data about the selected model: identity by content hash,
        plus everything the verified body table says (blocks, input, head,
        provenance) and the file exactly as it sits on disk right now.  A
        role marked 'hypothesis' is never presented as fact."""
        if not hasattr(self, "model_details"):
            return
        spec = model_registry.find(key)
        d = spec.details

        def fmt(value):
            return format(value, ",") if isinstance(value, int) else (value or "—")

        lines = [
            f"{self._t('md_name')}: {spec.label.split(' — ')[0]}",
            f"{self._t('md_file')}: {fmt(d.get('filename'))}",
            f"{self._t('md_location')}: {fmt(d.get('found_at') or d.get('path'))}",
        ]
        if d.get("data_from"):
            lines.append(f"{self._t('md_data')}: {d['data_from']}")
        lines += [
            f"{self._t('md_role')}: {fmt(d.get('role'))}",
            f"{self._t('md_blocks')}: {fmt(d.get('blocks'))}",
            f"{self._t('md_input')}: {fmt(d.get('input'))}",
            f"{self._t('md_head')}: {fmt(d.get('head'))}",
            f"{self._t('md_bytes')}: {fmt(d.get('bytes'))}",
            f"{self._t('md_params')}: {fmt(d.get('fp16_words'))}",
            f"{self._t('md_sha')}: {fmt(d.get('sha256'))}",
            f"{self._t('md_runnable')}: {self._t('md_yes') if spec.runnable else self._t('md_no')}",
            f"{self._t('md_boxes')}: {self._t('md_yes') if spec.decodable else self._t('md_no')}",
            f"{self._t('md_source')}: {fmt(d.get('source'))}",
        ]
        self.model_details.config(state="normal")
        self.model_details.delete("1.0", "end")
        self.model_details.insert("1.0", "\n".join(lines))
        self.model_details.config(state="disabled")

    def _load_selected_model(self, spec, threads: int, decoder_mode: str) -> None:
        try:
            model = model_registry.build(spec.key, threads=threads,
                                        decoder_mode=decoder_mode)
            model.load(log=lambda message: self.events.put(("log", message)))
        except Exception:
            self.model = None
            self.events.put(("model_load_failed", traceback.format_exc()))
            return
        self.model = model
        self.events.put(("model_loaded", spec.label))

    # ----------------------------------------------------------------- stage
    def _set_frame(self, raw, mapped, scored, crop):
        """Remember the frame + its boxes, then repaint the canvas."""
        self._raw = raw
        self._mapped = list(mapped)
        self._scored = list(scored)
        self._crop = crop
        # A selection the user made sticks across frames; nothing is auto-selected,
        # so a fresh frame shows all boxes until one is picked.
        if self._selected is not None and self._selected >= len(mapped):
            self._selected = None
        self._fill_tree()
        self._redraw()

    def _clear_boxes(self):
        self._mapped, self._scored = [], []
        self._selected = None
        self._af_box = None
        if hasattr(self, "tree"):
            self.tree.delete(*self.tree.get_children())
            self.selection_label.config(text=self._t("no_boxes"))

    def _heading_text(self, column: str) -> str:
        """Column heading, with a ▲/▼ marker while that column drives the order."""
        base = self._t(f"col_{column}")
        if self._sort_column != column:
            return base
        return base + (" ▼" if self._sort_descending else " ▲")

    _COLUMN_INDEX = {"rank": 0, "cls": 1, "score": 2, "p": 3, "token": 4, "slot": 5, "box": 6}

    def _sort_value(self, column: str, values):
        """Numeric sort key for one row; rows with '-' (no score data) sort last.

        The `box` column sorts by box area, which is what matters when several
        overlapping boxes compete for the same object.
        """
        raw = values[self._COLUMN_INDEX[column]]
        if column == "box":
            try:
                width, height = str(raw).split()[-1].split("x")
                return int(width) * int(height)
            except ValueError:
                return 0
        if column == "p":
            try:
                return float(str(raw).rstrip("%"))     # shown as 0–100 %
            except ValueError:
                return -1.0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return -1

    def _sort_by_heading(self, column: str):
        """Heading click: sort by that column; clicking it again reverses."""
        if self._sort_column == column:
            self._sort_descending = not self._sort_descending
        else:
            self._sort_column = column
            self._sort_descending = bool(self._SORT_FIRST_DESCENDING.get(column, False))
        self._apply_sort()
        for other in self.tree["columns"]:
            self.tree.heading(other, text=self._heading_text(other))

    def _apply_sort(self):
        """Reorder the existing rows; item ids stay tied to the box index, so the
        selection keeps pointing at the same detection after sorting."""
        column = self._sort_column
        if column is None or not hasattr(self, "tree"):
            return
        items = list(self.tree.get_children())
        items.sort(key=lambda iid: self._sort_value(column, self.tree.item(iid)["values"]),
                   reverse=self._sort_descending)
        for position, iid in enumerate(items):
            self.tree.move(iid, "", position)

    def _fill_tree(self):
        if not hasattr(self, "tree"):
            return
        self.tree.delete(*self.tree.get_children())
        if self._af_box:
            # The camera's own chosen AF box, as its own row: green source, not
            # a model detection, so score / p / token / slot stay empty.
            self.tree.insert(
                "", 0, iid="af",
                values=("AF", "CAM", "-", "-", "-", "-",
                        f"{int(self._af_box['x'])},{int(self._af_box['y'])} "
                        f"{int(self._af_box['w'])}x{int(self._af_box['h'])}"),
            )
        for index, (mapped, scored) in enumerate(zip(self._mapped, self._scored + [{}] * (len(self._mapped) - len(self._scored)))):
            x, y, w, h, score, cls = mapped
            combined = scored.get("combined")
            self.tree.insert(
                "", "end", iid=str(index),
                values=(
                    index, int(cls), int(score),
                    f"{100.0 * float(combined):.1f}%" if combined is not None else "-",
                    scored.get("token") if scored.get("token") is not None else "-",
                    scored.get("slot") if scored.get("slot") is not None else "-",
                    f"{int(x)},{int(y)} {int(w)}x{int(h)}",
                ),
            )
        self._apply_sort()      # keep the user's chosen order across refreshes
        if self._selected is not None:
            self.tree.selection_set(str(self._selected))
            self.selection_label.config(text=self._t("box_selected", index=self._selected))
        else:
            self.selection_label.config(
                text=self._t("box_none") if self._mapped else self._t("no_boxes")
            )

    def _tree_selected(self, event=None):
        selection = self.tree.selection()
        if not selection:
            return
        try:
            self._selected = int(selection[0])
        except ValueError:
            return  # the AF row is the camera's own box: display only, not selectable
        self.selection_label.config(text=self._t("box_selected", index=self._selected))
        self._redraw()

    def _clear_selection(self):
        self._selected = None
        if hasattr(self, "tree"):
            self.tree.selection_set(())
            self.selection_label.config(
                text=self._t("box_none") if self._mapped else self._t("no_boxes")
            )
        self._redraw()

    def _compose(self):
        """Full-res frame with overlays: selected box at 100%, everything else faded."""
        if self._raw is None:
            return None
        crop = self._crop
        scale = max(1.0, max(self._raw.width, self._raw.height) / 900.0)
        if self._selected is None or not self._mapped:
            return draw.draw_overlay(self._raw, self._mapped, crop, show_crop=True, show_labels=True,
                                     scale_for_display=scale, scored=self._scored, af_box=self._af_box)
        box = self._mapped[self._selected]
        faded = draw.dim_outside(self._raw, box)
        return draw.draw_overlay(faded, self._mapped, crop, show_crop=True, show_labels=True,
                                 scale_for_display=scale, highlight=self._selected, scored=self._scored,
                                 af_box=self._af_box)

    def _redraw(self):
        if self._raw is None:
            self.canvas.delete("all")
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        image = self._compose()
        scale = min(width / image.width, height / image.height)
        target = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        rendered = image.resize(target, Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(rendered)
        self.canvas.delete("all")
        self.canvas.create_image((width - target[0]) // 2, (height - target[1]) // 2,
                                 image=self._photo, anchor="nw")

    def _set_status(self, text: str):
        self._status_text = text
        if hasattr(self, "status_label"):
            self.status_label.config(text=text)

    def _set_caption(self, text: str):
        self._caption_text = text
        if hasattr(self, "stage_caption"):
            self.stage_caption.config(text=text)

    def _log(self, line: str):
        self._log_lines.append(line.rstrip())
        if hasattr(self, "log"):
            self.log.config(state="normal")
            self.log.insert("end", line.rstrip() + "\n")
            self.log.see("end")
            self.log.config(state="disabled")

    # ------------------------------------------------------------------ run
    def _run(self):
        if self.running:
            return
        if self.source is None:
            self._set_status(self._t("select_input_first"))
            return
        output = Path(self.output_entry.get().strip() or f"{self.source.stem}-e7-out")
        self.cancel.clear()
        self.running = True
        self.run_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.progress.config(maximum=1.0, value=0.0)
        self._set_status(self._t("running"))
        self._log(self._t("run_header", name=self.source.name, output=output))

        spec = model_registry.find(self.model_key)
        model = self.model or model_registry.build(
            spec.key, threads=int(self.threads_var.get()), decoder_mode=self.decoder_var.get()
        )
        self.model = model
        # Tk variables may only be read on the main thread (CPython's _tkinter
        # raises if another thread touches the interpreter), so snapshot every
        # setting here and hand plain values to the worker.
        settings = {
            "per_second": float(self.per_second_var.get() or 1.0),
            "max_frames": self.max_frames_var.get() or None,
            "save_heads": bool(self.save_heads_var.get()),
            "save_frames": bool(self.save_frames_var.get()),
            "export_video": bool(self.export_video_var.get()),
        }
        self.worker = threading.Thread(
            target=self._worker, args=(model, output, settings), name="e7-job", daemon=True
        )
        self.worker.start()

    def _worker(self, model, output: Path, settings: dict):
        try:
            result = job.run_job(
                model,
                self.source,
                output,
                per_second=settings["per_second"],
                max_frames=settings["max_frames"],
                save_heads=settings["save_heads"],
                save_frames=settings["save_frames"],
                export_video=settings["export_video"],
                progress=lambda event: self.events.put(("event", event)),
                should_cancel=self.cancel.is_set,
                log=lambda message: self.events.put(("log", message)),
            )
            self.events.put(("result", result))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def _cancel(self):
        self.cancel.set()
        self._log(self._t("cancelled"))

    def _drain_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "event":
                    self._handle_event(payload)
                elif kind == "result":
                    self._finish(payload, None)
                elif kind == "error":
                    self._log(payload)
                    self._finish(None, payload.strip().splitlines()[-1])
                elif kind == "model_load_failed":
                    self._set_status(self._t("model_load_failed",
                                             error=payload.strip().splitlines()[-1]))
                elif kind == "model_loaded":
                    self._set_status(self._t("model_loaded", name=payload))
        except queue.Empty:
            pass
        try:
            self._after_id = self.after(60, self._drain_events)
        except tk.TclError:      # window already torn down
            self._after_id = None

    def destroy(self):
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        super().destroy()

    def _handle_event(self, event):
        kind = event.get("kind")
        if kind == "plan":
            total = event.get("planned") or 1
            self.progress.config(maximum=float(total), value=0.0)
            truncated = self._t("truncated", planned=event.get("planned")) if event.get("truncated") else ""
            self._log(self._t("sampling", stride=event.get("stride"), planned=event.get("planned"),
                              truncated=truncated))
            af = event.get("af_box")
            if af:
                self._af_box = af
                self._log(self._t("af_log",
                                  box=f"{int(af['x'])},{int(af['y'])} {int(af['w'])}x{int(af['h'])}",
                                  mode=af.get("mode_name") or "?"))
                self._fill_tree()
        elif kind == "frame":
            if event.get("af_box"):
                self._af_box = event["af_box"]
            crop = event.get("crop")
            crop_object = pipeline.Crop(**crop) if crop else pipeline.crop_to_target(
                event["raw"].width, event["raw"].height
            )
            self._set_frame(event.get("raw") or event.get("annotated"), event.get("mapped") or [],
                            event.get("scored") or [], crop_object)
            self.progress.config(value=float(event.get("done") or 0))
            self._set_status(self._t(
                "frame_status",
                frame_index=event.get("frame_index"),
                seconds=f"{event.get('seconds', 0):.2f}",
                done=event.get("done"),
                total=int(self.progress["maximum"]),
                frame_seconds=f"{event.get('frame_seconds', 0):.1f}",
                detections=event.get("detections"),
                crop_mode=event.get("crop_mode"),
            ))
        elif kind == "done":
            self._log(self._t("done_log", processed=event.get("processed"),
                              detections_total=event.get("detections_total"),
                              total_seconds=f"{event.get('total_seconds', 0):.1f}"))

    def _finish(self, result, error):
        self.running = False
        spec = model_registry.find(self.model_key)
        self.run_button.config(state="normal" if spec.runnable else "disabled")
        self.cancel_button.config(state="disabled")
        if error:
            self._set_status(self._t("failed", error=error))
            return
        if result is None:
            return
        self._latest_result = result
        summary = self._t(
            "finish_summary",
            processed=result.processed_frames,
            planned=result.planned_frames,
            detections=result.detections_total,
            per_frame=f"{result.per_frame_seconds:.1f}",
            output=result.output,
        )
        if result.video_export:
            summary += self._t("finish_video", path=result.video_export)
        if result.truncated:
            summary += self._t("finish_cap")
        self._set_status(summary)
        self._log(self._t("exports", names="; ".join(Path(p).name for p in result.exports)))


def selftest(path: str | None):
    app = App()
    app.update_idletasks()
    app.update()
    assert app.title() == WINDOW_TITLE, app.title()

    # model list: every row carries the file name it loads, and selecting a
    # model auto-loads that exact file (identity by content hash, name is
    # decoration); networks without a decode contract stay listed but disabled.
    pairs = model_registry.labels()
    assert any("NET5_body.bin" in label for _, label in pairs), pairs
    for key, label in pairs:
        if key != model_registry.default_key() and not key.startswith("unknown:"):
            file_part = label.split(" — ", 1)
            assert len(file_part) == 2 and file_part[1].endswith(".bin"), \
                f"model rows must show the file name: {label}"
    if any(k == "z8-net7" for k, _ in pairs):
        idx = [k for k, _ in pairs].index("z8-net7")
        app.model_combo.current(idx)
        app._model_changed()
        assert str(app.run_button["state"]) == "normal", \
            "no decode contract → inference only, but run must be enabled"
        deadline = time.time() + 120
        while time.time() < deadline and app.model is None:
            app.update()
            time.sleep(0.05)
        assert app.model is not None and app.model.key == "z8-net7", \
            f"selecting NET7 must auto-load its file: status={app._status_text!r}"
        print(f"selftest: NET7 auto-loaded from {app.model.path.name}")
        # the model-data panel must show everything known about this body:
        # hash identity, file, geometry, head, flags — hypothesis roles only as hypotheses
        detail_text = app.model_details.get("1.0", "end")
        for needle in ("NET7", "NET7_body.bin", "66a34f24", "889", "384x256x3"):
            assert needle in detail_text, f"model data panel missing {needle!r}: {detail_text}"
        # a body the emulator cannot fully parse must stay run-disabled
        if any(k == "z8-net1" for k, _ in pairs):
            idx = [k for k, _ in pairs].index("z8-net1")
            app.model_combo.current(idx)
            app._model_changed()
            assert str(app.run_button["state"]) == "disabled", \
                "unparseable body → run must stay disabled"
        app.model_combo.current(0)
        app._model_changed()
        assert str(app.run_button["state"]) == "normal"
        detail_text = app.model_details.get("1.0", "end")
        for needle in ("NET5 (BIRD)", "NET5_body.bin", "e8d371e7",
                       "BIRD_body.bin", "engine-verified"):
            assert needle in detail_text, f"model data panel missing {needle!r}: {detail_text}"
        app.model = None

    if path:
        app.set_source(Path(path))
        app.update()
        print(f"selftest: source={app.source} raw={app._raw.size if app._raw else None}")

    # synthetic boxes exercise the selection + fade path without running inference
    if app._raw is not None:
        crop = app._crop
        width, height = app._raw.width, app._raw.height
        boxes = [
            (width * 0.30, height * 0.40, width * 0.25, height * 0.25, 974, 0),
            (width * 0.55, height * 0.45, width * 0.20, height * 0.30, 991, 1),
        ]
        scored = [
            {"token": 1, "slot": 3, "combined": 0.874, "objectness": 0.91},
            {"token": 18, "slot": 9, "combined": 0.612, "objectness": 0.78},
        ]
        app._set_frame(app._raw, boxes, scored, crop)
        app.update()
        rows = [c for c in app.tree.get_children() if str(c).isdigit()]
        assert len(rows) == 2, app.tree.get_children()
        # probabilities display as 0–100 %, not 0.000–1.000
        assert app.tree.item("0")["values"][3] == "87.4%", app.tree.item("0")["values"]
        assert app.tree.item("1")["values"][3] == "61.2%", app.tree.item("1")["values"]
        assert ("af" in app.tree.get_children()) == (app._af_box is not None), \
            f"AF row must appear exactly for NEF sources: {app.tree.get_children()}"
        if app.info.kind == "nef":
            # camera AF box: own list row + green rectangle on the canvas
            assert app._af_box is not None, "NEF source must surface the camera AF box"
            print(f"selftest: AF row {app.tree.item('af')['values']}")
            composed = app._compose().convert("RGB")
            af_x = int(app._af_box["x"])
            af_y = int(app._af_box["y"])
            af_h = int(app._af_box["h"])
            greens = []
            for frac in (0.6, 0.75, 0.9):   # label chips may cover the upper edge
                my = af_y + int(af_h * frac)
                greens.append(any(abs(r - 46) < 50 and abs(g - 204) < 50 and abs(b - 113) < 50
                                  for r, g, b in (composed.getpixel((x, my))
                                                  for x in range(max(0, af_x - 40), af_x + 40))))
            assert any(greens), f"camera AF box must draw green along x={af_x}: {greens}"
            app._handle_event({"kind": "plan", "planned": 1, "stride": 1, "af_box": app._af_box})
            assert any("AF 框" in l or "AF box" in l for l in app._log_lines), app._log_lines[-3:]
        app._selected = 1
        app._redraw()
        app.update()
        composed = app._compose().convert("RGB")
        base = app._raw.convert("RGB")
        box = boxes[1]
        scale = max(1.0, max(base.width, base.height) / 900.0)
        stroke = int(round(5.0 * scale)) + 10   # widest deliberate stroke is the
                                                # highlighted box (box_width * 1.6)
        def near_line(px, py, rect, pad):
            """True when the pixel sits within `pad` of the rectangle's outline."""
            x, y, w, h = rect
            vertical = (abs(px - x) <= pad or abs(px - (x + w)) <= pad) and (y - pad <= py <= y + h + pad)
            horizontal = (abs(py - y) <= pad or abs(py - (y + h)) <= pad) and (x - pad <= px <= x + w + pad)
            return vertical or horizontal
        crop_rect = (crop.x, crop.y, crop.width, crop.height)
        af_rect = (app._af_box["x"], app._af_box["y"], app._af_box["w"], app._af_box["h"]) \
            if app._af_box else None
        inside_left = int(box[0] - box[2] / 2) + stroke
        inside_right = int(box[0] + box[2] / 2) - stroke
        inside_top = int(box[1] - box[3] / 2) + stroke   # below the box outline
        label_band = stroke * 6                          # labels sit inside the top edge
        inside_bottom = int(box[1] + box[3] / 2) - stroke
        outside_same = 0
        outside_total = 0
        for x in range(0, base.width, max(1, base.width // 40)):
            for y in range(0, base.height, max(1, base.height // 20)):
                if inside_left <= x <= inside_right and inside_top <= y <= inside_bottom:
                    continue
                if near_line(x, y, crop_rect, stroke):
                    continue                             # crop overlay stroke
                if af_rect and near_line(x, y, af_rect, stroke):
                    continue                             # camera AF box stroke
                outside_total += 1
                if composed.getpixel((x, y)) == base.getpixel((x, y)):
                    outside_same += 1
        inside_same = 0
        inside_total = 0
        for x in range(inside_left, inside_right, max(1, (inside_right - inside_left) // 20)):
            for y in range(inside_top + label_band, inside_bottom,
                           max(1, (inside_bottom - inside_top) // 12)):
                if near_line(x, y, crop_rect, stroke):
                    continue                             # crop overlay crosses the box
                if af_rect and near_line(x, y, af_rect, stroke):
                    continue
                if af_rect:
                    font_px = max(12, int(round(14 * scale)))  # label = f"機身AF · {mode}"
                    if af_rect[0] <= x <= af_rect[0] + 14 * font_px and af_rect[1] <= y <= af_rect[1] + 2 * font_px:
                        continue                               # green AF label text inside the box
                inside_total += 1
                if composed.getpixel((x, y)) == base.getpixel((x, y)):
                    inside_same += 1
        assert outside_total and outside_same <= outside_total * 0.1, \
            f"outside the selected box should be faded ({outside_same}/{outside_total} unchanged)"
        assert inside_total and inside_same == inside_total, \
            f"inside the selected box must stay at 100% ({inside_same}/{inside_total})"
        app._clear_selection()
        app.update()

    app.set_language("en")
    app.update()
    assert app.title() == WINDOW_TITLE
    assert app._t("run") == "Run"
    app.set_language("zh")
    app.update()
    assert app._t("run") == "執行"
    app.destroy()
    print("GUI SELFTEST OK")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--selftest":
        selftest(argv[1] if len(argv) > 1 else None)
        return 0
    app = App()
    if argv:
        app.set_source(Path(argv[0]))
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
