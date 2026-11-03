"""E7 bird detection GUI — Tkinter, so it runs the same way on Windows and Linux.

    python app/gui.py                 (Windows:  py -3 app\\gui.py)

Six requirements from the brief:
  1. GUI 介面                    — this window (Tkinter, stdlib)
  2. 選擇 input photo or video   — 「選擇輸入」 button, photo/video file filter
  3. 顯示選擇了的 input          — right canvas shows the image / sampled video frames
  4. 選擇 Model                  — dropdown fed by `model_registry` (E7 bird model today)
  5. 執行                        — 執行 / 取消 buttons, worker thread, progress bar, log
  6. 顯示 model output + 目標框  — the canvas redraws each frame with detection boxes and
                                   the dashed 3:2 crop rectangle drawn on the original frame

Input that is not 3:2 is centre-cropped to 3:2 first (trimming height where
possible, width otherwise) and then resized to 384x256 — see `pipeline.py`.
"""

import os
import queue
import sys
import threading
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

import frames as frames_module
import job
import model_registry
import pipeline

FILE_TYPES = [
    ("圖片及影片", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.gif *.mp4 *.mov *.avi *.mkv *.webm *.m4v *.wmv *.mpg *.mpeg *.3gp"),
    ("圖片", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.gif"),
    ("影片", "*.mp4 *.mov *.avi *.mkv *.webm *.m4v *.wmv *.mpg *.mpeg *.3gp"),
    ("所有檔案", "*.*"),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("E7 Bird Detection — 鳥偵測 GUI")
        self.geometry("1180x780")
        self.minsize(920, 620)

        self.source: Path | None = None
        self.info: frames_module.SourceInfo | None = None
        self.model = None
        self.model_key = model_registry.default_key()
        self.events: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        self.running = False
        self._base_image = None          # PIL image currently shown (frame or annotated frame)
        self._photo: ImageTk.PhotoImage | None = None
        self._latest_result = None

        self._build_ui()
        self._set_status("準備好 / ready — 選擇一個圖片或影片檔案")
        self.after(60, self._drain_events)

    # ------------------------------------------------------------------- UI
    def _build_ui(self):
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

        # 2 + 3: input selection and preview
        ttk.Label(controls, text="1 · 輸入", style="Heading.TLabel").pack(anchor="w")
        row = ttk.Frame(controls)
        row.pack(fill="x", pady=(4, 2))
        ttk.Button(row, text="選擇輸入 …", command=self._choose_input).pack(side="left")
        self.input_label = ttk.Label(controls, text="(未選擇)", wraplength=300, foreground="#666")
        self.input_label.pack(anchor="w", pady=(2, 8))

        # 4: model selection
        ttk.Label(controls, text="2 · Model", style="Heading.TLabel").pack(anchor="w")
        self.model_combo = ttk.Combobox(
            controls, state="readonly",
            values=[label for _, label in model_registry.labels()],
        )
        self.model_combo.current(0)
        self.model_combo.bind("<<ComboboxSelected>>", self._model_changed)
        self.model_combo.pack(fill="x", pady=(4, 2))
        self.model_note = ttk.Label(controls, text=model_registry.find(self.model_key).note,
                                    wraplength=300, foreground="#666")
        self.model_note.pack(anchor="w", pady=(0, 8))

        # engine settings
        ttk.Label(controls, text="3 · 設定", style="Heading.TLabel").pack(anchor="w")
        self.threads_var = tk.IntVar(value=max(1, min(4, os.cpu_count() or 1)))
        self.decoder_var = tk.StringVar(value="auto")
        self.per_second_var = tk.DoubleVar(value=1.0)
        self.max_frames_var = tk.IntVar(value=0)          # 0 = auto (min(60, fps))
        self.save_heads_var = tk.BooleanVar(value=False)
        self.save_frames_var = tk.BooleanVar(value=False)
        self.export_video_var = tk.BooleanVar(value=True)

        grid = ttk.Frame(controls)
        grid.pack(fill="x", pady=(4, 6))
        ttk.Label(grid, text="取樣/秒").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(grid, from_=0.25, to=10.0, increment=0.25, width=7,
                    textvariable=self.per_second_var).grid(row=0, column=1, sticky="e")
        ttk.Label(grid, text="最大幀數 (0=auto)").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(grid, from_=0, to=100000, increment=5, width=7,
                    textvariable=self.max_frames_var).grid(row=1, column=1, sticky="e")
        ttk.Label(grid, text="執行緒").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(grid, from_=1, to=64, width=7, textvariable=self.threads_var).grid(row=2, column=1, sticky="e")
        ttk.Label(grid, text="Decoder").grid(row=3, column=0, sticky="w")
        ttk.Combobox(grid, values=["auto", "native", "port"], width=7, state="readonly",
                     textvariable=self.decoder_var).grid(row=3, column=1, sticky="e")

        for text, var in (("儲存每幀 PNG", self.save_frames_var),
                          ("儲存 heads (.npz)", self.save_heads_var),
                          ("輸出標註影片 .mp4", self.export_video_var)):
            ttk.Checkbutton(controls, text=text, variable=var).pack(anchor="w")

        ttk.Label(controls, text="輸出目錄").pack(anchor="w", pady=(8, 0))
        self.output_entry = ttk.Entry(controls, width=34)
        self.output_entry.pack(fill="x", pady=(2, 8))

        # 5: execution
        ttk.Label(controls, text="4 · 執行", style="Heading.TLabel").pack(anchor="w")
        buttons = ttk.Frame(controls)
        buttons.pack(fill="x", pady=(4, 4))
        self.run_button = ttk.Button(buttons, text="執行", style="Run.TButton", command=self._run)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(buttons, text="取消", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=(6, 0))
        self.progress = ttk.Progressbar(controls, length=300, mode="determinate", maximum=1)
        self.progress.pack(fill="x", pady=(4, 2))
        self.status_label = ttk.Label(controls, text="", wraplength=300)
        self.status_label.pack(anchor="w")

        self.log = scrolledtext.ScrolledText(controls, height=9, width=40, state="disabled",
                                            font=tkfont.nametofont("TkTextFont"))
        self.log.pack(fill="x", pady=(8, 0))

        # 3 + 6: the stage shows the selected input, then the model output on top of it
        self.canvas = tk.Canvas(stage, background="#141414", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self._redraw())
        self.stage_caption = ttk.Label(stage, text="(未有輸入)", wraplength=820, foreground="#666")
        self.stage_caption.pack(anchor="w", pady=(6, 0))

        self.bind("<Control-o>", lambda event: self._choose_input())
        self.bind("<Control-r>", lambda event: self._run())

    # ----------------------------------------------------------------- input
    def _choose_input(self):
        path = filedialog.askopenfilename(title="選擇圖片或影片", filetypes=FILE_TYPES)
        if not path:
            return
        self.set_source(Path(path))

    def set_source(self, path: Path):
        try:
            info = frames_module.probe(path)
        except Exception as error:  # unreadable / unsupported file
            self._set_status(f"無法讀取輸入：{error}")
            return
        self.source, self.info = path, info
        size = f"{info.width}x{info.height}"
        if info.kind == "video":
            detail = f"{info.fps:.2f} fps · {info.duration:.2f} s · 約 {info.total_frames or '?'} 幀"
        else:
            detail = "靜止圖片"
        self.input_label.config(text=f"{path.name}\n{info.kind} · {size} · {detail}")
        self.output_entry.delete(0, "end")
        self.output_entry.insert(0, str(path.with_name(f"{path.stem}-e7-out")))
        self._refresh_sampling_hint()
        self._preview_first_frame()

    def _refresh_sampling_hint(self):
        if self.info is None or self.info.kind != "video":
            self.stage_caption.config(text="預覽：選了的圖片")
            return
        per_second = float(self.per_second_var.get() or 1.0)
        cap = self.max_frames_var.get() or None
        stride, planned = frames_module.sampling_plan(self.info, per_second, cap)
        self.stage_caption.config(
            text=(f"預覽：影片第 0 幀。取樣計劃：每秒取 {per_second} 幀（stride {stride}），"
                  f"最多處理 {planned} 幀（cap = min(60, fps)）")
        )

    def _preview_first_frame(self):
        if self.info is None:
            return
        try:
            stride, planned = frames_module.sampling_plan(
                self.info, float(self.per_second_var.get() or 1.0), self.max_frames_var.get() or None
            )
            for _, _, image in frames_module.iter_frames(self.source, self.info, stride, 1):
                self._show(image)
                crop = pipeline.crop_to_target(image.width, image.height)
                self._set_status(pipeline.describe_crop(image.width, image.height, crop))
                break
        except Exception as error:
            self._set_status(f"預覽失敗：{error}")

    def _model_changed(self, event=None):
        label = self.model_combo.get()
        self.model_key = next(key for key, name in model_registry.labels() if name == label)
        self.model = None  # rebuilt lazily on 執行
        self.model_note.config(text=model_registry.find(self.model_key).note)

    # ----------------------------------------------------------------- stage
    def _show(self, image):
        self._base_image = image
        self._redraw()

    def _redraw(self):
        if self._base_image is None:
            self.canvas.delete("all")
            return
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        scale = min(width / self._base_image.width, height / self._base_image.height)
        target = (max(1, int(self._base_image.width * scale)), max(1, int(self._base_image.height * scale)))
        rendered = self._base_image.resize(target, Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(rendered)
        self.canvas.delete("all")
        self.canvas.create_image((width - target[0]) // 2, (height - target[1]) // 2,
                                 image=self._photo, anchor="nw")

    def _set_status(self, text: str):
        self.status_label.config(text=text)

    def _log(self, line: str):
        self.log.config(state="normal")
        self.log.insert("end", line.rstrip() + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    # ------------------------------------------------------------------ run
    def _run(self):
        if self.running:
            return
        if self.source is None:
            self._set_status("請先選擇輸入檔案")
            return
        output = Path(self.output_entry.get().strip() or f"{self.source.stem}-e7-out")
        self.cancel.clear()
        self.running = True
        self.run_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.progress.config(maximum=1.0, value=0.0)
        self._set_status("執行中 …")
        self._log(f"=== 執行 {self.source.name} → {output} ===")

        spec = model_registry.find(self.model_key)
        model = self.model or model_registry.build(
            spec.key, threads=int(self.threads_var.get()), decoder_mode=self.decoder_var.get()
        )
        self.model = model
        self.worker = threading.Thread(
            target=self._worker, args=(model, output), name="e7-job", daemon=True
        )
        self.worker.start()

    def _worker(self, model, output: Path):
        try:
            result = job.run_job(
                model,
                self.source,
                output,
                per_second=float(self.per_second_var.get() or 1.0),
                max_frames=(self.max_frames_var.get() or None),
                save_heads=bool(self.save_heads_var.get()),
                save_frames=bool(self.save_frames_var.get()),
                export_video=bool(self.export_video_var.get()),
                progress=lambda event: self.events.put(("event", event)),
                should_cancel=self.cancel.is_set,
                log=lambda message: self.events.put(("log", message)),
            )
            self.events.put(("result", result))
        except Exception:
            self.events.put(("error", traceback.format_exc()))

    def _cancel(self):
        self.cancel.set()
        self._log("要求取消 …")

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
        except queue.Empty:
            pass
        self.after(60, self._drain_events)

    def _handle_event(self, event):
        kind = event.get("kind")
        if kind == "plan":
            total = event.get("planned") or 1
            self.progress.config(maximum=float(total), value=0.0)
            truncated = "（已按 cap 截斷，只處理首 " + str(event.get("planned")) + " 幀）" if event.get("truncated") else ""
            self._log(f"取樣：stride {event.get('stride')} 幀，處理 {event.get('planned')} 幀{truncated}")
        elif kind == "frame":
            annotated = event.get("annotated")
            if annotated is not None:
                self._show(annotated)
            self.progress.config(value=float(event.get("done") or 0))
            self._set_status(
                f"幀 {event.get('frame_index')} · t={event.get('seconds'):.2f}s · "
                f"{event.get('done')}/{int(self.progress['maximum'])} · "
                f"推論 {event.get('frame_seconds'):.1f}s · 偵測 {event.get('detections')} 框 · "
                f"crop {event.get('crop_mode')}"
            )
        elif kind == "done":
            self._log(f"完成：{event.get('processed')} 幀，{event.get('detections_total')} 個偵測框，"
                      f"{event.get('total_seconds'):.1f} 秒")

    def _finish(self, result, error):
        self.running = False
        self.run_button.config(state="normal")
        self.cancel_button.config(state="disabled")
        if error:
            self._set_status(f"失敗：{error}")
            return
        if result is None:
            return
        self._latest_result = result
        summary = (f"{result.processed_frames}/{result.planned_frames} 幀 · {result.detections_total} 框 · "
                   f"每幀 {result.per_frame_seconds:.1f}s · 輸出 {result.output}")
        if result.video_export:
            summary += f"\n標註影片：{result.video_export}"
        if result.truncated:
            summary += "\n（cap = min(60, fps)，可用「最大幀數」放大）"
        self._set_status(summary)
        self._log("輸出：" + "; ".join(Path(p).name for p in result.exports))


def selftest(path: str | None):
    app = App()
    app.update_idletasks()
    app.update()
    if path:
        app.set_source(Path(path))
        app.update()
        print(f"selftest: source={app.source} base={app._base_image.size if app._base_image else None}")
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
