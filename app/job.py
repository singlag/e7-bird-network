"""Shared job runner used by both the GUI worker thread and `cli.py`.

One job = probe the input -> sample frames -> crop to 3:2 -> run the network ->
decode candidate boxes -> draw overlays on the original frames -> write exports.
"""

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import draw
import frames as frames_module
import pipeline


@dataclass
class JobResult:
    kind: str
    source: str
    output: str
    crop_mode: str
    crop: dict
    stride: int
    planned_frames: int
    processed_frames: int
    truncated: bool
    per_frame_seconds: float
    total_seconds: float
    detections_total: int
    video_export: Optional[str]
    exports: list
    frames: list
    model_status: dict


def emit(progress: Optional[Callable[[dict], None]], **event) -> None:
    if progress is not None:
        progress(event)


def run_job(
    model,
    source_path: Path,
    output_dir: Path,
    per_second: float = 1.0,
    max_frames: Optional[int] = None,
    save_heads: bool = False,
    save_frames: bool = False,
    export_video: bool = True,
    video_fps: Optional[float] = None,
    progress: Optional[Callable[[dict], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    log: Callable[[str], None] = lambda message: None,
) -> JobResult:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = frames_module.probe(source_path)
    stride, planned = frames_module.sampling_plan(info, per_second, max_frames)
    available = max(1, math.ceil((info.duration or 0.0) / per_second)) if info.kind == "video" else 1
    truncated = info.kind == "video" and planned < available
    emit(
        progress,
        kind="plan",
        source=str(source_path),
        source_kind=info.kind,
        width=info.width,
        height=info.height,
        fps=info.fps,
        duration=info.duration,
        stride=stride,
        planned=planned,
        available=available,
        truncated=truncated,
    )

    model.load(log=log)
    status = model.status()

    exports: list = []
    per_frame: list = []
    annotated_files: list = []
    detections_total = 0
    total_infer = 0.0
    processed = 0
    started = time.monotonic()

    for index, seconds, image in frames_module.iter_frames(source_path, info, stride, planned):
        if should_cancel is not None and should_cancel():
            log("已取消 / cancelled")
            break
        crop = pipeline.crop_to_target(image.width, image.height)
        encoded = pipeline.encode(image, crop)
        inference = model.infer(encoded, include_workspace=save_heads)
        report = model.decode(inference["head_buffer"])
        mapped = draw.map_records(report["records"], crop)
        display_scale = max(1.0, max(image.width, image.height) / 900.0)
        annotated = draw.draw_overlay(image, mapped, crop, show_crop=True, show_labels=True, scale_for_display=display_scale)
        total_infer += inference["seconds"]
        processed += 1
        detections_total += len(mapped)

        relative = f"frame-{index:05d}.png"
        if info.kind == "image" or save_frames:
            annotated_path = output_dir / ("annotated.png" if info.kind == "image" else relative)
            annotated.save(annotated_path)
            exports.append(str(annotated_path))
            annotated_files.append(annotated_path)
        if save_heads:
            import numpy as np

            head_path = output_dir / f"heads-{index:05d}.npz"
            payload = {name: np.asarray(array) for name, array in inference["heads"].items()}
            if save_heads and "workspace" in inference:
                payload["workspace"] = inference["workspace"]
            np.savez_compressed(head_path, **payload)
            exports.append(str(head_path))

        per_frame.append(
            {
                "frame_index": index,
                "seconds": seconds,
                "crop": {"x": crop.x, "y": crop.y, "width": crop.width, "height": crop.height, "mode": crop.mode},
                "input_sha256": inference["input_sha256"],
                "network_seconds": inference["seconds"],
                "decoder_status": report["status"],
                "decoder_visited": report["visited"],
                "decoder_accepted": report["accepted"],
                "decoder_backend": report["backend"],
                "records_model_space": report["records"],
                "boxes_source_pixels": [
                    {"cx": x, "cy": y, "w": w, "h": h, "score": s, "cls": c} for x, y, w, h, s, c in mapped
                ],
            }
        )
        emit(
            progress,
            kind="frame",
            frame_index=index,
            seconds=seconds,
            done=processed,
            total=planned,
            frame_seconds=inference["seconds"],
            detections=len(mapped),
            decoder_status=report["status"],
            crop_mode=crop.mode,
            annotated=annotated,
            mapped=mapped,
            report=report,
        )

    total_seconds = time.monotonic() - started

    csv_path = output_dir / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["frame_index", "frame_seconds", "cx_model", "cy_model", "w_model", "h_model", "score", "cls",
             "cx_source", "cy_source", "w_source", "h_source", "crop_mode", "decoder_status"]
        )
        for frame in per_frame:
            model_records = frame["records_model_space"]
            source_boxes = frame["boxes_source_pixels"]
            for record, box in zip(model_records, source_boxes):
                writer.writerow(
                    [frame["frame_index"], round(frame["seconds"], 3), *record,
                     round(box["cx"], 2), round(box["cy"], 2), round(box["w"], 2), round(box["h"], 2),
                     frame["crop"]["mode"], frame["decoder_status"]]
                )
    exports.append(str(csv_path))

    video_export = None
    if info.kind == "video" and export_video and annotated_files:
        fps = float(video_fps or (1.0 / max(per_second, 0.1)))
        video_path = output_dir / "annotated.mp4"
        from PIL import Image

        def generated():
            for path in annotated_files:
                with Image.open(path) as image:
                    yield image

        frames_module.write_annotated_video(video_path, info.width, info.height, fps, generated())
        video_export = str(video_path)
        exports.append(video_export)

    summary = {
        "model": status,
        "source": str(source_path),
        "source_kind": info.kind,
        "source_size": [info.width, info.height],
        "fps": info.fps,
        "duration": info.duration,
        "sampling": {"per_second": per_second, "stride_frames": stride, "planned": planned, "available": available,
                     "truncated_to_first_seconds": planned * per_second if info.kind == "video" else None},
        "processed_frames": processed,
        "detections_total": detections_total,
        "network_seconds_total": total_infer,
        "total_seconds": total_seconds,
        "frames": per_frame,
        "exports": exports,
        "video_export": video_export,
        "note": "模型本體、權重、command order 皆未修改；GUI 只加咗 3:2 center crop + overlay 顯示",
    }
    json_path = output_dir / "result.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    exports.append(str(json_path))

    emit(progress, kind="done", processed=processed, total=planned, total_seconds=total_seconds,
         detections_total=detections_total, video_export=video_export, exports=exports)

    first_crop = per_frame[0]["crop"] if per_frame else {}
    return JobResult(
        kind=info.kind,
        source=str(source_path),
        output=str(output_dir),
        crop_mode=first_crop.get("mode", "none"),
        crop=first_crop,
        stride=stride,
        planned_frames=planned,
        processed_frames=processed,
        truncated=truncated,
        per_frame_seconds=(total_infer / processed) if processed else 0.0,
        total_seconds=total_seconds,
        detections_total=detections_total,
        video_export=video_export,
        exports=exports,
        frames=per_frame,
        model_status=status,
    )
