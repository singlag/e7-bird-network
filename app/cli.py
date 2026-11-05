"""Headless runner — the same pipeline the GUI uses, for batch runs and self-tests.

    python app/cli.py --input photo.jpg
    python app/cli.py --input clip.mp4 --per-second 1 --max-frames 30
    python app/cli.py --input photo.jpg --decoder port --save-heads
"""

import argparse
import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
for path in (APP, APP.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import job  # noqa: E402
import model_registry  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="E7 bird detection engine (headless)")
    parser.add_argument("--input", required=True, help="photo or video file")
    parser.add_argument("--output", help="output directory (default: <input-name>-e7-out)")
    parser.add_argument("--model", default=model_registry.default_key())
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--decoder", choices=["auto", "native", "port"], default="auto")
    parser.add_argument("--per-second", type=float, default=1.0, help="video frames sampled per second")
    parser.add_argument("--max-frames", type=int, help="override the min(60, fps) cap")
    parser.add_argument("--save-heads", action="store_true", help="write fine/coarse head npz per frame")
    parser.add_argument("--save-frames", action="store_true", help="write an annotated PNG for every sampled frame")
    parser.add_argument("--no-video", action="store_true", help="skip annotated.mp4 for video input")
    parser.add_argument("--video-fps", type=float, help="annotated.mp4 frame rate (default: same as --per-second)")
    return parser.parse_args(argv)


def report(event):
    kind = event.get("kind")
    if kind == "plan":
        af = event.get("af_box")
        if af:
            print(f"camera AF box: {int(af['x'])},{int(af['y'])} "
                  f"{int(af['w'])}x{int(af['h'])} ({af.get('mode_name')})")
        detail = f" @ {event['fps']} fps" if event["source_kind"] == "video" else ""
        print(f"input   : {event['source']} ({event['source_kind']} {event['width']}x{event['height']}{detail})")
        suffix = " (truncated by the cap)" if event["truncated"] else ""
        print(f"sampling: stride {event['stride']} frames, {event['planned']} to process{suffix}")
    elif kind == "frame":
        print(f"frame {event['frame_index']:5d} t={event['seconds']:6.2f}s "
              f"{event['done']}/{event['total']} infer {event['frame_seconds']:.1f}s "
              f"detections {event['detections']} (crop {event['crop_mode']})")
    elif kind == "done":
        print(f"done    : {event['processed']} frames, {event['detections_total']} detections, "
              f"{event['total_seconds']:.1f}s total")


def main(argv=None):
    args = parse_args(argv)
    source = Path(args.input).expanduser()
    if not source.is_file():
        print(f"input not found: {source}", file=sys.stderr)
        return 2
    output = Path(args.output).expanduser() if args.output else source.with_name(source.stem + "-e7-out")

    model = model_registry.build(args.model, threads=args.threads, decoder_mode=args.decoder)
    spec = model_registry.find(args.model)
    if not spec.decodable:
        # The file was found by hash and loads/verifies, but this network has
        # no published decode contract — say so and stop before any job.
        model.load(log=print)
        print(json.dumps({"model": spec.label, "loaded": True, "decodable": False,
                          "reason": "no published decode contract — cannot "
                                    "produce detection boxes"},
                         ensure_ascii=False))
        return 0
    result = job.run_job(
        model,
        source,
        output,
        per_second=args.per_second,
        max_frames=args.max_frames,
        save_heads=args.save_heads,
        save_frames=args.save_frames,
        export_video=not args.no_video,
        video_fps=args.video_fps,
        progress=report,
        log=lambda message: print(message, flush=True),
    )
    print(json.dumps({"output": result.output, "model": result.model_status,
                      "processed_frames": result.processed_frames,
                      "detections_total": result.detections_total,
                      "per_frame_seconds": round(result.per_frame_seconds, 2),
                      "video_export": result.video_export,
                      "exports": result.exports}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
