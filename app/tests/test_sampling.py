#!/usr/bin/env python3
"""Sampling-plan rules: stride, the min(60, fps) cap, truncation reporting.

Run:  python app/tests/test_sampling.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import frames as fm

failures = []


def check(label: str, got, want) -> None:
    if got != want:
        failures.append(f"{label}: got {got}, want {want}")
    else:
        print(f"  ok {label} -> {got}")


video = fm.SourceInfo("video", "clip.mp4", 1280, 720, fps=25.0, duration=12.0)

# stride = fps / per_second, never 0
check("25 fps @ 1/s", fm.sampling_plan(video, 1.0, None), (25, 12))
check("25 fps @ 2/s", fm.sampling_plan(video, 2.0, None), (12, 24))    # 12.5 rounds to 12; 12 s -> 24 samples
check("25 fps @ 10/s", fm.sampling_plan(video, 10.0, None), (2, 25))   # 120 samples, cap min(60, 25) = 25
check("60 fps @ 1/s", fm.sampling_plan(
    fm.SourceInfo("video", "c", 640, 480, fps=60.0, duration=300.0), 1.0, None), (60, 60))
check("30 fps @ 1/s cap", fm.sampling_plan(
    fm.SourceInfo("video", "c", 640, 480, fps=30.0, duration=300.0), 1.0, None), (30, 30))
check("3 fps @ 1/s cap", fm.sampling_plan(
    fm.SourceInfo("video", "c", 640, 480, fps=3.0, duration=300.0), 1.0, None), (3, 3))
check("explicit max-frames wins", fm.sampling_plan(video, 2.0, 4), (12, 4))
check("sub-second sampling stays >= 1", fm.sampling_plan(video, 40.0, None), (1, 25))
check("image input", fm.sampling_plan(fm.SourceInfo("image", "p.jpg", 800, 600), 2.0, None), (1, 1))

# cap is min(60, fps), per the brief
check("cap 25 fps", fm.default_max_frames(25.0), 25)
check("cap 59.4 fps -> 59", fm.default_max_frames(59.4), 59)
check("cap 60 fps", fm.default_max_frames(60.0), 60)
check("cap 120 fps -> 60", fm.default_max_frames(120.0), 60)

# sampling count scales with duration * per_second
check("samples for 12 s @ 2/s", math.ceil(video.duration * 2.0), 24)

if failures:
    print("\n".join(f"FAIL {f}" for f in failures))
    print("SAMPLING TESTS FAILED")
    raise SystemExit(1)
print("ALL SAMPLING TESTS PASSED")
