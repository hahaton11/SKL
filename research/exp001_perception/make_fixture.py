"""Generate synthetic EXP-001 runs so analyze.py can be validated without a camera.

Not test data for the experiment -- a regression harness for the *analysis*. It exists because
writing it immediately exposed two real bugs: PowerShell not expanding globs passed to a native
command, and a partial final time bin being read as a catastrophic FPS drop (a false throttling
flag on a perfectly flat run).

Three fixtures, each with one intended detection:

    good       flat FPS, high hand presence      -> no flags
    fast       high dropout + a long no-hand gap -> dropout flag + swallowed-sign flag
    throttled  FPS decaying across the run       -> throttling flag

Usage
-----
    python make_fixture.py out_dir
    python analyze.py out_dir/*.jsonl --verbose
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path


def meta(tags: dict[str, str], fps: float) -> dict:
    return {
        "record": "meta",
        "experiment": "EXP-001",
        "started_utc": "2026-10-05T12:00:00Z",
        "source": "webcam",
        "video": None,
        "tags": tags,
        "note": "SYNTHETIC FIXTURE -- not a measurement",
        "requested": {"width": 1280, "height": 720, "fps": 30},
        "reported": {"width": 1280, "height": 720, "fps": fps},
        "model": "holistic_landmarker.task",
        "running_mode": "VIDEO",
        "mediapipe_version": "fixture",
        "opencv_version": "fixture",
        "python": "fixture",
        "platform": "SYNTHETIC",
        "processor": "synthetic",
    }


def probes(rng: random.Random, jit: float, hands: bool, i: int) -> dict[str, list[float]]:
    p: dict[str, list[float]] = {}
    if hands:
        for side in ("right_hand", "left_hand"):
            p[f"{side}.wrist"] = [
                0.5 + 0.1 * math.sin(i / 7) + rng.gauss(0, jit),
                0.5 + 0.1 * math.cos(i / 7) + rng.gauss(0, jit),
                0.0,
            ]
            p[f"{side}.index_tip"] = [
                0.52 + rng.gauss(0, jit),
                0.48 + rng.gauss(0, jit),
                0.0,
            ]
    p["face.upper_lip"] = [0.5 + rng.gauss(0, jit / 2), 0.40 + rng.gauss(0, jit / 2), 0.0]
    p["face.left_brow"] = [0.47 + rng.gauss(0, jit / 2), 0.33 + rng.gauss(0, jit / 2), 0.0]
    p["pose.nose"] = [0.5, 0.35, 0.0]
    return p


def write(
    out_dir: Path,
    name: str,
    tags: dict[str, str],
    n_frames: int,
    fps_fn,
    hand_rate: float,
    jit: float,
    gap_at: int | None = None,
    gap_len: int = 0,
    seed: int = 7,
) -> None:
    rng = random.Random(seed)
    path = out_dir / name
    t = 0.0
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta(tags, 30.0)) + "\n")
        for i in range(n_frames):
            t += 1.0 / fps_fn(t)
            in_gap = gap_at is not None and gap_at <= i < gap_at + gap_len
            hands = (not in_gap) and rng.random() < hand_rate
            fh.write(
                json.dumps(
                    {
                        "record": "frame",
                        "i": i,
                        "ts_ms": int(t * 1000),
                        "wall_s": round(t, 4),
                        "infer_ms": round(rng.gauss(22, 4), 3),
                        "present": {
                            "face": True,
                            "pose": True,
                            "left_hand": hands,
                            "right_hand": hands,
                        },
                        "counts": {
                            "face": 478,
                            "pose": 33,
                            "left_hand": 21 if hands else 0,
                            "right_hand": 21 if hands else 0,
                        },
                        "probes": probes(rng, jit, hands, i),
                        "warmup": i < 10,
                    }
                )
                + "\n"
            )
    print(f"wrote {path} ({n_frames} frames)")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    write(
        out_dir,
        "fixture_good.jsonl",
        {"fixture": "good", "speed": "natural"},
        n_frames=1800,
        fps_fn=lambda t: 28.0,
        hand_rate=0.98,
        jit=0.0012,
    )
    write(
        out_dir,
        "fixture_fast.jsonl",
        {"fixture": "fast", "speed": "fast"},
        n_frames=1800,
        fps_fn=lambda t: 27.0,
        hand_rate=0.78,
        jit=0.004,
        gap_at=600,
        gap_len=14,
    )
    write(
        out_dir,
        "fixture_throttled.jsonl",
        {"fixture": "throttled", "run": "10min"},
        n_frames=9000,
        fps_fn=lambda t: max(9.0, 28.0 - t / 30.0),
        hand_rate=0.96,
        jit=0.002,
    )

    print(
        "\nexpected: no flags for 'good'; dropout + swallowed-sign for 'fast'; "
        "throttling for 'throttled'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
