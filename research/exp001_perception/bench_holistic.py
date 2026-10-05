"""EXP-001 / Baseline 0 — MediaPipe Holistic Landmarker perception benchmark.

Measures the perception stage alone: no model, no dataset, no app. The question is whether
landmarks are reliable at natural signing speed and at what frame rate, which is the premise
every downstream decision in ADR-002 rests on.

Writes one JSONL per run: a `meta` record, then one `frame` record per frame. Nothing is
aggregated here on purpose -- `analyze.py` does that, so a run can be re-analysed without
re-recording it.

Usage
-----
    # one-off: download the model bundle (13 MB, float16) into models/
    python bench_holistic.py --fetch-model

    # webcam, 60 s, tagged with the condition being swept
    python bench_holistic.py --source webcam --duration 60 \
        --tag distance=2m --tag lighting=dim --tag speed=fast

    # a recorded phone clip (this is how we measure landmark quality on phone footage
    # without writing an Android app yet)
    python bench_holistic.py --source file --video runs/raw/phone_natural.mp4 \
        --tag device=phone --tag speed=natural
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "holistic_landmarker.task"
DEFAULT_RUNS = REPO_ROOT / "runs"

# Landmarks we keep coordinates for, to measure jitter without storing all 543 per frame.
# Hands carry the lexical content; mouth and eyebrow carry KRSL's non-manual grammar, so their
# stability is measured explicitly -- a face channel that is present but jittery is a different
# failure from one that is absent.
POSE_PROBES = {"nose": 0, "left_shoulder": 11, "right_shoulder": 12}
HAND_PROBES = {"wrist": 0, "index_tip": 8}
FACE_PROBES = {"upper_lip": 13, "lower_lip": 14, "left_brow": 105, "right_brow": 334}


@dataclass
class Channels:
    """Per-frame presence and probe coordinates, one entry per holistic channel."""

    present: dict[str, bool] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    probes: dict[str, list[float]] = field(default_factory=dict)


def _as_landmark_list(value):
    """Normalise a result field to a flat list of landmarks.

    Holistic returns flat lists (single subject), but sibling Tasks APIs return list-of-lists.
    Accepting both keeps this script working across mediapipe point releases instead of
    failing with an opaque index error.
    """
    if not value:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)):
        return list(first) if first else []
    if hasattr(first, "x"):
        return list(value)
    inner = getattr(first, "landmark", None)
    return list(inner) if inner else []


def _probe(landmarks, indices: dict[str, int], channel: str, out: Channels) -> None:
    for name, idx in indices.items():
        if idx < len(landmarks):
            lm = landmarks[idx]
            out.probes[f"{channel}.{name}"] = [
                round(float(lm.x), 5),
                round(float(lm.y), 5),
                round(float(getattr(lm, "z", 0.0)), 5),
            ]


def extract(result) -> Channels:
    ch = Channels()
    sources = {
        "face": _as_landmark_list(getattr(result, "face_landmarks", None)),
        "pose": _as_landmark_list(getattr(result, "pose_landmarks", None)),
        "left_hand": _as_landmark_list(getattr(result, "left_hand_landmarks", None)),
        "right_hand": _as_landmark_list(getattr(result, "right_hand_landmarks", None)),
    }
    for name, lms in sources.items():
        ch.present[name] = len(lms) > 0
        ch.counts[name] = len(lms)

    _probe(sources["pose"], POSE_PROBES, "pose", ch)
    _probe(sources["face"], FACE_PROBES, "face", ch)
    _probe(sources["left_hand"], HAND_PROBES, "left_hand", ch)
    _probe(sources["right_hand"], HAND_PROBES, "right_hand", ch)
    return ch


def fetch_model(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"model already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"downloading {MODEL_URL}\n  -> {dest}")
    urllib.request.urlretrieve(MODEL_URL, dest)
    print(f"done: {dest.stat().st_size / 1e6:.1f} MB")


def build_landmarker(model_path: Path, running_mode):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def open_source(args):
    import cv2

    if args.source == "webcam":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if args.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if args.request_fps:
            cap.set(cv2.CAP_PROP_FPS, args.request_fps)
    else:
        if not args.video:
            sys.exit("--source file requires --video PATH")
        cap = cv2.VideoCapture(str(args.video))

    if not cap.isOpened():
        sys.exit(f"could not open source: {args.source} {args.video or args.camera}")
    return cap


def main() -> int:
    p = argparse.ArgumentParser(description="EXP-001 Holistic Landmarker benchmark")
    p.add_argument("--fetch-model", action="store_true", help="download the model bundle and exit")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--source", choices=["webcam", "file"], default="webcam")
    p.add_argument("--video", type=Path, help="input clip for --source file")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--request-fps", type=int, default=30, help="requested camera fps")
    p.add_argument("--duration", type=float, default=60.0, help="seconds (webcam only)")
    p.add_argument("--max-frames", type=int, default=0, help="0 = unlimited")
    p.add_argument("--warmup", type=int, default=10, help="frames excluded from timing")
    p.add_argument("--out", type=Path, help="output .jsonl (default: runs/<timestamp>_<tags>.jsonl)")
    p.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="condition tag, repeatable: distance, lighting, speed, occlusion, background, device",
    )
    p.add_argument("--note", default="", help="free-text note stored in the run metadata")
    args = p.parse_args()

    if args.fetch_model:
        fetch_model(args.model)
        return 0

    if not args.model.exists():
        sys.exit(f"model bundle missing: {args.model}\nrun: {sys.argv[0]} --fetch-model")

    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    tags: dict[str, str] = {}
    for raw in args.tag:
        if "=" not in raw:
            sys.exit(f"bad --tag {raw!r}, expected KEY=VALUE")
        k, v = raw.split("=", 1)
        tags[k.strip()] = v.strip()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = "_".join(f"{k}-{v}" for k, v in sorted(tags.items())) or args.source
    out_path = args.out or (DEFAULT_RUNS / f"{stamp}_{slug}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = open_source(args)
    # The *actual* camera rate, not the requested one. They differ, and the difference
    # propagates silently into every temporal model downstream.
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    running_mode = vision.RunningMode.VIDEO
    landmarker = build_landmarker(args.model, running_mode)

    meta = {
        "record": "meta",
        "experiment": "EXP-001",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": args.source,
        "video": str(args.video) if args.video else None,
        "tags": tags,
        "note": args.note,
        "requested": {"width": args.width, "height": args.height, "fps": args.request_fps},
        "reported": {"width": frame_w, "height": frame_h, "fps": reported_fps},
        "model": args.model.name,
        "running_mode": "VIDEO",
        "mediapipe_version": getattr(mp, "__version__", "unknown"),
        "opencv_version": cv2.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    frames = 0
    t_start = time.perf_counter()
    deadline = t_start + args.duration if args.source == "webcam" else None

    print(f"writing {out_path}")
    print(f"source {frame_w}x{frame_h} @ reported {reported_fps or '?'} fps; tags={tags or '-'}")

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            loop_t = time.perf_counter()
            if args.source == "webcam":
                ts_ms = int((loop_t - t_start) * 1000)
            else:
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts_ms = int(pos) if pos and pos > 0 else int(
                    frames * 1000 / (reported_fps or 30.0)
                )
            # VIDEO mode requires strictly increasing timestamps.
            ts_ms = max(ts_ms, frames)

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            infer_t0 = time.perf_counter()
            result = landmarker.detect_for_video(mp_image, ts_ms)
            infer_ms = (time.perf_counter() - infer_t0) * 1000

            ch = extract(result)
            fh.write(
                json.dumps(
                    {
                        "record": "frame",
                        "i": frames,
                        "ts_ms": ts_ms,
                        "wall_s": round(loop_t - t_start, 4),
                        "infer_ms": round(infer_ms, 3),
                        "present": ch.present,
                        "counts": ch.counts,
                        "probes": ch.probes,
                        "warmup": frames < args.warmup,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            frames += 1
            if frames % 60 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"  {frames} frames, {elapsed:.1f}s, {frames / elapsed:.1f} fps (loop)")

            if args.max_frames and frames >= args.max_frames:
                break
            if deadline and time.perf_counter() >= deadline:
                break

    cap.release()
    landmarker.close()

    elapsed = time.perf_counter() - t_start
    print(f"\n{frames} frames in {elapsed:.1f}s -> {frames / max(elapsed, 1e-9):.1f} fps (loop)")
    print(f"raw: {out_path}")
    print(f"next: python {Path(__file__).with_name('analyze.py').name} {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
