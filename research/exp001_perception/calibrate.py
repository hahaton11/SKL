"""Live calibration view — set up a condition correctly *before* recording a benchmark run.

**This is not a measurement tool.** It draws an overlay, and drawing costs frames, so any FPS it
shows is the FPS of "landmarker plus rendering", not of the pipeline. Measurements come from
`bench_holistic.py`, which stays headless on purpose.

What it is for: framing yourself, checking that both hands stay in shot, making "2 m" mean the same
thing in every run, and watching which channel drops out when you sign fast. Condition tags are only
worth something if the conditions are reproducible, and eyeballing distance is not reproducible.

Keys
----
    q / Esc   quit
    SPACE     print the current readings to the console
    r         reset the rolling windows
    o         toggle the landmark overlay (HUD stays)

Usage
-----
    python calibrate.py
    python calibrate.py --camera 1 --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "holistic_landmarker.task"

# Drawn as polylines so a hand reads as a hand rather than a dot cloud.
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]
POSE_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
# The non-manual probes: mouth contour and eyebrows carry KRSL grammar, so they get drawn.
FACE_PROBE_IDS = [13, 14, 61, 291, 105, 334, 70, 300]

GREEN = (120, 220, 120)
AMBER = (80, 190, 240)
RED = (90, 90, 235)
GREY = (170, 170, 170)
WHITE = (240, 240, 240)


def flat(value):
    """Holistic returns flat landmark lists; siblings return list-of-lists. Accept both."""
    if not value:
        return []
    first = value[0]
    if isinstance(first, (list, tuple)):
        return list(first) if first else []
    if hasattr(first, "x"):
        return list(value)
    inner = getattr(first, "landmark", None)
    return list(inner) if inner else []


def px(lm, w: int, h: int) -> tuple[int, int]:
    return int(lm.x * w), int(lm.y * h)


def in_frame(lms, margin: float = 0.02) -> bool:
    """True when every landmark sits inside the frame with a little slack at the edges."""
    return all(margin <= lm.x <= 1 - margin and margin <= lm.y <= 1 - margin for lm in lms)


def draw_overlay(frame, left, right, face, pose) -> None:
    """Draw hands as skeletons, shoulders/arms as lines, and the non-manual probes as dots.

    Separated out so it can be exercised with synthetic landmarks -- the drawing path is the one
    part of this script that a subject-free test clip never reaches.
    """
    import cv2

    h, w = frame.shape[:2]
    for lms, colour in ((left, GREEN), (right, AMBER)):
        if not lms:
            continue
        for a, b in HAND_BONES:
            if a < len(lms) and b < len(lms):
                cv2.line(frame, px(lms[a], w, h), px(lms[b], w, h), colour, 2)
    if pose:
        for a, b in POSE_BONES:
            if a < len(pose) and b < len(pose):
                cv2.line(frame, px(pose[a], w, h), px(pose[b], w, h), GREY, 2)
    for idx in FACE_PROBE_IDS:
        if idx < len(face):
            cv2.circle(frame, px(face[idx], w, h), 3, WHITE, -1)


def main() -> int:
    p = argparse.ArgumentParser(description="EXP-001 live calibration view")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument(
        "--video",
        type=Path,
        help="review a recorded clip with the overlay instead of opening the camera "
        "(use this on phone footage, and to try the view without a webcam)",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--window", type=int, default=60, help="rolling window in frames")
    p.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    p.add_argument("--headless", action="store_true", help="no window; print readings only")
    args = p.parse_args()

    if not args.model.exists():
        sys.exit(
            f"model bundle missing: {args.model}\n"
            f"run: python {Path(__file__).with_name('bench_holistic.py').name} --fetch-model"
        )

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if args.video:
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            sys.exit(f"could not open clip: {args.video}")
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            sys.exit(f"could not open camera {args.camera}")

    landmarker = vision.HolisticLandmarker.create_from_options(
        vision.HolisticLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(args.model)),
            running_mode=vision.RunningMode.VIDEO,
        )
    )

    fps_hist: collections.deque[float] = collections.deque(maxlen=args.window)
    infer_hist: collections.deque[float] = collections.deque(maxlen=args.window)
    hand_hist: collections.deque[bool] = collections.deque(maxlen=args.window)
    both_hist: collections.deque[bool] = collections.deque(maxlen=args.window)
    face_hist: collections.deque[bool] = collections.deque(maxlen=args.window)
    show_overlay = True

    t0 = time.perf_counter()
    prev = t0
    frames = 0
    print("calibration view -- q to quit, SPACE to print readings, r to reset, o to toggle overlay")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.no_mirror and not args.video:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.perf_counter()
        fps_hist.append(1.0 / max(now - prev, 1e-6))
        prev = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t_inf = time.perf_counter()
        res = landmarker.detect_for_video(mp_image, max(int((now - t0) * 1000), frames))
        infer_hist.append((time.perf_counter() - t_inf) * 1000)

        left = flat(getattr(res, "left_hand_landmarks", None))
        right = flat(getattr(res, "right_hand_landmarks", None))
        face = flat(getattr(res, "face_landmarks", None))
        pose = flat(getattr(res, "pose_landmarks", None))

        hand_hist.append(bool(left or right))
        both_hist.append(bool(left and right))
        face_hist.append(bool(face))

        if show_overlay and not args.headless:
            draw_overlay(frame, left, right, face, pose)

        # --- readings ---------------------------------------------------------------
        fps = sum(fps_hist) / len(fps_hist)
        infer = sum(infer_hist) / len(infer_hist)
        any_hand_pct = 100.0 * sum(hand_hist) / len(hand_hist)
        both_pct = 100.0 * sum(both_hist) / len(both_hist)
        face_pct = 100.0 * sum(face_hist) / len(face_hist)

        # Distance proxy: shoulder width in normalised units. Reproducible across sessions in a
        # way that "about two metres" is not -- note the value you use for each distance tag.
        shoulder = None
        if len(pose) > 12:
            shoulder = abs(pose[11].x - pose[12].x)

        brightness = float(rgb[:, :, :].mean())

        hands_out = False
        for lms in (left, right):
            if lms and not in_frame(lms):
                hands_out = True

        warnings = []
        if any_hand_pct < 90:
            warnings.append(f"hand dropout {100 - any_hand_pct:.0f}%")
        if hands_out:
            warnings.append("hand leaving frame")
        if shoulder is not None and shoulder < 0.14:
            warnings.append("too far from camera")
        if shoulder is not None and shoulder > 0.55:
            warnings.append("too close")
        if brightness < 55:
            warnings.append("too dark")
        if not face:
            warnings.append("no face -- non-manual channel blind")

        lines = [
            (f"render fps {fps:4.1f}   infer {infer:5.1f} ms   (NOT a measurement)", WHITE),
            (
                f"any hand {any_hand_pct:5.1f}%   both {both_pct:5.1f}%   face {face_pct:5.1f}%",
                GREEN if any_hand_pct >= 90 else RED,
            ),
            (
                "shoulder width "
                + (f"{shoulder:.3f}" if shoulder is not None else "  -  ")
                + f"   brightness {brightness:3.0f}",
                GREY,
            ),
        ]
        y = 28
        for text, colour in lines:
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 1)
            y += 26
        for warn in warnings:
            cv2.putText(frame, "! " + warn, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, "! " + warn, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)
            y += 26

        frames += 1
        if args.headless:
            if frames % 30 == 0:
                print(
                    f"frame {frames}: any_hand={any_hand_pct:.1f}% both={both_pct:.1f}% "
                    f"face={face_pct:.1f}% infer={infer:.1f}ms "
                    f"shoulder={shoulder if shoulder is None else round(shoulder, 3)} "
                    f"brightness={brightness:.0f} warnings={warnings or '-'}"
                )
            continue

        cv2.imshow("EXP-001 calibration (not a measurement)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            for d in (fps_hist, infer_hist, hand_hist, both_hist, face_hist):
                d.clear()
            print("windows reset")
        if key == ord("o"):
            show_overlay = not show_overlay
        if key == ord(" "):
            print(
                f"render_fps={fps:.1f} infer_ms={infer:.1f} any_hand={any_hand_pct:.1f}% "
                f"both={both_pct:.1f}% face={face_pct:.1f}% "
                f"shoulder={shoulder if shoulder is None else round(shoulder, 3)} "
                f"brightness={brightness:.0f} warnings={warnings or '-'}"
            )

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("\nCalibrated? Now record the real run with bench_holistic.py (headless, no overlay).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
