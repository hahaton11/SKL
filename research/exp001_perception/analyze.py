"""EXP-001 / Baseline 0 — turn raw benchmark runs into the numbers that decide ADR-002.

Reads one or more .jsonl runs from bench_holistic.py and prints a markdown table ready to paste
into the vault's `04_Experiments/Baselines/EXP-001 — Baseline 0 Perception.md`.

Reporting discipline (from the vault's Data Quality rules): latency is always reported with the
hardware it was measured on, and FPS is reported as a decay check rather than a single mean,
because a two-minute benchmark hides thermal throttling and a conversation is not two minutes.

Usage
-----
    python analyze.py runs/*.jsonl
    python analyze.py runs/20261005-143000_distance-2m.jsonl --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from pathlib import Path

HAND_CHANNELS = ("left_hand", "right_hand")
ALL_CHANNELS = ("left_hand", "right_hand", "face", "pose")


def pct(xs: list[float], q: float) -> float:
    """Percentile with linear interpolation; q in [0, 100]."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def load(path: Path) -> tuple[dict, list[dict]]:
    meta: dict = {}
    frames: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record") == "meta":
                meta = rec
            elif rec.get("record") == "frame":
                frames.append(rec)
    return meta, frames


def jitter_px_equiv(frames: list[dict], probe: str, window: int = 15) -> float:
    """Median within-window stddev of a probe's position, in normalised units.

    Not a stillness measurement unless the run was recorded still -- on a moving run it mixes
    real motion with sensor noise, so only compare it across runs of the same condition, and
    prefer a dedicated `--tag condition=stillness` run for the headline number.
    """
    xs = [(f["probes"].get(probe) or [None])[0] for f in frames]
    ys = [(f["probes"].get(probe) or [None, None])[1] for f in frames]
    windows: list[float] = []
    for start in range(0, max(len(xs) - window, 0), window):
        wx = [v for v in xs[start : start + window] if v is not None]
        wy = [v for v in ys[start : start + window] if v is not None]
        if len(wx) >= 3 and len(wy) >= 3:
            windows.append(math.hypot(stats.pstdev(wx), stats.pstdev(wy)))
    return stats.median(windows) if windows else float("nan")


def fps_series(frames: list[dict], bin_s: float = 30.0) -> list[tuple[float, float]]:
    """Loop FPS per time bin -- the decay check. Returns [(bin_start_s, fps)]."""
    if not frames:
        return []
    bins: dict[int, int] = {}
    for f in frames:
        bins[int(f["wall_s"] // bin_s)] = bins.get(int(f["wall_s"] // bin_s), 0) + 1
    return [(b * bin_s, n / bin_s) for b, n in sorted(bins.items())]


def summarise(meta: dict, frames: list[dict]) -> dict:
    live = [f for f in frames if not f.get("warmup")]
    n = len(live)
    out: dict = {
        "file": meta.get("_file", "?"),
        "tags": meta.get("tags", {}),
        "source": meta.get("source"),
        "resolution": f"{meta.get('reported', {}).get('width')}x{meta.get('reported', {}).get('height')}",
        "reported_fps": meta.get("reported", {}).get("fps"),
        "frames": n,
        "platform": meta.get("platform", "?"),
        "mediapipe": meta.get("mediapipe_version", "?"),
    }
    if not n:
        return out

    duration = live[-1]["wall_s"] - live[0]["wall_s"]
    out["duration_s"] = round(duration, 1)
    out["loop_fps"] = round(n / duration, 1) if duration > 0 else float("nan")
    # The actual delivered rate, which is what downstream temporal models actually see.
    out["actual_capture_fps"] = out["loop_fps"]

    infer = [f["infer_ms"] for f in live]
    out["infer_p50_ms"] = round(pct(infer, 50), 1)
    out["infer_p95_ms"] = round(pct(infer, 95), 1)
    out["infer_max_ms"] = round(max(infer), 1)

    for chan in ALL_CHANNELS:
        present = sum(1 for f in live if f["present"].get(chan))
        out[f"{chan}_presence"] = round(100.0 * present / n, 1)
    out["both_hands_presence"] = round(
        100.0 * sum(1 for f in live if all(f["present"].get(c) for c in HAND_CHANNELS)) / n, 1
    )
    out["any_hand_presence"] = round(
        100.0 * sum(1 for f in live if any(f["present"].get(c) for c in HAND_CHANNELS)) / n, 1
    )

    # Longest consecutive run with no hand at all -- a 300 ms gap mid-sign is a different
    # problem from the same dropout scattered one frame at a time.
    worst = cur = 0
    for f in live:
        cur = 0 if any(f["present"].get(c) for c in HAND_CHANNELS) else cur + 1
        worst = max(worst, cur)
    out["longest_no_hand_frames"] = worst
    if out["loop_fps"] and not math.isnan(out["loop_fps"]):
        out["longest_no_hand_ms"] = round(1000.0 * worst / out["loop_fps"], 0)

    out["jitter_right_wrist"] = round(jitter_px_equiv(live, "right_hand.wrist"), 5)
    out["jitter_left_wrist"] = round(jitter_px_equiv(live, "left_hand.wrist"), 5)
    out["jitter_upper_lip"] = round(jitter_px_equiv(live, "face.upper_lip"), 5)
    out["jitter_left_brow"] = round(jitter_px_equiv(live, "face.left_brow"), 5)

    series = fps_series(live)
    out["fps_bins"] = [(round(t), round(v, 1)) for t, v in series]
    if len(series) >= 2:
        first, last = series[0][1], series[-1][1]
        out["fps_decay_pct"] = round(100.0 * (first - last) / first, 1) if first else float("nan")
    return out


def md_table(rows: list[dict]) -> str:
    cols = [
        ("tags", "Condition"),
        ("resolution", "Res"),
        ("frames", "Frames"),
        ("duration_s", "Dur s"),
        ("loop_fps", "FPS"),
        ("fps_decay_pct", "FPS decay %"),
        ("infer_p50_ms", "Infer p50"),
        ("infer_p95_ms", "Infer p95"),
        ("both_hands_presence", "Both hands %"),
        ("any_hand_presence", "Any hand %"),
        ("face_presence", "Face %"),
        ("pose_presence", "Pose %"),
        ("longest_no_hand_ms", "Max gap ms"),
        ("jitter_right_wrist", "Jitter R wrist"),
        ("jitter_left_brow", "Jitter L brow"),
    ]
    head = "| " + " | ".join(label for _, label in cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    lines = [head, rule]
    for r in rows:
        cells = []
        for key, _ in cols:
            v = r.get(key, "")
            if key == "tags":
                v = ", ".join(f"{k}={val}" for k, val in sorted((v or {}).items())) or "-"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verdict(rows: list[dict]) -> list[str]:
    """Flag the conditions that would fail EXP-001's pass criterion."""
    notes = []
    for r in rows:
        tag = ", ".join(f"{k}={v}" for k, v in sorted((r.get("tags") or {}).items())) or r.get("file")
        fps = r.get("loop_fps")
        hands = r.get("any_hand_presence")
        decay = r.get("fps_decay_pct")
        gap = r.get("longest_no_hand_ms")
        if isinstance(fps, float) and not math.isnan(fps) and fps < 15:
            notes.append(f"- **{tag}**: loop FPS {fps} is below the 15 FPS target")
        if isinstance(hands, float) and hands < 90:
            notes.append(f"- **{tag}**: any-hand presence only {hands}% -- landmark dropout is high")
        if isinstance(decay, float) and not math.isnan(decay) and decay > 20:
            notes.append(f"- **{tag}**: FPS decayed {decay}% over the run -- possible throttling")
        if isinstance(gap, float) and gap > 300:
            notes.append(f"- **{tag}**: longest no-hand gap {gap:.0f} ms -- long enough to swallow a sign")
    return notes or ["- no condition tripped the EXP-001 failure thresholds"]


def main() -> int:
    p = argparse.ArgumentParser(description="Summarise EXP-001 runs")
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--verbose", action="store_true", help="also print the per-bin FPS series")
    args = p.parse_args()

    rows = []
    for path in args.runs:
        if not path.exists():
            print(f"skip (missing): {path}")
            continue
        meta, frames = load(path)
        meta["_file"] = path.name
        rows.append(summarise(meta, frames))

    if not rows:
        print("no runs loaded")
        return 1

    print("\n### EXP-001 results\n")
    print(md_table(rows))

    print("\n**Environment**\n")
    seen = set()
    for r in rows:
        key = (r.get("platform"), r.get("mediapipe"))
        if key not in seen:
            seen.add(key)
            print(f"- {r.get('platform')} · mediapipe {r.get('mediapipe')}")

    print("\n**Flags**\n")
    for line in verdict(rows):
        print(line)

    if args.verbose:
        print("\n**FPS per 30 s bin**\n")
        for r in rows:
            tag = ", ".join(f"{k}={v}" for k, v in sorted((r.get("tags") or {}).items())) or r["file"]
            print(f"- {tag}: {r.get('fps_bins')}")

    print(
        "\n> Jitter is in normalised image units (0-1), comparable only between runs of the same "
        "condition. Use a `condition=stillness` run for the headline jitter number.\n"
        "> Loop FPS from a laptop webcam does **not** transfer to a phone -- it bounds landmark "
        "quality, not on-device throughput. Phone FPS needs the Android measurement (M2, part 2).\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
