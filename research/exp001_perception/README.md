# EXP-001 — Baseline 0: perception quality on target hardware

**Decides:** whether a keypoint pipeline is viable at all. **Can kill ADR-002.**
**Cost:** ~1 week. **Blocked by:** nothing.

## The question

Can MediaPipe Holistic Landmarker produce stable landmarks at **natural and fast signing speed**, at
a usable frame rate, without collapsing over a realistic session length?

Everything downstream assumes yes. Nobody has measured it on our hardware.

## What this measures, and what it does not

| Measured here | Transfers to the phone? |
|---|---|
| Landmark presence / dropout per channel | **Yes** — it is a property of the model and the footage |
| Longest no-hand gap | **Yes** |
| Jitter, incl. mouth and eyebrow probes | **Yes** |
| Degradation by distance / lighting / speed / occlusion | **Yes** |
| Inference latency, loop FPS, FPS decay | **No** — laptop numbers bound nothing on a phone |

So this covers M2 part 1. **Phone FPS, thermal behaviour and battery need the Android measurement
(M2 part 2)** — the official `holistic_landmarker` Android sample, instrumented. Do not quote a
laptop FPS number as a product metric; the vault's reporting rules forbid it.

A cheap half-step: record clips **on the phone**, analyse them here with `--source file`. That gives
real phone-camera footage (rolling shutter, compression, autofocus, actual frame rate) against the
same landmark model, without writing an app yet.

## Setup

```powershell
# from the repo root
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe research\exp001_perception\bench_holistic.py --fetch-model
```

The model bundle (**13 MB**, float16) lands in `models/` and is gitignored — we download weights, we
do not redistribute them.

Verified working on 2026-10-05: mediapipe 0.10.21, opencv 4.11.0, Python 3.12.10, Windows 10.
Inference on this machine measured **~18 ms p50 per frame at 640×480** on a synthetic clip with no
subject in frame — that is a plumbing figure, not a result; a real subject costs more.

## Protocol

Run each condition for **60 s** of continuous signing, plus one **10-minute** run at
`speed=natural` for the decay check. Tag every run; the tags are what `analyze.py` groups by.

```powershell
$py = ".\.venv\Scripts\python.exe"
$b  = "research\exp001_perception\bench_holistic.py"

# baseline
& $py $b --duration 60 --tag distance=1m --tag lighting=bright --tag speed=natural --tag background=plain

# distance sweep
& $py $b --duration 60 --tag distance=2m --tag lighting=bright --tag speed=natural
& $py $b --duration 60 --tag distance=3m --tag lighting=bright --tag speed=natural

# lighting sweep
& $py $b --duration 60 --tag distance=2m --tag lighting=dim    --tag speed=natural
& $py $b --duration 60 --tag distance=2m --tag lighting=backlit --tag speed=natural

# speed sweep -- the one that matters most
& $py $b --duration 60 --tag distance=2m --tag lighting=bright --tag speed=slow
& $py $b --duration 60 --tag distance=2m --tag lighting=bright --tag speed=fast

# occlusion
& $py $b --duration 60 --tag distance=2m --tag occlusion=hand-over-hand --tag speed=natural
& $py $b --duration 60 --tag distance=2m --tag occlusion=hand-over-face --tag speed=natural

# background
& $py $b --duration 60 --tag distance=2m --tag background=cluttered --tag speed=natural

# jitter reference: hold both hands still in frame, do not sign
& $py $b --duration 30 --tag condition=stillness --tag distance=2m

# decay check
& $py $b --duration 600 --tag distance=2m --tag speed=natural --tag run=10min

# then
& $py research\exp001_perception\analyze.py runs\*.jsonl --verbose
```

`analyze.py` prints a markdown table and a **Flags** section. Paste both into the vault's
`04_Experiments/Baselines/EXP-001 — Baseline 0 Perception.md`.

### Validating the analysis without a camera

```powershell
python research\exp001_perception\make_fixture.py runs\fixtures
python research\exp001_perception\analyze.py runs\fixtures\*.jsonl --verbose
```

Expected: no flags for `fixture=good`, dropout **and** swallowed-sign flags for `fixture=fast`,
a throttling flag for `fixture=throttled`. If that does not hold, the analysis is broken, not the
camera. Writing these fixtures caught two real bugs before any measurement was taken — PowerShell
not expanding globs for native commands, and a partial final time bin reading as a catastrophic
FPS drop (a false throttling flag on a flat run).

### Sign faster than feels natural, on purpose

The documented weak spot of keypoint methods is **fast and complex gestures** — the Applied Sciences
15(10) 5685 work built an optical-flow mitigation specifically for it. Demoing at a comfortable pace
hides exactly the failure we are looking for. The `speed=fast` run is the most informative in the
sweep.

## Pass / fail

**Pass:** usable hand presence and acceptable jitter at natural *and* fast speed, loop FPS ≥ 15 on
the reference machine, no collapse over 10 minutes.

**Fail → ADR-002 is superseded.** Pre-specified fallbacks, in order:

1. Hand + Pose landmarkers separately with a reduced face model
2. Lower perception rate with interpolation
3. RTMPose via ONNX (Apache-2.0 code; audit weight licences per checkpoint)
4. Add an RGB branch

The fallbacks are listed now so a bad result is a decision, not a crisis.

## Thresholds the Flags section checks

| Flag | Trigger |
|---|---|
| FPS below target | loop FPS < 15 |
| High dropout | any-hand presence < 90% |
| Throttling | FPS decayed > 20% from first to last bin |
| Swallowed signs | longest no-hand gap > 300 ms |

## Notes that will save time

- The script records the **actual** delivered frame rate, not the requested one. They differ, and the
  difference propagates silently into every temporal model downstream.
- Run metadata logs the MediaPipe version and model bundle. Landmark indices change across versions,
  and a silently shifted index looks exactly like bad accuracy.
- `runs/` and `models/` are gitignored. Raw runs stay local; only the summary table goes in the vault.
- Never commit recorded video. Participant footage is covered by the vault's consent rules.
