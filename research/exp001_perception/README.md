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

## Calibrate first — `calibrate.py`

A live overlay view: hand skeletons, shoulder/arm lines, the mouth and eyebrow probes, and a HUD
with hand/face presence, inference time, a **distance proxy** and a brightness reading, plus
warnings when a hand leaves frame, the subject is too far, it is too dark, or the face is not
detected.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py research\exp001_perception\calibrate.py                        # webcam
& $py research\exp001_perception\calibrate.py --video clip.mp4        # review a phone clip
& $py research\exp001_perception\calibrate.py --video clip.mp4 --headless   # no window, prints readings
```

Keys: `q` quit · `SPACE` print readings · `r` reset windows · `o` toggle overlay.

**It is not a measurement tool, and it says so in its own title bar.** Drawing costs frames, so the
FPS it shows is "landmarker + rendering". Measurements come from `bench_holistic.py`, which stays
headless for exactly that reason.

### The distance proxy is the point

The HUD shows **shoulder width in normalised units** (pose landmarks 11↔12). Note the value you use
for each distance tag — "about two metres" is not reproducible between sessions, a shoulder width of
`0.22` is. Without this, the `distance=` tags in the sweep below are decoration.

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

## Browser options, for a sanity check with zero setup

Useful to see what the model *can* do before trusting our own plumbing — and to let someone else
look at it without installing anything.

| Where | What it gives |
|---|---|
| [MediaPipe Studio](https://ai.google.dev/edge/mediapipe/solutions/studio) | Google's playground: webcam or uploaded files, per-task settings to tweak, runs in the browser |
| [mediapipe-samples-web](https://google-ai-edge.github.io/mediapipe-samples-web/) | Live Tasks demos including **Holistic Landmarker** — face, hands and pose together, the exact configuration we use |
| [source repo](https://github.com/google-ai-edge/mediapipe-samples-web) | The same demos to run locally if we ever want a browser-based review tool |

What they are **not** good for: numbers. Browser WASM/WebGL throughput is a third thing, different
from both our Python path and an Android app. Use them to check behaviour, never to produce a figure
for the vault.

## If the only phone you have is an iPhone

| Use | Verdict |
|---|---|
| **iPhone as a camera** → `bench_holistic.py --source file` | **Do this.** Real phone optics, no Mac, no app. First set iOS Camera → Formats → **Most Compatible** (H.264), or OpenCV may fail on HEVC — otherwise transcode with ffmpeg. Gives landmark **quality** |
| **iPhone Safari** + MediaPipe web | ⚠ Google's own web examples sit at **~6–7 FPS on iPhone 11 / 12 Pro Max / 13 Pro** while running well on Android ([mediapipe#3303](https://github.com/google/mediapipe/issues/3303)). Also: iOS Safari only grants camera access over **HTTPS or localhost**, so a plain-HTTP LAN page silently has no camera |
| **Native iOS app** | Needs Xcode, which needs macOS. Blocked without a Mac |

> ⚠ **A ~6 FPS Safari number does not refute ADR-002.** It measures Safari's WASM runtime. No
> browser or emulator figure may accept or reject an architecture decision.

For performance figures the answer is a **cheap or borrowed mid-tier Android phone**. That is not a
compromise: Kazakhstan's market is Android-dominant, so Android is the product target regardless of
which phone the developer owns.

## Android emulator — read this before relying on it

The emulator **can** take the PC webcam as the device camera (AVD Manager → Advanced → front/back
camera → `webcam0`; DirectShow on Windows). That is genuinely useful later for app-level work:
wiring CameraX, permissions, UI, the frame-source interface.

**It is useless for the numbers we actually need.** Emulator FPS, latency, thermal throttling and
battery are properties of the host PC and the virtualisation layer, not of a phone — and M2 part 2
exists specifically to measure those on real hardware. Reported webcam-passthrough quirks
(resolution changes, long-run crashes, the camera being grabbed by another app) make it worse still
as a measurement surface.

So: emulator for plumbing, **real mid-tier phone for every performance figure**. The vault's
reporting rule — latency is always quoted with the hardware it was measured on — applies here.

## Notes that will save time

- The script records the **actual** delivered frame rate, not the requested one. They differ, and the
  difference propagates silently into every temporal model downstream.
- Run metadata logs the MediaPipe version and model bundle. Landmark indices change across versions,
  and a silently shifted index looks exactly like bad accuracy.
- `runs/` and `models/` are gitignored. Raw runs stay local; only the summary table goes in the vault.
- Never commit recorded video. Participant footage is covered by the vault's consent rules.
