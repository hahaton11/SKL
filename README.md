# SKL — KRSL sign language recognition

Real-time Kazakh-Russian Sign Language (KRSL) → text. Mobile app first, wearable later.

**Phase 0 — research and baseline selection. There is no application code here, by design.**
What lives here now is measurement: harnesses that produce numbers which accept or reject the
proposed architecture.

Documentation, decisions and research are **not** in this repo. They live in the project vault:

```text
D:\OneDrive\ksl_translator_repository\ksl_translator
```

Read `NEXT_SESSION.md` there first. The architecture under test is
`05_Decisions/Architecture Decision Records/ADR-002 — Initial Recognition Architecture.md`
(status: PROPOSED).

## Layout

```text
research/     experiments and measurement. Research-licensed assets are allowed here.
product/      anything shipped. Research-only assets are FORBIDDEN here. See product/README.md.
```

That split is a licence boundary, not a style preference — see the rule in `product/README.md`.

## Current experiment

| Id | What | Decides |
|---|---|---|
| **EXP-001** | `research/exp001_perception/` — MediaPipe Holistic Landmarker perception quality, latency, stability | Whether a keypoint pipeline is viable at all. **Can kill ADR-002.** |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe research\exp001_perception\bench_holistic.py --fetch-model
```

Python 3.12 on Windows is the tested configuration.

## Why measurement before code

Everything downstream assumes landmarks are reliable at natural signing speed. That assumption is
unverified on our hardware, and Uni-Sign's central finding is that keypoint inaccuracy is a
first-class problem. One week of measurement here is cheaper than three months of modelling on a
broken premise.
