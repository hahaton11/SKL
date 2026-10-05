# product/ — the shipping path

**Empty on purpose.** Nothing ships until ADR-002 is ACCEPTED on measured evidence.

## The rule this directory exists to enforce

> Research-only assets may appear in `research/`. They may **never** appear here.

This is a licence boundary, not a style preference. Concretely forbidden in this tree:

| Asset | Licence | Why it cannot ship |
|---|---|---|
| Ultralytics YOLO (`ultralytics`) | **AGPL-3.0** | Obliges source release for anything that conveys the work. A closed mobile app cannot comply |
| WLASL data or models derived from it | **C-UDA 1.0** | "Academic and computational use only", "No commercial usage is allowed" |
| Uni-Sign code or weights | **no licence file** | No licence means all rights reserved |
| Slovo | custom "variant of CC-BY-SA 4.0" | **Unresolved** — commercial use and ShareAlike inheritance by trained weights are both open questions |

Allowed so far: MediaPipe (Apache-2.0), MMPose/RTMPose code (Apache-2.0 — audit each checkpoint's
weights separately).

Three of the four forbidden items are what the local published literature is built on, which is
exactly why the boundary is physical rather than remembered. When a build is added here, it must
fail if a forbidden dependency is reachable — a licence violation is found by a lawyer, not a test,
so the test has to exist first.

Full audit: the vault's `01_Research/Datasets.md` and `01_Research/Existing Models.md`.
