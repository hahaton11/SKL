# CLAUDE.md — SKL (KRSL recognition)

This repo is the **code** for the KRSL sign-language recognition project. The documentation,
decisions and research live in a separate vault, and the vault is the source of truth:

```text
Vault:  D:\OneDrive\ksl_translator_repository\ksl_translator
Charter: D:\OneDrive\Desktop\promtff.txt
```

**Read first, in this order:**

1. `D:\OneDrive\ksl_translator_repository\CLAUDE.md` — full session protocol
2. `D:\OneDrive\ksl_translator_repository\ksl_translator\NEXT_SESSION.md` — entry point

Resume codeword: `RESUME_KSL_VAULT`

## Phase 0 — measurement only

No application code until `ADR-002 — Initial Recognition Architecture` is ACCEPTED. Measurement
harnesses are not application code; a camera UI is. Current experiment: `research/exp001_perception/`
(EXP-001 Baseline 0) — it accepts or kills ADR-002.

## Rules that bind this repo

- `research/` may use research-licensed assets. **`product/` may not** — that split is a licence
  boundary (AGPL Ultralytics, C-UDA WLASL, unlicensed Uni-Sign), enforced by layout, not memory.
- Commit identity is per-repo: `Abylaikhan <oxineman@gmail.com>`. Never set it globally.
- **No Anthropic/Claude attribution or co-author footers in commit messages.**
- Weights, `runs/`, `.venv/` and any video are gitignored. **Never commit participant footage.**
- Pushing to `origin` is pre-authorised (user, 2026-10-05).
- Any measurement is reported with the hardware it ran on. Laptop and browser figures are never
  product figures; only a real mid-tier Android phone may produce one (ISS-011).

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe research\exp001_perception\bench_holistic.py --fetch-model
```

Python 3.12.10, mediapipe 0.10.21, opencv 4.11.0, Windows 10 — the tested configuration.
