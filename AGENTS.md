# AGENTS.md

Guidance for automated agents working in this repository. Human-facing instructions live in
`README.md` and `docs/使用说明.md`; the research result lives in `docs/最终实验报告.md`.

## Scope & current state

- Windows/PowerShell repo: a LanPaint-based 256x256 image inpainting prototype **plus a completed
  inference-budget study**. App code in `app/` (`data`, `masks`, `features`, `budget`, `restore`,
  `cache`, `config`, `metrics`, `experiment`, `policy_freeze`, `ui`); entry scripts in `scripts/`;
  experiment specs in `configs/*.yaml`; offline analysis in `analysis/`; CI in `ci/` and
  `.github/workflows/cpu.yml`. The web UI is **native H5 + FastAPI**: `app/ui.py` (routes `GET /`,
  `POST /api/preview`, `POST /api/repair`, static `/runs`) serving `app/ui_static/{index.html,app.js,style.css}`;
  Gradio is no longer used.
- **S1–S6, the frozen rule, the full test scale and the write-up are done.** 540 real generations
  (dev 180 + main 270 + area 90 + seeds 2026/3407 90 each), 1080 policy records, 0 failures.
  Result: the adaptive rule M1 is **worse** than fixed K=5 on the 30-image test set
  (LPIPS 0.119657 vs 0.117012, +2.26%, paired 95% CI [-0.001336, 0.006592] crossing zero).
  The fixed K=5 default is kept; M1 is archived as an evaluated (negative) strategy, not a feature.
- Do not claim quality improvements that the report does not support; quote the paired statistics.

## Environment & prerequisites

- Real inference/experiments use the shared env
  `D:\Study\Postgraduate\GPT\envs\lanpaint\Scripts\python.exe` (Python 3.12.16, torch 2.11.0+cu128)
  and need an NVIDIA GPU. `start_ui.ps1` hardcodes this interpreter.
- The repo currently ships `requirements.yml` (conda, torch 2.14.1+cu126). **That is not the env the
  archived results were produced in**, and the three `requirements-lanpaint-*.txt` lock files were
  removed from the repo — so `frozen_policy.json`'s `source_sha256` no longer matches the tree.
  Treat this as an open issue, not as a working configuration (see Open issues).
- Offline checks need only CPU: `ci/requirements.txt` (torch CPU wheels, fastapi/uvicorn/httpx, test deps).
- Not vendored (gitignored): `checkpoints/256x256_diffusion_uncond.pt` and
  `third_party/LanPaintBench/`. Restore hash/commit from `docs/上游溯源.md` before touching
  inference. Never modify upstream and never import its `run.py`.

## Common commands (from repo root)

```powershell
$py = 'D:\Study\Postgraduate\GPT\envs\lanpaint\Scripts\python.exe'
& $py -m pytest tests -q                                            # 133 passed (offline)
& $py -m ruff check app scripts tests analysis ci                   # must stay clean
& $py ci\run_cpu_tests.py                                           # offline CPU gate, blocks network
& $py tools\check_lanpaint_env.py                                   # env/CUDA/ffmpeg audit
.\start_ui.ps1                                                      # H5 UI on 127.0.0.1:7860 (LANPAINT_PORT overrides)
& $py scripts\check_ui.py                                           # real GPU check over the HTTP API
& $py scripts\smoke_inference.py --runs 1                           # real inference, NFE/VRAM evidence
& $py scripts\run_experiment.py --config configs\dev.yaml --split dev --freeze-development --warmup
& $py analysis\diagnose_budget.py --cases-per-side 5                # needs local originals + run PNGs
```

Coverage (optional, kept out of the shared env): see `docs/覆盖率报告.md`.

## Architecture & conventions

- Tensors: images `(1,3,256,256)` float32 in `[-1,1]`; masks `(1,1,256,256)` `{0,1}` with **1 = missing**.
  `app.data.mask_to_upstream` inverts for LanPaintBench (upstream 1 = keep).
- Budget: tiers K=2/5/8, T=20; batch allocation gives the lowest/highest `floor(N/3)` of sorted
  scores the low/high tier so total NFE equals all-middle. Policies: B1/B2/B3 fixed, M1 (area+boundary,
  frozen weights `(1,-1)`), A1 (area only), C1 (M1 tiers shuffled). NFE is measured by wrapping the UNet
  forward, never derived from a formula.
- `app.cache` keys cover code provenance, checkpoint hash, image/mask hashes, seed, precision, K, outer
  steps and all sampling params. `app.experiment` runs each sample at all three tiers once and reuses
  the outputs across the six policies; `comparison_complete` and `equal_nfe` in `report.json` decide
  whether a run is a strict equal-budget comparison.
- **Frozen-source rule:** the fingerprint in `app.cache.source_provenance` covers `app/**`, `scripts/**`,
  `third_party/LanPaintBench/**` and `requirements*.txt`. Changing any of them invalidates
  `frozen_policy.json` and every cache entry. Put new analysis in `analysis/` (not fingerprinted);
  if inference code must change, register a new version and re-freeze — never patch hashes by hand.
- Data: `data/manifest.csv` (300 rows = 50 images x 3 mask types x 2 areas), `data/masks/` and
  `data/samples/original/` (50 PNGs) are tracked. `data/_cache/`, `checkpoints/`,
  `results/raw/**/images/`, `results/ui/`, `.cache/` stay untracked.

## Gotchas

- UI input (`app/ui_static/app.js` + `app.ui.prepare_input`): the H5 page keeps the untouched source
  canvas and a separate mask canvas; empty mask -> return original; all-missing mask rejected. GPU work
  is serialized; runs land in `results/ui/<ts>_<id>/` and are served read-only under `/runs/`.
- `.gitignore` currently ignores `/results` and `/results/*`; existing evidence is tracked, but **new**
  files under `results/` will be silently untracked. Force-add (`git add -f`) anything that is evidence.
- `results/raw/s1_smoke/report.json` currently holds a 1-run record, not the original 10-run D2 gate.
- The `paper/` PDF (~15 MB) is a document attachment, not a code artifact.
- An analysis directory can only be regenerated on a machine that still has the originals and the
  per-run `images/`; a clean clone can run tests, CI and the UI but not `analysis/diagnose_budget.py`.

## Open issues (do not "fix" silently)

1. `requirements-lanpaint-{direct,lock,shared-base}.txt` are missing, so the frozen source fingerprint
   (`b2f9f3ad…`) does not match `frozen_policy.json` (`a5921acb…`); any `--frozen-policy` run fails.
2. `.gitignore` ignores all of `/results`, which can hide new experiment artifacts.
3. `results/raw/s1_smoke/report.json` was overwritten with a 1-run record (sample/mask pair mismatched).
4. `paper/` PDF inflates the repo (~15 MB); consider LFS or an external link.

## Docs index

| Topic | File |
|---|---|
| Result, cases, limits, PRD acceptance | `docs/最终实验报告.md` |
| How to run UI / experiments / analysis / troubleshooting | `docs/使用说明.md` |
| Clean-clone reproduction record | `docs/复现记录.md` |
| Test coverage (83%) and gaps | `docs/覆盖率报告.md` |
| Collaboration status & upload log | `docs/项目进度.md`, `docs/下一步计划.md`, `docs/队友改进意见.md` |
| Upstream commit, weight URL and SHA-256 | `docs/上游溯源.md` |
