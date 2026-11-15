# ade2e-wm: Hybrid Global–Object JEPA for compositional generalization in driving

Research code. Problem: compositional generalization (CG) of camera-based end-to-end planners — driving well in a
combination of situations never seen together in training, although each part was seen. Method: HGO v3
([docs/METHOD.md](docs/METHOD.md), rationale [docs/METHOD_RATIONALE.md](docs/METHOD_RATIONALE.md)); experiment design
[docs/RESEARCH_DESIGN.md](docs/RESEARCH_DESIGN.md); CG split, protocol and all CG results on NAVSIM
[docs/CG_SPLIT.md](docs/CG_SPLIT.md). **Current status and next steps: [docs/STATUS.md](docs/STATUS.md).**

Status (15/11/2026): CG split v1 on NAVSIM built; DrivoR withheld / exposed pilot showed no exposure gap, and an
exposure-dose check showed the measurement is not sensitive, so the pilot does not answer the question. Next: validate
the evaluation protocol with cheap checks (reference planners, kNN positive control) before more training; CARLA
closed-loop for the main CG evidence.

## Layout

| Path | Contents |
|---|---|
| `src/ade2e_wm/hgo/`, `scripts/hgo/`, `tests/hgo/` | HGO v3 (data, model, losses, PDM labels, training) and the CG tooling (labels, split, DrivoR pilot, gap statistics) |
| `src/ade2e_wm/agents/drivor/` | DrivoR's agent vendored by `scripts/phase0/vendor_drivor.py` |
| `configs/navsim/`, `configs/navsim_cg/` | Hydra configs for the official NAVSIM v2.2 devkit (via `hydra.searchpath`); CG scene filters |
| `artifacts/cg_lists/` | CG split v1 (`split_v1/`: sessions, pool, folds, manifests, test groups) and the per-frame situation labels (`events_*.parquet`) |
| `artifacts/cg_eval/` | per-scene scores (CSV) and gap / dose results (JSON) of step 0, the DrivoR pilot and the dose check |
| `scripts/phase0/` | data download, environment, DrivoR reproduction jobs; `env.sh` sets all paths |
| `scripts/common/` | generic Slurm CPU / test jobs |
| `docs/figures/cg_samples/` | sample images of the CG groups (3 cameras + top view from GT) |
| not in git | `data/` (NAVSIM / OpenScene, nuPlan maps), `exp/` (`NAVSIM_EXP_ROOT`: metric caches, runs), `envs/` (freeze in `envs/navsim2_freeze.txt`), `checkpoints/` (`SHA256SUMS`), `logs/`, `third_party/` |
| `third_party/` (clone) | DrivoR (valeoai/DrivoR `fc6e5aa`), NAVSIM (autonomousvision/navsim `359c7f7`), py123d_garage (kesai-labs `ad6cb51`), WA-JEPA (AFARI-Research `bec2966`); unmodified |

## Setup on a new machine

```bash
git clone https://github.com/nhat120904/ADE2E-WM.git ade2e-wm && cd ade2e-wm
mkdir -p third_party && cd third_party
git clone https://github.com/valeoai/DrivoR && git -C DrivoR checkout fc6e5aa
git clone https://github.com/autonomousvision/navsim navsim_v2 && git -C navsim_v2 checkout 359c7f7
cd ..
# edit scripts/phase0/env.sh (ADE2E_ROOT, data / exp roots), then build the env (scripts/phase0/build_env_navsim2.sbatch)
mkdir -p data/_lists && cp -r artifacts/cg_lists data/_lists/cg     # scripts read data/_lists/cg/...
# checkpoints/: released DrivoR (drivor/drivor_Nav2_10epochs.pth), DINOv2 ViT-S/14-reg4, pilot checkpoints (cg_pilot/)
```

NAVSIM evaluation also needs the OpenScene sensor data, nuPlan maps and the metric caches (`exp/metric_cache/`), which
are not in git: rebuild them with `scripts/phase0/metric_cache.sbatch` and `scripts/hgo/cg_cache.sbatch`.

## CG pipeline on NAVSIM (docs/CG_SPLIT.md)

All compute through Slurm on compute nodes (old cluster: ≤ 16 CPUs per job, ≤ 2 jobs, nothing on the login node).

```bash
sbatch scripts/hgo/cg_events.sbatch                     # situation labels of every frame -> data/_lists/cg/events_*.parquet
python scripts/hgo/cg_split.py                          # sessions, pool, folds, test groups -> data/_lists/cg/split_v1/
sbatch scripts/hgo/cg_cache.sbatch                      # metric cache / BEV targets of the added frames
python scripts/hgo/cg_drivor_prep.py                    # cached + balanced manifests, eval subset (inside a job)
sbatch --export=ALL,VARIANT=withheld scripts/hgo/cg_drivor_train.sbatch   # and VARIANT=exposed (~19 h GPU each)
sbatch --export=ALL,CKPT=<abs last.ckpt>,NAME=<run> scripts/hgo/cg_drivor_eval.sbatch
python scripts/hgo/cg_gap.py --fold data/_lists/cg/split_v1/turn_vru --exposed <csvs> --withheld <csvs> --out <json>
python scripts/hgo/cg_dose.py prep; sbatch ... scripts/hgo/cg_dose_eval.sbatch; python scripts/hgo/cg_dose.py gap ...
```

Data are under the nuScenes / nuPlan Terms of Use (non-commercial); `artifacts/cg_lists` holds tokens and labels derived
from the nuPlan annotations.
