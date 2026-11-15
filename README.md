# ade2e-wm: Hybrid Global–Object JEPA for compositional generalization in driving

Research code for the staged plan in the proposal (Phase 0 → 5). Status: Phase 0 (audit, DrivoR reproduction) and Phase 1 (diagnostic, frozen
DrivoR features, GT objects) are closed. Current method: HGO v3
([docs/METHOD.md](docs/METHOD.md), rationale [docs/METHOD_RATIONALE.md](docs/METHOD_RATIONALE.md)); experiments
[docs/RESEARCH_DESIGN.md](docs/RESEARCH_DESIGN.md); compositional split [docs/CG_SPLIT.md](docs/CG_SPLIT.md); current status [docs/STATUS.md](docs/STATUS.md). Superseded docs
(pilot H32, v2 design) are archived in `exp/archive/docs_2026-11-12/`.

## Layout

| Path | Contents |
|---|---|
| `third_party/` | unmodified upstream clones, not in git: DrivoR (valeoai/DrivoR `fc6e5aa`), NAVSIM (autonomousvision/navsim `359c7f7`), py123d_garage (kesai-labs `ad6cb51`), WA-JEPA (AFARI-Research `bec2966`) |
| `src/ade2e_wm/` | our code; `agents/drivor` is DrivoR's agent vendored by `scripts/phase0/vendor_drivor.py` |
| `configs/navsim/agent/` | Hydra agent configs for the official NAVSIM v2.2 devkit (added via `hydra.searchpath`) |
| `scripts/phase0/` | Slurm jobs and audit scripts for Phase 0; `env.sh` sets all paths |
| `src/ade2e_wm/hgo/`, `scripts/hgo/`, `tests/hgo/` | the method (docs/METHOD.md): data, model, losses, PDM labels, training, evaluation |
| `scripts/common/` | Slurm queue driver (≤ 2 jobs), generic command / test jobs |
| `checkpoints/` | released DrivoR checkpoints and DINOv2 ViT-S/14-reg4 weights (`SHA256SUMS`) |
| `data/` | NAVSIM/OpenScene data in the official layout (`OPENSCENE_DATA_ROOT`), nuPlan maps, kesai nuPlan-123D sample |
| `envs/navsim2/` | uv virtualenv (Python 3.9); freeze in `envs/navsim2_freeze.txt` |
| `exp/` | `NAVSIM_EXP_ROOT`: metric caches and evaluation outputs |
| `logs/slurm/` | job logs |

## Reproducing Phase 0

All compute runs through Slurm on compute nodes (QOS: ≤16 CPUs per job, ≤32 CPUs / 2 GPUs per user).

```bash
sbatch scripts/phase0/download_phase0_data.sbatch        # maps, metadata, mini split 0, navhard, 123D sample
sbatch scripts/phase0/build_env_navsim2.sbatch           # envs/navsim2
python3 scripts/phase0/vendor_drivor.py                  # src/ade2e_wm/agents/drivor
sbatch scripts/phase0/fetch_navhard_stage1_sensors.sbatch  # stage-one real frames (lists in data/_lists)
sbatch scripts/phase0/metric_cache_navhard.sbatch
sbatch scripts/phase0/inspect_drivor.sbatch              # checkpoint keys and tensor shapes
sbatch scripts/phase0/audit_data.sbatch                  # OpenScene / navtrain / 123D audits
sbatch scripts/phase0/gpu_latency_drivor.sbatch

# after accepting the data terms: 40-split trainval camera pilot + full test cameras (4 cams kept)
python3 scripts/phase0/select_trainval_splits.py
sbatch --export=ALL,KIND=trainval,SPLITS_FILE=$PWD/data/_lists/trainval_pilot_splits.txt scripts/phase0/stream_camera_splits.sbatch
sbatch --export=ALL,KIND=test,SPLITS_FILE=$PWD/data/_lists/test_all_splits.txt scripts/phase0/stream_camera_splits.sbatch
sbatch --export=ALL,SPLIT=navtest scripts/phase0/metric_cache.sbatch
sbatch scripts/phase0/verify_camera_data.sbatch          # coverage + leakage -> docs/phase0/verify_camera_data.json

# DrivoR baseline (official devkit code; run_pdm_balanced.py only schedules one Ray task per log)
sbatch scripts/phase0/check_balanced_runner.sbatch       # 2-log smoke run must match the plain devkit CSV
sbatch scripts/phase0/eval_drivor_navhard.sbatch         # LOGS_FILE=configs/subsets/<file> for a log subset
sbatch scripts/phase0/eval_drivor_navtest.sbatch
```

Data are under the nuScenes/nuPlan Terms of Use (non-commercial); see `data/OPENSCENE_LICENSE.txt`.
