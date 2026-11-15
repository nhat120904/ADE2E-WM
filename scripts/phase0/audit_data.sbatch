#!/bin/bash
#SBATCH --job-name=p0-audit-data
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Phase 0 data audits: OpenScene temporal/track/navtrain-coverage and kesai nuPlan-123D schema.
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
python $ADE2E_ROOT/scripts/phase0/audit_openscene.py --data-root $OPENSCENE_DATA_ROOT \
    --navsim-root $NAVSIM_DEVKIT_ROOT --out $ADE2E_ROOT/docs/phase0/audit_openscene.json \
    --max-logs 64 --workers $SLURM_CPUS_PER_TASK
python $ADE2E_ROOT/scripts/phase0/audit_kesai_nuplan.py --root $ADE2E_ROOT/data/kesai_nuplan \
    --out $ADE2E_ROOT/docs/phase0/audit_kesai_nuplan.json
