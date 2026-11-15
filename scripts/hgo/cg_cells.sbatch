#!/bin/bash
#SBATCH --job-name=cg-cells
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Candidate held-out cells of the compositional split (scripts/hgo/cg_cells.py, docs/CG_SPLIT.md). CPU, counts only.
set -euo pipefail
source /home/nhatnc129/nhat.nc/ade2e-wm/scripts/phase0/env.sh
cd $ADE2E_ROOT
export OMP_NUM_THREADS=4
python -c "import pandas as pd; e=pd.read_parquet('data/_lists/cg/events_trainval.parquet', columns=['inter_f','has_img','log']); print('map NaN frac', e.inter_f.isna().mean(), 'inter frac', round(e.inter_f.mean(), 3), 'logs with images', e[e.has_img].log.nunique())"
python scripts/hgo/cg_cells.py --pet ${PET:-3} --tconf ${TCONF:-6}
