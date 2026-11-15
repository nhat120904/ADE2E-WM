# Source in every Phase 0 job: paths for the official NAVSIM v2.2 devkit + ade2e_wm.
export ADE2E_ROOT=/home/nhatnc129/nhat.nc/ade2e-wm
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT=$ADE2E_ROOT/data/maps
export NAVSIM_EXP_ROOT=$ADE2E_ROOT/exp
export NAVSIM_DEVKIT_ROOT=$ADE2E_ROOT/third_party/navsim_v2
export OPENSCENE_DATA_ROOT=$ADE2E_ROOT/data
export PYTHONPATH=$ADE2E_ROOT/src${PYTHONPATH:+:$PYTHONPATH}
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR=/tmp/ray_${SLURM_JOB_ID:-local}
source $ADE2E_ROOT/envs/navsim2/bin/activate
# extra hydra search path holding configs/navsim/agent/drivor_nav2.yaml
ADE2E_HYDRA_SEARCHPATH="hydra.searchpath=[pkg://navsim.planning.script.config.common,file://$ADE2E_ROOT/configs/navsim]"
