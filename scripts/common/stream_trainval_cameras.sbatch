#!/bin/bash
#SBATCH --job-name=stream-trainval-cams
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Stream the OpenScene trainval camera archives not yet on disk (160 of 200 parts, user request 2026-11-12: all 1,310
# logs) straight into /mnt/data, keeping only the 4 cameras the models use (CAM_F0, CAM_L0, CAM_R0, CAM_B0); nothing
# else touches disk. Resumable: a part is skipped when its done marker exists (written only if wget and tar both exit 0).
set -uo pipefail
S=/mnt/data/nhatnc129/ade2e-wm/_stream
DEST=/mnt/data/nhatnc129/ade2e-wm/data/sensor_blobs/trainval
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_trainval_camera
mkdir -p $DEST $S/done $S/err
one() {
  k=$1; P=openscene-v1.1/sensor_blobs/trainval
  [ -f $S/done/$k ] && { echo "part $k: already done"; return; }
  wget -q --tries=5 --timeout=60 -O - $HF/openscene_sensor_trainval_camera_$k.tgz \
    | tar -xzf - --strip-components=3 -C $DEST --wildcards \
        "$P/*/CAM_F0/*" "$P/*/CAM_L0/*" "$P/*/CAM_R0/*" "$P/*/CAM_B0/*" 2> $S/err/part$k.err
  st=("${PIPESTATUS[@]}")
  [ "${st[0]}" = 0 ] && [ "${st[1]}" = 0 ] && touch $S/done/$k
  echo "part $k: wget|tar exit ${st[*]} $(date +%T)"
}
export -f one; export S DEST HF
hostname; date; df -h $DEST
grep -v '^$' $S/parts_todo.txt | xargs -P 4 -I{} bash -c 'one {}'
echo "parts done: $(ls $S/done | wc -l) / $(grep -cv '^$' $S/parts_todo.txt)"
echo "log folders in DEST: $(ls $DEST | wc -l)"; df -h $DEST; date
