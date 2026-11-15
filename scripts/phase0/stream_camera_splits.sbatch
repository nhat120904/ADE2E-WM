#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Stream OpenScene camera splits and keep only the 4 cameras used by DrivoR and our variants
# (CAM_F0, CAM_L0, CAM_R0, CAM_B0); nothing else touches disk.
# Env: KIND=trainval|test, SPLITS_FILE=<one split index per line>, optional DEST (default data/sensor_blobs/$KIND).
set -uo pipefail
ROOT=/home/nhatnc129/nhat.nc/ade2e-wm/data
: "${KIND:?}" "${SPLITS_FILE:?}"
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_${KIND}_camera
DEST=${DEST:-$ROOT/sensor_blobs/$KIND}
mkdir -p $DEST $ROOT/_lists/stream_logs
one() {
  k=$1; P=openscene-v1.1/sensor_blobs/$KIND
  wget -q --tries=5 --timeout=60 -O - $HF/openscene_sensor_${KIND}_camera_$k.tgz \
    | tar -xzf - --strip-components=3 -C $DEST --wildcards \
        "$P/*/CAM_F0/*" "$P/*/CAM_L0/*" "$P/*/CAM_R0/*" "$P/*/CAM_B0/*" \
        2> $ROOT/_lists/stream_logs/${KIND}_split$k.err
  echo "$KIND split $k: wget|tar exit ${PIPESTATUS[*]} $(date +%T)"
}
export -f one; export ROOT HF KIND DEST
grep -v '^$' $SPLITS_FILE | xargs -P 4 -I{} bash -c 'one {}'
echo "jpg total: $(find $DEST -name '*.jpg' | wc -l)"; du -sh $DEST
