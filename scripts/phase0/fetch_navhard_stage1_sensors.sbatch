#!/bin/bash
#SBATCH --job-name=p0-navhard-s1
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# navhard_two_stage stage-one (real) scenes read cameras from sensor_blobs/test, which the navhard
# tarballs do not contain. Stream the needed OpenScene test camera splits and extract only the
# 4 history frames x 8 cameras of the 450 stage-one tokens (lists in data/_lists/).
set -uo pipefail
ROOT=/home/nhatnc129/nhat.nc/ade2e-wm/data
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera
mkdir -p $ROOT/sensor_blobs/test
one() {
  k=$1; list=$ROOT/_lists/navhard_stage1_test_split$k.txt
  wget -q --tries=5 --timeout=60 -O - $HF/openscene_sensor_test_camera_$k.tgz \
    | tar -xzf - --strip-components=3 -C $ROOT/sensor_blobs/test -T $list 2> $ROOT/_lists/tar_split$k.err
  echo "split $k: expected $(wc -l < $list) extracted $(sed 's|openscene-v1.1/sensor_blobs/test/||' $list | (cd $ROOT/sensor_blobs/test && xargs ls 2>/dev/null | wc -l))"
}
export -f one; export ROOT HF
ls $ROOT/_lists/navhard_stage1_test_split*.txt | sed -E 's/.*split([0-9]+)\.txt/\1/' | xargs -P 4 -I{} bash -c 'one {}'
echo "total jpg: $(find $ROOT/sensor_blobs/test -name '*.jpg' | wc -l)"; du -sh $ROOT/sensor_blobs/test
