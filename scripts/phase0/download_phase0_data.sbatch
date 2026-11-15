#!/bin/bash
#SBATCH --job-name=p0-download
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Phase 0 data download (audit-sized subset only; ~53 GB).
# Licenses: nuPlan / OpenScene data are CC BY-NC-SA 4.0 under the nuPlan terms of use
# (https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE); non-commercial research only.
# Each step writes a .done marker so the job can be resubmitted safely.
set -euo pipefail
ROOT=/home/nhatnc129/nhat.nc/ade2e-wm/data
HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main
mkdir -p "$ROOT"/{navsim_logs,sensor_blobs,_dl,kesai_nuplan}
cd "$ROOT/_dl"

step() { [ -f "$ROOT/.done_$1" ]; }
mark() { touch "$ROOT/.done_$1"; echo "[$(date +%T)] done: $1"; }
fetch() { wget -q -c --tries=5 --timeout=60 "$1" -O "$2"; }

# 1. nuPlan maps (needed by metric caching and PDM scoring).
# The official bucket (AWS ap-northeast-1) is unreachable from this cluster (verified 2026-10-28),
# so fall back to the Hugging Face mirror pengxiang/nuplan_maps and check its LFS sha256.
MAPS_SHA256=444860429f9a3bcf89a6459d683fa82eb9219aa259d8d9b8cdefdb37f0b56b05
if ! step maps; then
  if timeout 60 wget -q --tries=1 --connect-timeout=20 \
      https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip -O maps.zip; then
    echo "maps source: official S3" | tee "$ROOT/MAPS_PROVENANCE.txt"
  else
    fetch https://huggingface.co/datasets/pengxiang/nuplan_maps/resolve/main/nuplan-maps-v1.1.zip maps.zip
    echo "maps source: HF mirror pengxiang/nuplan_maps (official S3 unreachable)" | tee "$ROOT/MAPS_PROVENANCE.txt"
  fi
  sha256sum maps.zip | tee -a "$ROOT/MAPS_PROVENANCE.txt"
  echo "expected (mirror LFS oid): $MAPS_SHA256" >> "$ROOT/MAPS_PROVENANCE.txt"
  unzip -q -o maps.zip && rm maps.zip && rm -rf "$ROOT/maps" && mv nuplan-maps-v1.0 "$ROOT/maps"
  mark maps
fi

# 2. OpenScene metadata: test (navtest/navhard logs), mini, trainval (navtrain logs)
for split in test mini trainval; do
  if ! step meta_$split; then
    fetch $HF/openscene-v1.1/openscene_metadata_$split.tgz meta_$split.tgz
    tar -xzf meta_$split.tgz && rm meta_$split.tgz
    rm -rf "$ROOT/navsim_logs/$split" && mv openscene-v1.1/meta_datas/$split "$ROOT/navsim_logs/$split" && rm -rf openscene-v1.1
    mark meta_$split
  fi
done

# 3. One OpenScene mini camera split: full-rate (2 Hz) past+future frames for the temporal audit
if ! step mini_cam0; then
  fetch $HF/openscene-v1.1/openscene_sensor_mini_camera/openscene_sensor_mini_camera_0.tgz mini_cam0.tgz
  tar -xzf mini_cam0.tgz && rm mini_cam0.tgz
  src=openscene-v1.1/sensor_blobs/mini; echo "mini sensors extracted at $src"
  mkdir -p "$ROOT/sensor_blobs/mini" && cp -rl "$src"/. "$ROOT/sensor_blobs/mini/" && rm -rf ./openscene-v1.1 "$src"
  mark mini_cam0
fi

# 4. NAVSIM-v2 navhard_two_stage (real + synthetic sensors, scene pickles)
if ! step navhard; then
  for part in curr_sensors hist_sensors scene_pickles; do
    f=navsim_v2.2_navhard_two_stage_$part.tar.gz
    fetch $HF/navsim-v2/$f $f
    tar -xzvf $f -C "$ROOT" | sed -n "1,3p;\$p" && rm $f
  done
  mark navhard
fi

# 5. kesai-labs/nuplan (123D, 10 Hz): two train logs, front camera + boxes, for the track/timestamp audit
KH=https://huggingface.co/datasets/kesai-labs/nuplan/resolve/main
if ! step kesai; then
  fetch $KH/index.parquet "$ROOT/kesai_nuplan/index.parquet"
  fetch $KH/LICENSE "$ROOT/kesai_nuplan/LICENSE"
  for log in 2021.08.17.15.02.08_veh-45_02003_02086 2021.06.09.17.23.18_veh-38_00016_00120; do
    d="$ROOT/kesai_nuplan/logs/nuplan_train/$log"; mkdir -p "$d"
    for f in box_detections_se3.arrow ego_state_se3.arrow sync.arrow custom.scenario.arrow \
             camera.pcam_f0.arrow camera.pcam_l0.arrow camera.pcam_r0.arrow camera.pcam_b0.arrow; do
      fetch $KH/logs/nuplan_train/$log/$f "$d/$f" || echo "MISSING $log/$f"
    done
  done
  mark kesai
fi

du -sh "$ROOT"/* 2>/dev/null
echo "ALL DONE $(date)"
