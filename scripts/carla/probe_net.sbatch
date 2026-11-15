#!/bin/bash
#SBATCH --job-name=carla-net
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Workarounds for CARLA on compute nodes (HEAD / metadata requests only): the official CARLA Docker image on Docker Hub,
# and the NVIDIA 570.133.20 driver package whose user-space Vulkan / GL libraries the host lacks.
set -uo pipefail
curl -sS -m 30 "https://hub.docker.com/v2/repositories/carlasim/carla/tags/0.9.15" | head -c 600; echo
for u in https://us.download.nvidia.com/tesla/570.133.20/NVIDIA-Linux-x86_64-570.133.20.run \
         https://download.nvidia.com/XFree86/Linux-x86_64/570.133.20/NVIDIA-Linux-x86_64-570.133.20.run \
         https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.15.tar.gz; do
  out=$(curl -sSI -m 30 "$u" 2>&1 | grep -i -E "^HTTP|^content-length|curl:" | tr '\r\n' '  ')
  echo "$u -> $out"
done
echo "enroot: $(enroot version 2>&1)"; grep -v "^#" /etc/enroot/enroot.conf 2>/dev/null | grep -v "^\s*$" | head
