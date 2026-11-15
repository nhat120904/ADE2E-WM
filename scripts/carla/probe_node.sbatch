#!/bin/bash
#SBATCH --job-name=carla-probe
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Can a compute node run CARLA 0.9.15 (Bench2Drive)? Read-only probe: GPU visibility, NVIDIA Vulkan / GL libraries, local
# scratch space, container tools, and whether the CARLA, HF, PyPI and Docker Hub endpoints answer (HEAD requests only).
set -uo pipefail
echo "== host $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv 2>&1 | head -12
echo "== /dev/nvidia*: $(ls /dev/nvidia* 2>/dev/null | tr '\n' ' ')"
echo "== vulkan icd files"; ls -la /usr/share/vulkan/icd.d/ /etc/vulkan/icd.d/ 2>&1
echo "== nvidia graphics libs"; ldconfig -p 2>/dev/null | grep -i -E "libvulkan|libGLX_nvidia|libEGL_nvidia|libnvidia-glcore|libnvidia-glvkspirv|libnvidia-eglcore" | head -12
echo "== vulkaninfo: $(which vulkaninfo 2>/dev/null || echo none)"
which vulkaninfo >/dev/null 2>&1 && timeout 60 vulkaninfo --summary 2>&1 | head -40
echo "== X / display libs"; ldconfig -p 2>/dev/null | grep -i -E "libX11.so|libXext|libSM.so|libICE" | head -5
echo "== containers: $(which enroot apptainer singularity docker 2>/dev/null | tr '\n' ' ')"
echo "== disk"; df -h /tmp /dev/shm /home/nhatnc129 2>/dev/null; echo "TMPDIR=${TMPDIR:-unset} SLURM_TMPDIR=${SLURM_TMPDIR:-unset}"
echo "== network (HEAD)"
for u in https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz \
         https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz \
         https://huggingface.co/api/datasets/rethinklab/Bench2Drive https://pypi.org/simple/carla/ \
         https://registry-1.docker.io/v2/ https://github.com; do
  out=$(curl -sSI -m 30 "$u" 2>&1 | grep -i -E "^HTTP|^content-length" | tr '\r\n' '  ')
  echo "$u -> rc=$? $out"
done
echo done
