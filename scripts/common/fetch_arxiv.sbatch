#!/bin/bash
#SBATCH --job-name=fetch-arxiv
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/home/nhatnc129/nhat.nc/ade2e-wm/logs/slurm/%x-%j.out
# Download arXiv papers (HTML + PDF) from a compute node (the login node cannot reach arxiv.org) for reading only.
# Each paper goes to its own new directory exp/papers/<id>/; the HTML is converted to text with the standard library.
# usage: sbatch --export=ALL,IDS="2611.02526 2606.16274" scripts/common/fetch_arxiv.sbatch
set -uo pipefail
ROOT=/home/nhatnc129/nhat.nc/ade2e-wm
UA="Mozilla/5.0 (X11; Linux x86_64) research-reading"
for ID in $IDS; do
  D=$ROOT/exp/papers/$ID
  mkdir -p $D
  curl -sSL -m 120 -A "$UA" -o $D/page.html "https://arxiv.org/html/$ID" ; echo "$ID html rc=$? $(wc -c < $D/page.html 2>/dev/null) bytes"
  curl -sSL -m 120 -A "$UA" -o $D/paper.pdf "https://arxiv.org/pdf/$ID" ; echo "$ID pdf rc=$? $(wc -c < $D/paper.pdf 2>/dev/null) bytes"
  curl -sSL -m 60 -A "$UA" -o $D/abs.html "https://arxiv.org/abs/$ID" ; echo "$ID abs rc=$?"
  if [ -s $D/page.html ]; then
    (cd /tmp && python3 -I $ROOT/scripts/common/html2text.py $D/page.html > $D/page.txt); echo "$ID text $(wc -l < $D/page.txt) lines"
  fi
  sleep 3
done
echo done
