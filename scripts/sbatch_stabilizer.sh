#!/usr/bin/env bash
#SBATCH -J inact_decoding
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=1G
#SBATCH -p common
#SBATCH -t 0-40:00:00
#SBATCH --array=0-199

WORKER_ID=$(( ${SLURM_ARRAY_TASK_ID:-0} * 1000000 ))
echo "task ${SLURM_ARRAY_TASK_ID:-0}, seed offset ${WORKER_ID}"
tic=$(date +%s)

python src/decoder.py --seed-offset "$WORKER_ID" "$@"

toc=$(date +%s)
echo "Elapsed time: $((toc - tic)) s"
