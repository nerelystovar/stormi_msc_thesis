#!/usr/bin/env bash
#SBATCH --job-name=stormi-stability-180
#SBATCH --partition=gpu-el8
#SBATCH --array=0-179%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --no-requeue

set -euo pipefail

WORKFLOW=/g/stegle/tovar/apps/stormi_msc_thesis/trajectory_stability_180_runs
PYTHON=/g/stegle/tovar/apps/stormi_cc_env/.venv/bin/python
RESULTS=${STORMI_RESULTS_ROOT:-/scratch/tovar/xenium_trajectory_stability_180_runs}
SEED=$((SLURM_ARRAY_TASK_ID % 5 + 1))

export PYTHONHASHSEED=${SEED}
export XLA_FLAGS=--xla_gpu_autotune_level=4
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPLBACKEND=Agg
export TMPDIR=/tmp/${USER}/stormi-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}
export NUMBA_CACHE_DIR=${TMPDIR}/numba
export MPLCONFIGDIR=${TMPDIR}/matplotlib
export JAX_COMPILATION_CACHE_DIR=${TMPDIR}/jax
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}" "${JAX_COMPILATION_CACHE_DIR}"

srun --unbuffered "${PYTHON}" -u "${WORKFLOW}/run_stormi.py" \
    "${SLURM_ARRAY_TASK_ID}" --output-root "${RESULTS}"
