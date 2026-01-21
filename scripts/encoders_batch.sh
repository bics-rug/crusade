#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:01:00
#SBATCH --mem=8000
#SBATCH --array=0-1

module purge
module load Python/3.9.6-GCCcore-11.2.0
source $HOME/rfcx_snn/.venv/bin/activate

# modes=("async" "edge" "constantine" "adaptive" "freq")
modes=("freq")

python encoders_study.py --mode ${modes[$SLURM_ARRAY_TASK_ID]} --optimize --n_trials 500 --storage sqlite:///sigma_delta_study_new.db