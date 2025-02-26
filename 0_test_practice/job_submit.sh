#!/bin/sh

## DO NOT EDIT ANY LINE BELOW ##
## THIS IS THE STANDARD TEMPLATE FOR SUBMITTING A JOB IN DGX BOX ##

## The line above is used by the system to process it as
## a bash script.

## The file is a template for running experiments in the ISL Lab.

## SLURM Configuration ##
#SBATCH --job-name=test ## Job name
#SBATCH --ntasks=1 ## Run on a single CPU
##SBATCH --gres=gpu:a100_1g.5gb:1 ## Example GPU request
#SBATCH --time=00:10:00 ## Time limit (hh:mm:ss)
#SBATCH --partition=shortq ## Partition name
#SBATCH --qos=shortq ## Queue name
#SBATCH --mem=16G ## Memory allocation

## Output Files ##
#SBATCH --output=/scratch/%u/%x-%N-%j.out ## Standard output
#SBATCH --error=/scratch/%u/%x-%N-%j.err ## Error output

## Module Loading ##
. /etc/profile.d/modules.sh

module load anaconda/2023.03-1

eval "$(conda shell.bash hook)"
conda activate pytorch_gpu

## Python Script Execution ##
python try.py

## Instructions ##
# Submit the script using: sbatch job_submit.sh
# Note the job ID for tracking.
# Check job status with: sacct -u
# After completion, output and error files are located in /scratch//.
# - The output file: --.out contains the program's output.
# - The error file: --.err contains error logs.
