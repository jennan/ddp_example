# Example DDP model training


## Installation on NeSI

Create a conda environment and install dependencies

```
module purge
module load Miniconda3/22.11.1-1
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda env create -f environment.yml -p ./venv
```


## Getting started

Run the example via slurm using

```
sbatch --account=ACCOUNT train.sl
```

where `ACCOUNT` is your NeSI account.

The log files are saved in the `logs/` folder.
