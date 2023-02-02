# Example DDP model training


## Installation on NeSI

Create a conda environment and install dependencies

```
module purge
module load Miniconda3/22.11.1-1
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1
conda env create -f environment.lock.yml -p ./venv
```

*Note: The `environment.lock.yml` file has been generated from a conda environment created with the `environment.yml` file and then exported with*

```
conda env export -p ./venv --no-builds | sed '/^name: .*/d; /^prefix: .*/d' > environment.lock.yml
```


## Getting started

Run the example via slurm using

```
sbatch --account=ACCOUNT train.sl
```

where `ACCOUNT` is your NeSI account.

The log files are saved in the `logs/` folder.
