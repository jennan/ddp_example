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
sbatch --account=ACCOUNT train_1node.sl
```

where `ACCOUNT` is your NeSI account.

The log files are saved in the `logs/` folder.


## References

- https://pytorch.org/docs/stable/distributed.html
- https://pytorch.org/docs/stable/elastic/run.html


## Notes

Rendez-vous address using SLURMD_NODENAME or

```
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
```

from https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904?permalink_comment_id=3751671#gistcomment-3751671 (edited) 
