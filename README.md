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


## Notes

- Torchrun https://pytorch.org/docs/stable/elastic/run.html
- Basics https://pytorch.org/docs/stable/distributed.html#basics
    - use torch.nn.parallel.DistributedDataParallel()
    - launch start 1 process per  GPU (use torchrun for that)
    - use NCCL backend
    -  :warning: check NCCL is using ib and not eth (or set NCCL_SOCKET_IFNAME to force it)
        - debug using export NCCL_DEBUG=INFO
- rendez-vous address using SLURMD_NODENAME or

```
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
```

  from https://gist.github.com/TengdaHan/1dd10d335c7ca6f13810fff41e809904?permalink_comment_id=3751671#gistcomment-3751671 (edited) 
