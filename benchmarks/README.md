# Benchmarks

## Getting Started

Install the conda environment. All commands must be run from the root of the repository.

```bash
conda env create -f benchmarks/conda.yaml
conda activate benchmarks
```

Running experiments on a laptop might take a few days. However, it is possible with the following command assuming 10 processes:

```bash
PYTHONPATH=. snakemake --configfile benchmarks/conf/pipeline.yaml -C concurrency=1 cpus_per_task=1 --use-conda -j=10
```

Here, `j=10` indicates running 10 tasks simultaneously, `cpus_per_task=1` means to assign one cpu to each of these tasks, `concurrency=1` to use 1 async process for parallel hyperparameter tuning within each task, when hyperparameter tuning is required. Adjust these values as needed.

 The code uses `ray[tune]` to launch async experiments on each process. See `conf/config.yaml` for more details. Concurrency controls the number of simultaneous experiments.


We run experiments on a slurm cluster. First, we need a cluster configuration file. See `benchmarks/conf/cluster.yaml`. Then to submit the job, see the example in `benchmarks/slurm/job.sh`, which essentially uses the following command:


```bash
opts="--use-conda  --configfile benchmarks/conf/pipeline.yaml -C concurrency=10  cpus_per_task=2"
slurm_opts="/usr/bin/sbatch --ntasks {cluster.ntasks} -N {cluster.N} -t {cluster.t} \
    --cpus-per-task {cluster.cpus_per_task} -p {cluster.p} --mem {cluster.mem} -o {cluster.output} \
    -e {cluster.error} --mail-type={cluster.mail_type}"
PYTHONPATH=. snakemake $opts --cluster "${slurm_opts}" --cluster-config benchmarks/conf/cluster.yaml  -j 100
```
