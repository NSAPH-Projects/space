# Benchmarks

## Getting Started

Install the conda environment. All commands must be run from the root of the repository.

```bash
conda env create -f benchmarks/conda.yaml
conda activate benchmarks
```

Running experiments on a laptop might take a few days. However, it is possible with the following command assuming 10 processes:

```bash
PYTHONPATH=. snakemake --configfile benchmarks/conf/pipeline.yaml --use-conda -j=1 
```

Here, `j=1` indicates to train one algo at a time, while `concurrency=4` indicates to use 8 processes to train each algo. The code uses `ray[tune]` to launch async experiments on each process. See `conf/config.yaml` for more details. Concurrency controls the number of simultaneous experiments.


We run experiments on a slurm cluster. First, we need a cluster configuration file. See `benchmarks/conf/cluster.yaml`. Then to submit the job, see the example in `benchmarks/slurm/job.sh`, which essentially uses the following command:


```bash
opts="--use-conda  --configfile benchmarks/conf/pipeline.yaml -C concurrency=10"
slurm_opts="/usr/bin/sbatch --ntasks {cluster.ntasks} -N {cluster.N} -t {cluster.t} \
    --cpus-per-task {cluster.cpus_per_task} -p {cluster.p} --mem {cluster.mem} -o {cluster.output} \
    -e {cluster.error} --mail-type={cluster.mail_type}"
PYTHONPATH=. snakemake $opts --cluster "${slurm_opts}" --cluster-config benchmarks/conf/cluster.yaml  -j 12
```
