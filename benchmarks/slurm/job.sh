#!/bin/bash

#SBATCH -n 1
#SBATCH -t 0-16:00:00
#SBATCH -p shared
#SBATCH --mem=1G
#SBATCH -o slurm/slurm.%N.%j.out
#SBATCH -e slurm/slurm.%N.%j.err
#SBATCH --mail-type=ALL

slurm_options="/usr/bin/sbatch --ntasks {cluster.ntasks} -N {cluster.N} -t {cluster.t} \
    --cpus-per-task {cluster.cpus_per_task} -p {cluster.p} --mem {cluster.mem} -o {cluster.output} \
    -e {cluster.error} --mail-type={cluster.mail_type}"

options="--nolock --rerun-incomplete --use-conda --configfile benchmarks/conf/pipeline.yaml -C concurrency=10 overwrite=true"
export PYTHONPATH=.
snakemake $options --cluster "${slurm_options}" --cluster-config benchmarks/conf/cluster.yaml -j 87
