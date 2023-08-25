# This file is used to generate all the baselines for the paper.

from omegaconf import OmegaConf

conda: "benchmarks/conda.yaml"


# == Load configs ==
if len(config) == 0:
    raise Exception(
        "No config file passed to snakemake."
        " Use flag --configfile benchmarks/conf/pipeline.yaml"
    )


# make target files
targets = []
for t_type in ("disc", "cont"):
    envs = config["spaceenvs"][t_type]
    algos = config["algorithms"][t_type]
    for env in envs:
        for algo in algos:
            logfile = f"{config['logdir']}/{env}/{algo}.jsonl"
            targets.append(logfile)


# == Define rules ==
rule all:
    input:
        targets,


rule train_spaceenv:
    output:
        config["logdir"] + "/{spaceenv}/{algo}.jsonl",
    threads: config["concurrency"]
    params:
        concurrency=config["concurrency"],
    log:
        err="outputs/logs/{spaceenv}/{algo}.err",
    shell:
        """
        python benchmarks/run.py \
            algo={wildcards.algo} \
            spaceenv={wildcards.spaceenv} \
            concurrency={params.concurrency} \
            2> {log.err}
        """
