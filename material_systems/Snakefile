
configfile: "config.yaml"

rule all:
    """
    Final rule that ensures the pipeline completes once 'done.txt' is created.
    """
    input:
        f"{config['output_dir']}/done.txt"

rule run_workflow:
    """
    Rule that executes the SimpleWorkflowPipeline in the Python script
    with the arguments from our config file.
    """
    output:
        touch(f"{config['output_dir']}/done.txt")
    log:
        "logs/workflow.log"
    params:
        abstracts=config["abstracts_csv"],
        materials=config["materials_dir"],
        output=config["output_dir"],
        config_file="config.yaml"
    shell:
        """
        python script/full_model_results.py \
            --abstracts_csv {params.abstracts} \
            --materials_dir {params.materials} \
            --output_dir {params.output} \
            --config_file {params.config_file} > {log} 2>&1"""