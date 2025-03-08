
# Iterative Corpus Refinement for Materials Property Prediction Based on Scientific Texts

This repository contains the code and workflows described in the paper *"Iterative Corpus Refinement for Materials Property Prediction Based on Scientific Texts"*. The project demonstrates an iterative approach to refining a corpus for improved predictions in materials science.

## Getting Started

### Prerequisites
- Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Clone this repository to your local machine.

### Setting Up the Environment
1. Use the `environment.yml` file to create the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate your_env_name
   ```
   Replace `your_env_name` with the environment name in `environment.yml`.

2. Ensure material systems are placed in the correct directories:
   - ORR and HER material systems: `./material_systems/MinDMaxC`
   - OER material systems: `./material_systems/MaxDMinC`

## Workflow Overview

### Step 1: Collecting Papers
1. Navigate to the `01_collect_papers` directory.
2. Update the `config.yaml` file:
   - Add your Scopus API key under the `APIKey` entry.
   - Adjust other settings if necessary.
3. Run the Snakefile to collect and process the papers:
   ```bash
   snakemake -c1
   ```

### Step 2: Paper Selection
1. Navigate to the `02_paper_selection` directory.
2. Open and run the code blocks in the `02_paper_selection.ipynb` Jupyter notebook.
3. Outputs:
   - Selected paper results for each material system.
   - Word2Vec models trained on the selected papers.
   - Similarity results calculated between each material and the property words "dielectric" and "conductivity."

### Step 3: Full Corpus Model
1. Navigate to the `03_full_model` directory.
2. Run the Snakefile:
   ```bash
   snakemake -c1
   ```
3. Outputs:
   - Model trained on all papers collected in Step 1.
   - Similarity results based on this model.

### Step 4: Pareto Optimization
1. Navigate to the `04_pareto_prediction` directory.
2. Run the Snakefile:
   ```bash
   snakemake -c1
   ```
3. Outputs:
   - Pareto optimization results for material systems:
     - Selected papers: `02_paper_selection/selection_results`
     - Full papers: `./full_model_results`

### Step 5: Generating Tables
1. Navigate to the `05_tables` directory.
2. Run the Snakefile:
   ```bash
   snakemake -c1
   ```
   Tables will be generated in this step and are required for generating figures.

### Step 6: Generating Figures
1. Navigate to the `06_figs` directory.
2. Run the Snakefile:
   ```bash
   snakemake -c1
   ```
   All plots will be generated and saved in this directory.

## License
This project is licensed under the LGPL-3.0 License. See the [LICENSE](LICENSE) file for details.
