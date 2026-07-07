# Physics Constrained Diffusion Models for Weather Forecasting

This is a repository for a Master Thesis and it implements physics‑constrained diffusion models for weather forecasting and related PDE problems. It provides training pipelines, evaluation scripts, hyperparameter tuning utilities, and visualization helpers for the ERA5 and Darcy Flow tasks. Use the configs directory and Hydra-style overrides to reproduce experiments or run new trials.

# :fire: Data generation
Before running experiments on **Darcy Flow** generate the data using 
```bash
python src/pde_diff/data/darcy_data_generation.py
```
## ERA5 data pipeline
- For **ERA5**, use this script to download one datafile for each year data: ```python src/pde_diff/data/download_data.py```. NB! You must register to download the data, a guide is proviced in the script. Each zarr will contain three months' hourly data: January, February and December.
- Process each zarr file into one large zarr file using ```merge_zarrs.sh```.
- To speed up training, the zarr is precomputed into memory-mapped input/target arrays for training: an input tensor formed from the two preceding states, and a target tensor representing the next-step residual/change. This is done because the last two states are used to predict the following weather state. Note: triples crossing the February-December boundary are dropped as they are not time-coherent. Run the precomputation with ```precompute_dataset.sh```
- You are now ready to start training, for example using ```train_tiny.sh``` or ```train_kfolds.sh```.


# 🚀 Experiments:

To train the model on the **Darcy dataset** with physical regularization, run:

```bash
python src/pde_diff/train.py dataset=darcy loss.name=darcy
```

To train the model on the **ERA5** with physical regularization, run:
```bash
python ./src/pde_diff/train.py model.name=unet3d_conditional loss.name=vorticity loss.c_residual=1e-2
```
To train without regularization set `loss.c_residual=0`. The residual validation error will still be logged for comparison.

## **Training Options**

- **`dataset:`** : Choose dataset to train on. Examples: `dataset=darcy`, `dataset=era5`.
- **`model.name:`** : Model architecture. Example: `model.name=unet3d_conditional`.
- **`loss.name:`** : Loss configuration. Examples: `loss.name=mse`, `loss.name=vorticity`, `loss.name=darcy`.
- **`loss.c_residual:`** : Physical-regularization strength (float). Example: `loss.c_residual=1e-2` to enable a small PDE constraint. For ERA5 one can chose residual specific weights, e.g., `loss.c_residual=[1e-2,0]` to only use the planetary vorticity residual.
- **Other overrides:** Hydra-style overrides are supported — any field from `configs/` may be overridden on the command line (e.g. `configs/dataset/era5.yaml`).

During training, logs and metrics are written under `logs/<experiment-name>-<run-id>/version_0/metrics.csv` and checkpoints under `models/<experiment-name>-<run-id>/` (or into the configured checkpoint directory).

## Hyperparameter tuning

You can run automated hyperparameter studies using the included tuning script. Results (trials and best configs) are written to the `logs/` folder and the study name you provide.

Example run (ERA5 study):
```bash
python ./src/pde_diff/hyperparam_tuning.py dataset.validation_metrics=["mse"] hp_study=era5_hp_3d_4 model.name=unet3d_conditional
```

Visualize the results
- Use `src/pde_diff/visualize_hp.py` to produce plots summarising the hyperparameter study (performance across trials, pareto fronts, best configs). By default the script reads the study outputs under `logs/<hp_study>` and saves figures under `reports/figures/hp_visualizations/`.

## **Evaluate Models**

- **Run the evaluation script**: evaluation is implemented with Hydra. Provide a path or overrides for the trained model configuration using `model.path` in the command-line overrides.

Example — run the evaluation flow (loads best fold, runs tests, produces sample figures and forecasting diagnostics):
```bash
python src/pde_diff/evaluate.py model.path=models/<your_model_id> steps=5
```

- **Outputs**: Evaluation saves test metrics to the run logs (CSV logger) and produces sample visualizations (variable fields, forecasting errors, residual errors etc.) under `reports/figures/evaluation/<run-name>_eval/sample_*` and forecasting diagnostics under `reports/figures/evaluation/<run-name>_eval/forecast_error_distributions/`.

- **To plot forecast errors of two models together** (utility): change paths to models forecast error csvs in the script `src/pde_diff/plot_forecast_losses.py` and run the script.

## **Plotting & Visualization**

- **Quick model comparison**: A helper `plot_cv_val_metrics` is available in `src/pde_diff/visualize.py`. It reads `logs/<model_id>/version_0/metrics.csv` for each model and saves a three-panel PNG comparing validation residual errors, validation mse, and val/train loss.

Example (run interactively or from the shell):
```bash
python -c "from pde_diff.visualize import plot_cv_val_metrics; 
plot_cv_val_metrics(model_ids=['modelA','modelB'],fold_num=5,log_path='logs',data_type='era5')"
```
- **Extensive visualization functions**: `src/pde_diff/visualize.py` contains an abundance of functions for plotting samples, training metrics, errors etc.


# Project structure

The directory structure of the project looks like this:
```txt
├── configs/                  # Configuration files
├── data/                     # Data directory (is made when downloading data)
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   └── all project code
└── tests/                    # Tests
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

# Prerequisites

Make sure you have the following installed on your system:

Conda (either Anaconda or Miniconda)

Python 3.8 or higher (managed via Conda)

## Step-by-Step Instructions

1. Clone the Repository

First, clone the repository to your local machine:

- ```git clone https://github.com/MarieJuhlJ/PDE_Diffusion.git```

- ```cd PDE_Diffusion```

2. Install Invoke

Install invoke using Conda or pip:

- ```conda install -c conda-forge invoke``` or ```pip install invoke```

You can verify the installation by running:

- ```invoke --version```

3. Create the Environment

This repository includes a custom create-environment function defined in the tasks.py file. Use invoke to create the environment by running:

- ```invoke create-environment```

This function will:

Create a Conda environment with the appropriate name "pde_diff" (specified in the script) of the current compatible python version, activate the environment and install invoke.

4. Install Requirements

Once the environment is set up, install additional Python dependencies using the requirements function. Activate the new environment and run the following command:

- ```invoke requirements```

This function will:

Install dependencies listed in the requirements.txt file (if applicable).

Note that windows users have to manually run the following commands:
- ```pip install -r requirements.txt```
- ```pip install -e .```

### For developers:
There are extra packages you need if you want to make changes to the project. You can install them using the `requirements_dev.txt` by invoking the task `dev_requirements"`:

```invoke dev-requirements```

or installing `requirements_dev.txt` directly with pip:

```pip install -r requirements_dev.txt```

To use pre-commit checks run the following line:
- ```pre-commit install```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Split
### train & validation (k-folds)
min_year: 2015
max_year: 2022

### test
min_year: 2023
max_year: 2024

### experiments
tiny_firstUNet3D_conditional : first run with correct model for ERA5! 
baseline: 5-fold validation of model with hyperparameters as detailed in J&M thesis