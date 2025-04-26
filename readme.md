# Vec2Text Reproducability Study

This repository contains code for reproducing results from the paper ["Text Embeddings Reveal (Almost) As Much As Text"](https://arxiv.org/abs/2310.06816).

> **Abstract**: This work investigates the Vec2Text method, which frames the embedding inversion problem as a controlled generation task. Our study aims to validate Vec2Text's ability to reconstruct text from embeddings, highlighting the privacy risks associated with embedding inversion. We reproduce Vec2Text's performance in both in-domain and out-of-domain settings, verifying its effectiveness while noting some discrepancies due to experimental details. We extend the study by exploring parameter sensitivity, password reconstruction, and embedding quantization as a defense strategy. The study concludes with insights into Vec2Text's robustness and potential risks, emphasizing the need for further research on embedding inversion methods and strategies to prevent them.

## Project structure

We use the following project structure:

```
ir2/ - source directory
notebooks/ - jupyter notebooks
runs/ - configuration files for running experiments.
jobs/ - sbatch jobs for running experiments
scripts/ - scripts for running experiments
tests/ - tests for the project
docs/ - documentation for some of the findings
```

## Installation

1. Clone the repository
2. Setup correct Python environment using poetry and pyenv:

```bash
pyenv install 3.11.6
pyenv local 3.11.6
poetry install
```

3. Install pre-commit hooks:

```
pre-commit install
```

4. To export pyproject.toml to conda environment:

```
poetry run poetry2conda pyproject.toml environment.yaml
```

<<<<<<< HEAD
## Snellius setup

1. Add the following to your `~/.bashrc` to enable conda in the terminal - `$ echo 'export PATH="/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin:$PATH"' >> ~/.bashrc`
2. Allocate some GPU recorces by running `$ srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18  --time=00:30:00 --pty bash -i`
3. Navigate to the project
4. Run `$ conda env create -f environment.yaml`
5. Activate environment: `$ source activate uva-ir2`
6. Additionally install project by running `$ pip install -e .`
7. Login to wandb: `$ wandb login`
8. Verify that all is working by running `$ python tests/integration/test_inference_model.py`

### Running sbatch jobs

To run sbatch job, just modify the `snellius.job` file with the needed resources (mostly, change the execution time) and then run desired command by running: `$ sbatch snellius.job <COMMAND>`.

=======
>>>>>>> ed39763 (Removing uva connection)
## Usage

To run inference scripts, you need to first login to wandb:

```bash
poetry run wandb login
```

Then you can run the scripts:

```bash
python scripts/inference.py <RUN_CONFIG>
```

where `<RUN_CONFIG>` is the path to the run config file. All of the config files are located in the `runs` directory.

### Configuration

- `model_name` - name (path) of the encoder model to use
- `corrector_name` - name (path) of the corrector model to use
- `dataset` - name of the dataset to use (e.g. `quora`)
- `batch_size` - batch size for inference
- `num_steps` - number of steps to run while correcting the embedding inversion
- `add_gaussian_noise` - (default `false`) whether to add Gaussian noise to the embeddings
- `noise_mean` - (not used if `add_gaussian_noise` is set to false) mean of the Gaussian noise
- `noise_std` - (not used if `add_gaussian_noise` is set to false) standard deviation of the Gaussian noise
- `noise_lambda`(not used if `add_gaussian_noise` is set to false) scalar for Gaussian noise
