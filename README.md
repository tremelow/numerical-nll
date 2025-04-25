# numerical-nll

## Setup and prerequisites

Before running the experiments, you need to install the required libraries. We recommend to use a virtual environment like `venv`, a lightweight native Python solution.

1. Clone this repository:
    ```bash
    git clone https://github.com/tremelow/numerical-nll.git
    cd numerical-nll
    ```

2. Create a virtual environment and activate it (optional):
   ```bash
    python -m venv numerical_nll_env
    source numerical_nll_env/bin/activate  # On Windows, use `numerical_nll_env\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Toy model experiments

List of experiments provided (in alphabetical order):
- `broyden_benchmark.ipynb`: Benchmark of average Broyden iterations required for each time-step per total steps and per dimension.
- `timesteps_comparison.ipynb`: Comparison between linear and [EDM](https://arxiv.org/abs/2206.00364) time-steps scheduling, with mean squared error (MSE) and plots between the computed and quasi-exact NLL for various total steps.
- `viz_numerical_cube.ipynb`: Plots and NLL computation on uniform Gaussian mixtures in varying dimension, with components centered at
each vertex of an hypercube. Also contains Explicit Euler method issue illustration.

Temp:
- `naive_time_evol.ipynb`
- `nll_cube.ipynb`
- `viz_cube.ipynb`
- `viz_dynamic.ipynb`
- `viz_mixture.ipynb`
- `viz_numerical_hexa.ipynb`
- `viz_ode.ipynb`
- `viz_single.ipynb`

## Fashion-MNIST experiments

### Data and trained model

We provide the dataset, as well as the trained model: [Google Drive link](https://drive.google.com/drive/folders/1GKwCEf9rwETgW80E78kggmw-6cd0Dypk?usp=sharing)

The model has been trained using the [EDM repo](https://github.com/NVlabs/edm). The dataset has been pre-processed as indicated in their [README](https://github.com/NVlabs/edm?tab=readme-ov-file#preparing-datasets). We then ran their `train.py` script with `-cond=0 -arch=ddpmpp` hyperparameters.

In addition to the base model (`fmnist_trained_model.pkl`), we provide another version (same weights) that is compatible with `torch.func.jacrev` (`fmnist_trained_model_jacrev.pkl`).

### Compute NLL

#### Quasi-exact NLL (black-box ODE solver)

To compute the quasi-exact NLL with the use of `solve_ivp`, run the following script:
```bash
python fmnist_solve_ivp_nll.py --model-path <path_to_fmnist_trained_model.pkl> --data-path <path_to_fmnist_dataset.zip>
```

#### NLL for the forward implicit Euler method (Broyden's method)

- To compute the NLL for the forward implicit Euler method, you can run:
   ```bash
   python fmnist_broyden_jacrev_nll.py --model-path <path_to_fmnist_trained_model_jacrev.pkl> --data-path <path_to_fmnist_dataset.zip> --num-steps <total_number_of_timesteps>
   ```
   This version is the fastest, thanks to the use of `torch.func.jacrev` for computing jacobians, but requires significant GPU memory (around 40GB of VRAM).

- Or, if you don't have a GPU with enough memory, you can run this other script:
   ```bash
   python fmnist_broyden_nll.py --model-path <path_to_fmnist_trained_model.pkl> --data-path <path_to_fmnist_dataset.zip> --num-steps <total_number_of_timesteps> --batch-size <batch_size>
   ```
   In this script, the jacobians are assembled by hand by computing the gradients manually, which is much less costly in terms of GPU memory, but also slower in the end.

In both cases, given the total calculation time, the NLL values are stored in files (`nll_vals/num_steps/*.pt` by default), this ensures that the values obtained are not lost. To calculate the mean NLL value, you can use the following script: 
```bash
python fmnist_mean_nll.py --nll-folder <path_to_nll_vals>
```
