# numerical-nll

## Setup and prerequisites

Before running the experiments, you need to install the required libraries. We recommend to use a virtual environment like `venv`, a lightweight native Python solution.

1. Clone this repository:
    ```bash
    git clone https://github.com/tremelow/numerical-nll.git
    cd numerical-nll
    ```

1. Create a virtual environment and activate it (optional):
   ```bash
    python -m venv numerical_nll_env
    source numerical_nll_env/bin/activate  # On Windows, use `numerical_nll_env\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Experiments

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
