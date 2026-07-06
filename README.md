# Compare various solvers for floquet. 

## Install commands

1. Install `uv` locally or via `conda`. Instruction [https://docs.astral.sh/uv/getting-started/installation/](here). 
2. Clone this repo. Inside the directory run `uv init --name floquet_gpu` to install the necessary dependencies. 
3. The benchmarking shell scripts are straightforward to run (e.g. `./submit.sh`). This submits all the benchmarking scripts as a series of SLURM array jobs.
You might have to delete a whole bunch of output files.
4. To run a notebook, select the virtual environment `floquet_gpu`. 