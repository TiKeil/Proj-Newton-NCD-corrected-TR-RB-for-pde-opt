```
# ~~~
# This file is part of the paper:
#
#           "An adaptive projected Newton non-conforming dual approach
#         for trust-region reduced basis approximation of PDE-constrained
#                           parameter optimization"
#
#   https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt
#
# Copyright 2019-2020 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Luca Mechelli (2019 - 2020)
#   Tim Keil      (2019 - 2020)
# ~~~
```

In this repository, we provide jupyter-notebooks and the entire code for the numerical experiments in Section 4 of the paper 
"An adaptive projected Newton non-conforming dual approach for trust-region reduced basis approximation of PDE-constrained parameter optimization"
by Stefan Banholzer, Tim Keil, Luca Mechelli, Mario Ohlberger, Felix Schindler and Stefan Volkwein. 

For just taking a look at the provided (precompiled) jupyter-notebooks, you do not need to install the software.
Just go to [`notebooks/Paper2_simulations`](https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt/tree/master/notebooks). If you want to have a closer look at the implementation or compile the results by
yourself, we provide simple setup instructions for configuring your own Python environment in a few steps.
We note that our setup instructions are written for Linux or Mac OS only and we do not provide setup instructions for Windows.
We also emphasize that our experiments have been computed on a fresh Ubuntu 20 system with Python version 3.8.5. with 12 GB RAM. 

# Organization of the repository

Our implementation is based on pyMOR (https://github.com/pymor/pymor).
Further extensions that we used for this paper can be found in the directory [`pdeopt/`](https://github.com/TiKeil//tree/master/pdeopt). 
For our three optimization experiments we considered ten different starting parameters with different seeds in the random generator. 
The complete results from these starting values are stored in each respective directory under

* **Section 4.3:** [here](https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt/tree/master/notebooks/Paper2_simulations/EXC_12_Parameters)
* **Section 4.4:** [`here`](https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt/tree/master/notebooks/Paper2_simulations/EXC_28_Parameters)

We also provide an extensive view of the results of the estimator study under 

# How to find figures and tables from the paper

We provide instructions on how to find all figures and tables from the paper. 

**Figure 1**: The data of the blueprint is in [`EXC_data/`](https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt/tree/master/EXC_data). 
The used file for Figure 3 is `full_diffusion_with_big_numbers_with_D.png`

**Figure 2**: This result is based on starting value seed 1 (Starter1)
[`here`](https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt/blob/master/notebooks/Paper2_simulations/EXC_12_Parameters/). 
The three different directories correspond to both FOC tolerances and the parameter control.
Go to the bottom of the notebook to see the Figure.
If you followed the setup instructions you can also construct this figure by running the file `figure.py` for each directory
in [`here`](https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt/tree/master/notebooks/Paper2_simulations/EXC_12_Parameters/results) 

**Figure 3**: This result is based on starting value seed 4 (Starter4) which you can view in
[`here`](https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt/blob/master/notebooks/Paper2_simulations/EXC_28_Parameters/Compare_methods_starter4.ipynb) 
Also, you can run the corresponding file in the respective `results/` directory.

The tables are also constructable in the corresponding `results/` directory of both experiments.

# Setup

On a Linux or Mac OS system with Python and git installed, clone
the repo in your favorite directory

```
git clone https://github.com/TiKeil/Proj-Newton-NCD-corrected-TR-RB-for-pde-opt
```

Initialize all submodules via

```
cd Proj-Newton-NCD-corrected-TR-RB-for-pde-opt
git submodule update --init --recursive
```

Now, run the provided setup file via 

```
./setup.sh
```

# Running the jupyter-notebooks

If you want to interactively view or compile the notebooks, just activate and start jupyter-notebook 

```
source venv/bin/activate
jupyter-notebook --notebook-dir=notebooks
```

We recommend to use notebook extensions for a better overview in the notebooks.
After starting the jupyter-notebook server go to Nbextensions, deactivate the first box and activate at least `codefolding` and `Collapsible Headings`. 
