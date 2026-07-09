# Experimental Design and Uncertainty Propagation for Process Scale-Up Paper

Optimal experimental design for the isolated lab-scale batch reactor (BR) model is performed in the `batch_reactor_exp_design.py` file.

Parameter estimation and uncertainty quantification for the kinetic parameters of the isolated BR model is performed in the 
`batch_reactor_parameter_est.ipynb` file.

The uncertainty in the estimated kinetic parameters of the isolated BR model is propagated to the design of the Williams-Otto [1] process in 
the `williams_otto_process.ipynb` file.

The `batch_experiment.py` file holds the `Experiment` class and the mathematical model of the isolated BR.

Running these files require the Pyomo and IDAES-PSE packages. The following section provides guidance on how to install Pyomo and IDAES-PSE.

## Making an IDAES-PSE environment

We recommend using a Conda environment.

### 1. Create a new Conda environment (replace my-idaes-env with your preferred name)

```bash
conda create --yes --name my-idaes-env python=3.10
conda activate my-idaes-env
```

### 2. Install IDAES

```bash
conda install --yes -c conda-forge idaes-pse
```

### 3. Install the IDAES extensions

```bash
idaes get-extensions
```

The IDAES extensions include the compiled solver binaries and function libraries required by many IDAES examples.

### 4. Install Pyomo with your preferred package manager

```bash
pip install pyomo
```

### 5. Install NumPy and Pandas

```bash
pip install numpy pandas
```

### 6. Install Matplotlib and SciPy

```bash
pip install scipy matplotlib
```

[1] Biegler LT. Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes. Society for Industrial and Applied Mathematics (2010). 
ISBN 978-0-898717-02-0
