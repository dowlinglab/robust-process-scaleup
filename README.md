Optimal experimental design for the isolated lab-scale batch reactor (BR) model is performed in the `batch_reactor_exp_design.py` file.

Parameter estimation and uncertainty quantification for the kinetic parameters of the isolated BR model is performed in the 
`batch_reactor_parameter_est.ipynb` file.

The uncertainty in the estimated kinetic parameters of the isolated BR model is propagated to the design of the Williams-Otto[1] process in 
the `williams_otto_process.ipynb` file.

The `batch_experiment.py` file holds the `Experiment` class and mathematical model of the isolated BR.

[1] Biegler LT. Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes. Society for Industrial and Applied Mathematics (2010). 
ISBN 978-0-898717-02-0
