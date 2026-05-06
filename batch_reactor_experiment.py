import pyomo.environ as pyo
import math
from matplotlib.ticker import MaxNLocator
from pyomo.network import Arc
from scipy import stats
import pandas as pd
import numpy as np
import pyomo.dae as dae
import matplotlib.pyplot as plt
from pyomo.contrib.parmest.experiment import Experiment
from pyomo.contrib.doe import DesignOfExperiments


class BatchReactorExperiment(Experiment):
    """Creates and labels the Pyomo model of the batch reactor

    Parameters
    ----------
    data: Pandas dataframe or .csv file,
        Data containing the sample time and measured values of mass fractions
    XA0: float,
        Initial mass fraction of species A
    temp_control: int, float, or dict,
        Constant or piecewise-linear profile of the reaction temperature (in R)
    const_temp: Boolean,
        Species if the batch reactor is a constant or variable temperature system
    comb_const_temp: Boolean,
        Specifies if the data for the constant low and high temperatures are combined
    doe_solve: Boolean,
        Species if the batch reactor model is being used for optimal experimental design
    sample_time: float or int,
        Time to collect a sample for mass fraction measurements

    Returns
    -------
    m: annotated Pyomo model of the batch reactor
    """

    def __init__(self, XA0, temp_control, const_temp, data=None, doe_solve=False,
                 sample_time=None, ):

        self.data = data
        self.XA0 = XA0
        self.temp_control = temp_control
        self.const_temp = const_temp
        self.doe_solve = doe_solve
        self.sample_time = sample_time
        self.model = None

    def get_labeled_model(self):
        self.create_model()
        self.label_model()

        return self.model

    def create_model(self):
        if self.const_temp and not self.doe_solve:
            self.model = reform_batch_reactor_model_const(self.XA0, self.temp_control)
        elif not self.const_temp and not self.doe_solve:
            self.model = reform_batch_reactor_model_var(self.XA0, self.temp_control)
        elif not self.const_temp and self.doe_solve:
            self.model = reform_batch_reactor_model_doe(self.XA0, self.sample_time, self.temp_control)

        return self.model

    def label_model(self):

        m = self.model

        if self.doe_solve:
            # label the experimental decision variables
            m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_inputs[m.XA0] = None
            m.experiment_inputs[m.tf] = None

            # label the measured variables
            m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_outputs.update(
                (m.XA[tau], None) for tau in m.tau
            )
            m.experiment_outputs.update(
                (m.XB[tau], None) for tau in m.tau
            )
            m.experiment_outputs.update(
                (m.XC[tau], None) for tau in m.tau
            )
            m.experiment_outputs.update(
                (m.XP[tau], None) for tau in m.tau
            )
            m.experiment_outputs.update(
                (m.XE[tau], None) for tau in m.tau
            )
            m.experiment_outputs.update(
                (m.XG[tau], None) for tau in m.tau
            )

            # add the measurement errors
            m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.measurement_error.update(
                (m.XA[tau], 0.01) for tau in m.tau
            )
            m.measurement_error.update(
                (m.XB[tau], 1e-06) for tau in m.tau
            )
            m.measurement_error.update(
                (m.XC[tau], 1e-06) for tau in m.tau
            )
            m.measurement_error.update(
                (m.XP[tau], 0.01) for tau in m.tau
            )
            m.measurement_error.update(
                (m.XE[tau], 0.01) for tau in m.tau
            )
            m.measurement_error.update(
                (m.XG[tau], 0.01) for tau in m.tau
            )
        else:
            meas_time_points = self.data["Time (hr)"]

            # label the measured variables
            m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_outputs.update(
                (m.XA[t], self.data["XA"][ind]) for ind, t in enumerate(meas_time_points)
            )
            m.experiment_outputs.update(
                (m.XB[t], self.data["XB"][ind]) for ind, t in enumerate(meas_time_points)
            )
            m.experiment_outputs.update(
                (m.XC[t], self.data["XC"][ind]) for ind, t in enumerate(meas_time_points)
            )
            m.experiment_outputs.update(
                (m.XP[t], self.data["XP"][ind]) for ind, t in enumerate(meas_time_points)
            )
            m.experiment_outputs.update(
                (m.XE[t], self.data["XE"][ind]) for ind, t in enumerate(meas_time_points)
            )
            m.experiment_outputs.update(
                (m.XG[t], self.data["XG"][ind]) for ind, t in enumerate(meas_time_points)
            )

            # add the measurement errors
            m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.measurement_error.update(
                (m.XA[t], 0.01) for t in meas_time_points
            )
            m.measurement_error.update(
                (m.XB[t], 1e-06) for t in meas_time_points
            )
            m.measurement_error.update(
                (m.XC[t], 1e-06) for t in meas_time_points
            )
            m.measurement_error.update(
                (m.XP[t], 0.01) for t in meas_time_points
            )
            m.measurement_error.update(
                (m.XE[t], 0.01) for t in meas_time_points
            )
            m.measurement_error.update(
                (m.XG[t], 0.01) for t in meas_time_points
            )

        # label the unknown parameters
        m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        m.unknown_parameters.update(
            (k, pyo.value(k)) for k in [m.alpha_1, m.alpha_2, m.alpha_3, m.E1, m.E2, m.E3]
        )

        return m


def reform_batch_reactor_model_doe(XA0, sample_time, temp_control_points):
    """
    Reformulates the variable temperature batch reactor model for parameter estimation

    Parameters
    ----------
    XA0: float,
        Initial mass fraction of species A
    sample_time: float or int,
        Time to collect a sample for mass fraction measurements
    temp_control_points: dict,
        Piecewise-linear profile of the reaction temperature (in R)

    Returns
    -------
    model: Pyomo model of the variable temperature batch reactor

    """
    model = pyo.ConcreteModel()

    # define sets
    reaction_number = [1, 2, 3]
    model.tau = dae.ContinuousSet(bounds=[0, 1])  # dimensionless time

    # define the final time variable
    model.tf = pyo.Var(bounds=[1e-6, 3],)
    model.tf.fix(sample_time)

    # define the initial mass fraction of species A
    model.XA0 = pyo.Var(bounds=[0, 1])
    model.XA0.fix(XA0)

    # define the model parameters
    model.alpha_1 = pyo.Var(bounds=(0, None),)
    model.alpha_1.fix(22.600)

    model.alpha_2 = pyo.Var(bounds=(0, None),)
    model.alpha_2.fix(28.296)

    model.alpha_3 = pyo.Var(bounds=(0, None),)
    model.alpha_3.fix(37.020)

    model.E1 = pyo.Var(bounds=(0, None),)
    model.E1.fix(120.526)

    model.E2 = pyo.Var(bounds=(0, None),)
    model.E2.fix(148.294)

    model.E3 = pyo.Var(bounds=(0, None),)
    model.E3.fix(201.293)

    # add the mass fraction variables
    model.XA = pyo.Var(model.tau, bounds=(0, 1), initialize=XA0)
    model.XB = pyo.Var(model.tau, bounds=(0, 1), initialize=1-XA0-0.04)
    model.XC = pyo.Var(model.tau, bounds=(0, 1), initialize=0.01)
    model.XE = pyo.Var(model.tau, bounds=(0, 1), initialize=0.01)
    model.XP = pyo.Var(model.tau, bounds=(0, 1), initialize=0.01)
    model.XG = pyo.Var(model.tau, bounds=(0, 1), initialize=0.01)

    # add the temperature variables
    model.T_reparam = pyo.Var(model.tau, bounds=(0, 1))

    # add the rate constants
    model.k_reparam = pyo.Var(reaction_number, model.tau, bounds=(0, None))
    model.k = pyo.Var(reaction_number, model.tau, bounds=(0, None))

    # calculate the reparameterized rate constants
    def k_reparam_rule(m, i, tau):
        if i == 1:
            return m.k_reparam[i, tau] == model.alpha_1 - m.E1 * m.T_reparam[tau]
        elif i == 2:
            return m.k_reparam[i, tau] == model.alpha_2 - m.E2 * m.T_reparam[tau]
        else:
            return m.k_reparam[i, tau] == model.alpha_3 - m.E3 * m.T_reparam[tau]

    model.k_reparam_eq = pyo.Constraint(
        reaction_number, model.tau, rule=k_reparam_rule
    )

    # calculate the original rate constants
    def k_rule(m, i, tau):
        return m.k[i, tau] == pyo.exp(m.k_reparam[i, tau])

    model.k_eq = pyo.Constraint(
        reaction_number, model.tau, rule=k_rule
    )

    # reaction rates
    model.r = pyo.Var(reaction_number, model.tau, bounds=(0, None),)

    # define the rate of reactions
    def reaction_rate_rule(m, i, tau):
        if i == 1:
            return m.r[i, tau] == m.k[i, tau] * m.XA[tau] * m.XB[tau]
        elif i == 2:
            return m.r[i, tau] == m.k[i, tau] * m.XB[tau] * m.XC[tau]
        else:
            return m.r[i, tau] == m.k[i, tau] * m.XC[tau] * m.XP[tau]

    model.reaction_rate = pyo.Constraint(
        reaction_number, model.tau, rule=reaction_rate_rule
    )

    # add the differential equations for XA, XB, XC, XE, XP, and XG
    model.dXA = dae.DerivativeVar(model.XA, wrt=model.tau)
    model.dXB = dae.DerivativeVar(model.XB, wrt=model.tau)
    model.dXC = dae.DerivativeVar(model.XC, wrt=model.tau)
    model.dXE = dae.DerivativeVar(model.XE, wrt=model.tau)
    model.dXP = dae.DerivativeVar(model.XP, wrt=model.tau)
    model.dXG = dae.DerivativeVar(model.XG, wrt=model.tau)

    @model.Constraint(model.tau)
    def xa_rate_ode(m, tau):
        if tau == m.tau.first():
            return pyo.Constraint.Skip
        return m.dXA[tau] == m.tf * (-m.k[1, tau] * m.XA[tau] * m.XB[tau])

    @model.Constraint(model.tau)
    def xb_rate_ode(m, tau):
        if tau == m.tau.first():
            return pyo.Constraint.Skip
        return m.dXB[tau] == - m.tf * (m.k[1, tau] * m.XA[tau] * m.XB[tau] + m.k[2, tau] * m.XB[tau] * m.XC[tau])

    @model.Constraint(model.tau)
    def xc_rate_ode(m, tau):
        if tau == m.tau.first():
            return pyo.Constraint.Skip
        return m.dXC[tau] == m.tf * (2 * m.k[1, tau] * m.XA[tau] * m.XB[tau] - 2 * m.k[2, tau] * m.XB[tau] * m.XC[tau] - m.k[3, tau] * m.XC[tau] * m.XP[tau])

    @model.Constraint(model.tau)
    def xe_rate_ode(m, tau):
        if tau == m.tau.first():
            return pyo.Constraint.Skip
        return m.dXE[tau] == m.tf * (2 * m.k[2, tau] * m.XB[tau] * m.XC[tau])

    @model.Constraint(model.tau)
    def xg_rate_ode(m, tau):
        if tau == m.tau.first():
            return pyo.Constraint.Skip
        return m.dXG[tau] == m.tf * (1.5 * m.k[3, tau] * m.XC[tau] * m.XP[tau])

    # add the mass fraction constraint
    @model.Constraint(model.tau)
    def sum_mass_fraction(m, tau):
        if tau == m.tau.first():
            return pyo.Constraint.Skip
        return m.XA[tau] + m.XB[tau] + m.XC[tau] + m.XE[tau] + m.XG[tau] + m.XP[tau] == 1

    # fix the initial conditions
    tau_0 = model.tau.first()
    model.XA_init = pyo.Constraint(expr=model.XA[tau_0] == model.XA0)
    model.XB_init = pyo.Constraint(expr=model.XB[tau_0] == 1 - model.XA0)
    model.XC_init = pyo.Constraint(expr=model.XC[tau_0] == 0.0)
    model.XE_init = pyo.Constraint(expr=model.XE[tau_0] == 0.0)
    model.XP_init = pyo.Constraint(expr=model.XP[tau_0] == 0.0)
    model.XG_init = pyo.Constraint(expr=model.XG[tau_0] == 0.0)

    # discretize the model
    disc = pyo.TransformationFactory("dae.finite_difference")
    disc.apply_to(model, nfe=60, scheme="BACKWARD")

    # add a piecewise-linear temperature profile
    tf_val = pyo.value(model.tf)
    for tau in model.tau:
        t_actual = tau * tf_val

        if t_actual < 0.5:
            temp_at_time_tau = temp_control_points["0.0"] + 0.2 * t_actual

        elif t_actual < 1.0:
            temp_at_time_tau = temp_control_points["0.5"] + 0.6 * (t_actual - 0.5)

        else:
            temp_at_time_tau = 6.2

        model.T_reparam[tau].fix(1 / temp_at_time_tau)

    # define the solver
    solver = pyo.SolverFactory('ipopt')

    # solve the model
    results = solver.solve(model, tee=True)

    return model

# define the prior information (Fisher information matrix)
prior_FIM = np.array([
    [1.54923906e+10, 1.46769057e+09, -1.70116610e+09, -2.64078888e+09, -2.47886401e+08, 2.88800421e+08],
    [1.46769032e+09, 2.62585135e+09, 2.82304932e+08, -2.50870575e+08, -4.46282077e+08, -4.80859752e+07],
    [-1.70116614e+09, 2.82304901e+08, 3.17900344e+08, 2.88573724e+08, -4.84468430e+07, -5.37121274e+07],
    [-2.64078888e+09, -2.50870616e+08, 2.88573717e+08, 4.50173794e+08, 4.23788425e+07, -4.89966831e+07],
    [-2.47886360e+08, -4.46282077e+08, -4.84468482e+07, 4.23788354e+07, 7.58506979e+07, 8.25060451e+06],
    [2.88800428e+08, -4.80859700e+07, -5.37121274e+07, -4.89966843e+07, 8.25060362e+06, 9.07666881e+06]
])

# define the temperature control points
temp_control = {"0.0":5.8, "0.5":5.9, "1.0":6.2, "1.5":6.4, "2.0":6.6, "3.0":6.8}

# Sensitivity analysis
# Range to evaluate designs
design_ranges_plot = {"XA0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,],
                      "tf": [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25]}

# store the FIM, initial mass fraction of species A, and final time for mass fraction measurements
FIM_results = []
XA0_container = []
tf_container = []

count = 0
# Grid search
for XA0 in design_ranges_plot["XA0"]:
    for tf in design_ranges_plot["tf"]:
        count += 1
        print("=======Iteration Number: {} =======".format(count))
        print("Design variable values for this iteration: (XA0: {}, tf: {})".format(XA0, tf))
        # update flowrates
        XA0_container.append(XA0)
        tf_container.append(tf)

        # create MembraneExperiment_plot object
        experiment = BatchReactorExperiment(XA0=XA0, sample_time=tf, temp_control=temp_control,
                                            const_temp=False, doe_solve=True)

        # Use a central difference, with step size 1e-3
        fd_formula = "central"
        step_size = 1E-2

        # Use the determinant objective
        objective_option = "determinant"

        # Create the DesignOfExperiments object
        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_formula,
            step=step_size,
            objective_option=objective_option,
            prior_FIM=prior_FIM,
            tee=True)

        # Compute the full factorial design with the sequential FIM calculation
        FIM_results.append(doe_obj.compute_FIM(method="sequential"))

# Extract criteria from FIM
def get_FIM_metrics(result):
    """Computes A, D, E, and ME optimalities

    Argument:
        result: FIM of individual experiments

    Returns:
        A_opt, D_opt, E_opt, and ME_opt: A, D, E, and ME optimality, respectively"""

    eigenvalues, eigenvectors = np.linalg.eig(result)  # eigen decomposition of FIM
    min_eig = min(eigenvalues)

    # compute optimalities
    A_opt = np.log10(np.trace(result))
    D_opt = np.log10(np.linalg.det(result))
    E_opt = np.log10(min_eig)
    ME_opt = np.log10(np.linalg.cond(result))

    return A_opt, D_opt, E_opt, ME_opt

# computing optimalities of experiments
FIM_metrics = []
for i in FIM_results:  # loop through FIM of experiments
    FIM_metrics.append(get_FIM_metrics(i))

FIM_metrics_np = np.asarray(FIM_metrics)

# Make heat map
def plot_heatmap(data, title, y_label, x_label, colorbar_label):
    # set heatmap x,y ranges
    x_tick_labels_0 = np.sort(np.unique(data[:, 0]))
    y_tick_labels_0 = np.sort(np.unique(data[:, 1]))[::-1]

    x_tick_labels = [round(x, 1) for x in x_tick_labels_0]
    y_tick_labels = [round(y, 1) for y in y_tick_labels_0]

    # optimality-values
    opt_vals = np.asarray(data[:, 2]).reshape(len(x_tick_labels), len(y_tick_labels))

    # Plot the colormap
    fig = plt.figure()

    # Plotting options
    ax = fig.add_subplot(111)
    params = {"mathtext.default": "regular"}
    plt.rcParams.update(params)

    # Plotting data
    plt.title(title, fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(y_tick_labels)))
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')

    ax.set_xticks(range(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')

    ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=9))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    im = ax.imshow(opt_vals.T, cmap=plt.cm.hot_r)
    ba = plt.colorbar(im)
    ba.set_label(colorbar_label, fontsize=14, fontweight='bold')
    ba.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(f"{colorbar_label}_10_by_10.png")
    plt.show()

# X and Y axis labels
x_label = "Sample Collection Time (hr)"
y_label = "Initial Mass Fraction of A"

# Draw A-optimality figure
data_A = np.zeros((len(FIM_metrics), 3))
data_A[:, 0] = tf_container
data_A[:, 1] = XA0_container
data_A[:, 2] = FIM_metrics_np[:, 0]

plot_heatmap(data_A, "A-optimality", y_label, x_label, "log10(trace(FIM))")

# Draw D-optimality figure
data_D = np.zeros((len(FIM_metrics), 3))
data_D[:, 0] = tf_container
data_D[:, 1] = XA0_container
data_D[:, 2] = FIM_metrics_np[:, 1]

plot_heatmap(data_D, "D-optimality", y_label, x_label, "log10(det(FIM))")

# Draw E-optimality figure
data_E = np.zeros((len(FIM_metrics), 3))
data_E[:, 0] = tf_container
data_E[:, 1] = XA0_container
data_E[:, 2] = FIM_metrics_np[:, 2]

plot_heatmap(data_E, "E-optimality", y_label, x_label, "log10(min-eig(FIM))")

# Draw ME-optimality figure
data_ME = np.zeros((len(FIM_metrics), 3))
data_ME[:, 0] = tf_container
data_ME[:, 1] = XA0_container
data_ME[:, 2] = FIM_metrics_np[:, 3]

plot_heatmap(data_ME, "ME-optimality", y_label, x_label, "log10(cond(FIM))")