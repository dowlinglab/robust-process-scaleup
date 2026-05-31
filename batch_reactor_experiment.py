import pyomo.environ as pyo
import idaes
import math
from matplotlib.ticker import MaxNLocator
from pyomo.network import Arc
from scipy import stats
import pandas as pd
import numpy as np
import pyomo.dae as dae
import matplotlib.pyplot as plt
from pyomo.contrib.parmest.experiment import Experiment
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.dae import Simulator


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

    def __init__(self, temp_control, const_temp, XA0=None, data=None, doe_solve=False,):

        self.data = data
        self.XA0 = XA0
        self.temp_control = temp_control
        self.const_temp = const_temp
        self.doe_solve = doe_solve
        self.model = None

    def get_labeled_model(self):
        self.create_model()
        self.label_model()

        return self.model

    def create_model(self):
        if self.const_temp and not self.doe_solve:
            self.model = reform_const_temp_reactor_model(self.XA0, self.temp_control)
        elif not self.const_temp and not self.doe_solve:
            self.model = reform_var_temp_reactor_model(self.XA0, self.temp_control)
        elif not self.const_temp and self.doe_solve:
            self.model = reform_var_temp_doe_model(self.temp_control)

        return self.model

    def label_model(self):

        m = self.model

        if self.doe_solve:
            # label the experimental decision variables
            m.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_inputs[m.XA[0]] = None
            m.experiment_inputs.update(
                (m.T_reparam[t], None) for t in m.t
            )

            # label the measured variables
            m.experiment_outputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.experiment_outputs.update(
                (m.XA[t], None) for t in m.t
            )
            m.experiment_outputs.update(
                (m.XB[t], None) for t in m.t
            )
            m.experiment_outputs.update(
                (m.XC[t], None) for t in m.t
            )
            m.experiment_outputs.update(
                (m.XP[t], None) for t in m.t
            )
            m.experiment_outputs.update(
                (m.XE[t], None) for t in m.t
            )
            m.experiment_outputs.update(
                (m.XG[t], None) for t in m.t
            )

            # add the measurement errors
            m.measurement_error = pyo.Suffix(direction=pyo.Suffix.LOCAL)
            m.measurement_error.update(
                (m.XA[t], 0.01) for t in m.t
            )
            m.measurement_error.update(
                (m.XB[t], 0.001) for t in m.t
            )
            m.measurement_error.update(
                (m.XC[t], 0.001) for t in m.t
            )
            m.measurement_error.update(
                (m.XP[t], 0.01) for t in m.t
            )
            m.measurement_error.update(
                (m.XE[t], 0.01) for t in m.t
            )
            m.measurement_error.update(
                (m.XG[t], 0.01) for t in m.t
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
                (m.XB[t], 0.001) for t in meas_time_points
            )
            m.measurement_error.update(
                (m.XC[t], 0.001) for t in meas_time_points
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


def reform_var_temp_doe_model(temp_control_points):
    """
    Reformulates the variable temperature batch reactor model for parameter estimation

    Parameters
    ----------
    temp_control_points: dict,
        Piecewise-linear profile of the reaction temperature (in R)

    Returns
    -------
    model: Pyomo model of the variable temperature batch reactor

    """
    model = pyo.ConcreteModel()

    # define sets
    reaction_number = [1, 2, 3]
    model.t = dae.ContinuousSet(bounds=[0, 3])  # hour

    # define the model parameters
    model.alpha_1 = pyo.Var(bounds=(0, None), )
    model.alpha_1.fix(23.80)

    model.alpha_2 = pyo.Var(bounds=(0, None), )
    model.alpha_2.fix(24.46)

    model.alpha_3 = pyo.Var(bounds=(0, None), )
    model.alpha_3.fix(35.61)

    model.E1 = pyo.Var(bounds=(0, None), )
    model.E1.fix(127.80)

    model.E2 = pyo.Var(bounds=(0, None), )
    model.E2.fix(125.98)

    model.E3 = pyo.Var(bounds=(0, None), )
    model.E3.fix(193.64)

    # add the mass fraction variables
    model.XA = pyo.Var(model.t, bounds=(0, 1), initialize=0.3)
    model.XB = pyo.Var(model.t, bounds=(0, 1), initialize=0.7)
    model.XC = pyo.Var(model.t, bounds=(0, 1), initialize=0)
    model.XE = pyo.Var(model.t, bounds=(0, 1), initialize=0)
    model.XP = pyo.Var(model.t, bounds=(0, 1), initialize=0)
    model.XG = pyo.Var(model.t, bounds=(0, 1), initialize=0)

    # add the temperature variables
    model.T_reparam = pyo.Var(model.t, bounds=[1/6.8, 1/5.8],)

    # add the rate constants
    model.k_reparam = pyo.Var(reaction_number, model.t, bounds=(0, None),)
    model.k = pyo.Var(reaction_number, model.t, bounds=(0, None),)

    # calculate the reparameterized rate constants
    def k_reparam_rule(m, i, t):
        if i == 1:
            return m.k_reparam[i, t] == model.alpha_1 - m.E1 * m.T_reparam[t]
        elif i == 2:
            return m.k_reparam[i, t] == model.alpha_2 - m.E2 * m.T_reparam[t]
        else:
            return m.k_reparam[i, t] == model.alpha_3 - m.E3 * m.T_reparam[t]

    model.k_reparam_eq = pyo.Constraint(
        reaction_number, model.t, rule=k_reparam_rule
    )

    # calculate the original rate constants
    def k_rule(m, i, t):
        return m.k[i, t] == pyo.exp(m.k_reparam[i, t])

    model.k_eq = pyo.Constraint(
        reaction_number, model.t, rule=k_rule
    )

    # add the differential equations for XA, XB, XC, XE, XP, and XG
    model.dXA = dae.DerivativeVar(model.XA, wrt=model.t)
    model.dXB = dae.DerivativeVar(model.XB, wrt=model.t)
    model.dXC = dae.DerivativeVar(model.XC, wrt=model.t)
    model.dXE = dae.DerivativeVar(model.XE, wrt=model.t)
    model.dXP = dae.DerivativeVar(model.XP, wrt=model.t)
    model.dXG = dae.DerivativeVar(model.XG, wrt=model.t)

    @model.Constraint(model.t)
    def xa_rate_ode(m, t):
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.dXA[t] == - m.k[1, t] * m.XA[t] * m.XB[t]

    @model.Constraint(model.t)
    def xb_rate_ode(m, t):
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.dXB[t] == - (m.k[1, t] * m.XA[t] * m.XB[t] + m.k[2, t] * m.XB[t] * m.XC[t])

    @model.Constraint(model.t)
    def xc_rate_ode(m, t):
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.dXC[t] == 2 * m.k[1, t] * m.XA[t] * m.XB[t] - 2 * m.k[2, t] * m.XB[t] * m.XC[t] - m.k[3, t] * m.XC[t] * m.XP[t]

    @model.Constraint(model.t)
    def xe_rate_ode(m, t):
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.dXE[t] == 2 * m.k[2, t] * m.XB[t] * m.XC[t]

    @model.Constraint(model.t)
    def xp_rate_ode(m, t):
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.dXP[t] == m.k[2, t] * m.XB[t] * m.XC[t] - 0.5 * m.k[3, t] * m.XC[t] * m.XP[t]

    @model.Constraint(model.t)
    def xg_rate_ode(m, t):
        if t == m.t.first():
            return pyo.Constraint.Skip
        return m.dXG[t] == 1.5 * m.k[3, t] * m.XC[t] * m.XP[t]

    # # add the mass fraction constraint
    # @model.Constraint(model.t)
    # def sum_mass_fraction(m, t):
    #     return m.XA[t] + m.XB[t] + m.XC[t] + m.XE[t] + m.XG[t] + m.XP[t] == 1

    # # fix the initial conditions
    t0 = model.t.first()
    model.XB_init = pyo.Constraint(expr=model.XB[t0] == 1 - model.XA[t0])
    model.XC[t0].fix(0.0)
    model.XE[t0].fix(0.0)
    model.XP[t0].fix(0.0)
    model.XG[t0].fix(0.0)

    # discretize the model
    disc = pyo.TransformationFactory("dae.finite_difference")
    disc.apply_to(model, nfe=60, scheme="BACKWARD")

    # initialize the temperature using the profile of the prior experiment
    for t in model.t:
        if t < 0.5:
            temp_at_time_t = temp_control_points["0.0"] + 0.2 * t

        elif t < 1.5:
            temp_at_time_t = temp_control_points["0.5"] + 0.4 * (t - 0.5)

        else:
            temp_at_time_t = temp_control_points["1.5"]

        model.T_reparam[t].set_value(1 / temp_at_time_t)

    return model


def main(doe_objective):
    """
    Solves the experimental design problem for XA(0) and T(t) for 0 <= t <= 3

    Parameters
    ----------
    doe_objective: str,
        Objective function for the experimental design formulation
    :param objective_option:
    :return:
    """

    prior_FIM = np.array([
        [1.642115e+04, 1.318845e+03, -1.527089e+03, -2.798949e+03, -2.228744e+02, 2.595965e+02],
        [1.318845e+03, 3.470710e+03, -2.317680e+02, -2.266803e+02, -5.883556e+02, 3.791954e+01],
        [-1.527089e+03, -2.317680e+02, 7.473955e+02, 2.597131e+02, 3.765902e+01, -1.255188e+02],
        [-2.798949e+03, -2.266803e+02, 2.597131e+02, 4.771091e+02, 3.831287e+01, -4.415367e+01],
        [-2.228744e+02, -5.883556e+02, 3.765902e+01, 3.831287e+01, 9.974374e+01, -6.155345e+00],
        [2.595965e+02, 3.791954e+01, -1.255188e+02, -4.415367e+01, -6.155345e+00, 2.108267e+01]
    ])

    temp_control = {
        "0.0": 5.8,
        "0.5": 5.9,
        "1.5": 6.3,
    }

    experiment = BatchReactorExperiment(
        temp_control=temp_control,
        const_temp=False,
        doe_solve=True,
    )

    doe_obj = DesignOfExperiments(
        experiment=experiment,
        fd_formula="backward",
        step=1e-2,
        use_grey_box_objective=True,
        objective_option=doe_objective,
        prior_FIM=prior_FIM,
        tee=True,
        grey_box_tee=True,
    )

    doe_obj.run_doe()

    solved_model = doe_obj.model.scenario_blocks[0]
    return solved_model


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    # define the experimental design objectives
    doe_objectives = ["trace", "determinant", "minimum_eigenvalue", "condition_number"]
    # doe_objectives = ["trace", "determinant"]

    for objective in doe_objectives:
        solved_model = main(objective)
        print("The optimal XA0 is:", pyo.value(solved_model.XA[0]))

        # Extract the time and temperature
        t_vals = list(solved_model.t)
        reparm_temp_vals = [pyo.value(solved_model.T_reparam[t]) for t in t_vals]
        XA_vals = [pyo.value(solved_model.XA[t]) for t in t_vals]
        XB_vals = [pyo.value(solved_model.XB[t]) for t in t_vals]
        XC_vals = [pyo.value(solved_model.XC[t]) for t in t_vals]
        XP_vals = [pyo.value(solved_model.XP[t]) for t in t_vals]
        XE_vals = [pyo.value(solved_model.XE[t]) for t in t_vals]
        XG_vals = [pyo.value(solved_model.XG[t]) for t in t_vals]

        # create a dataframe to store the results
        results_df = pd.DataFrame({
            "Time (hr)": t_vals,
            "Temperature (1/R)": reparm_temp_vals,
            "XA": XA_vals,
            "XB": XB_vals,
        })

        # save to a csv file
        results_df.to_csv(f".\{objective}_optimal_design_cond.csv", index=False)

        # get the temperature
        temp_actual = [1 / val for val in reparm_temp_vals]

        # Plot
        fig, ax1 = plt.subplots(figsize=(6.5, 5))

        ax1.plot(t_vals, XA_vals, label='X$_\mathbf{A}$', color='blue', linewidth=2)
        ax1.plot(t_vals, XB_vals, label='X$_\mathbf{B}$', color='orange', linewidth=2)
        ax1.plot(t_vals, XC_vals, label='X$_\mathbf{C}$', color='tab:cyan', linewidth=2)
        ax1.plot(t_vals, XP_vals, label='X$_\mathbf{P}$', color='green', linewidth=2)
        ax1.plot(t_vals, XE_vals, label='X$_\mathbf{E}$', color='purple', linewidth=2)
        ax1.plot(t_vals, XG_vals, label='X$_\mathbf{G}$', color='black', linewidth=2)

        ax1.set_xlabel('Time (hr)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Mass Fraction', fontsize=16, fontweight='bold')
        ax1.tick_params(axis='x', labelsize=16, direction="in", top=True, right=True)
        ax1.tick_params(axis='y', labelsize=16, direction="in", top=True, right=True)

        # set the secondary axis: mass fractions
        ax2 = ax1.twinx()
        # set the primary axis: temperature
        ax2.plot(t_vals, temp_actual, color='red', linewidth=3)
        ax2.set_ylabel('Temperature (R)', color='red', fontsize=16, fontweight='bold')
        ax2.tick_params(axis='y', labelsize=16, labelcolor='red', direction="in", top=True, right=True)

        # add legend
        plt.grid()
        plt.tight_layout()
        plt.savefig(f".\design_profile_{objective}.png")
        plt.show()

# if __name__ == "__main__":
#     # define the prior information (Fisher information matrix)
#     prior_FIM = np.array([
#         [1.64106296e+04, 1.30303771e+03, -1.50599157e+03, -2.79713263e+03, -2.20184400e+02, 2.55975519e+02],
#         [1.30303771e+03, 3.46262170e+03, -2.25070337e+02, -2.24084396e+02, -5.86688258e+02, 3.65194461e+01],
#         [-1.50599157e+03, -2.25070337e+02, 7.37528512e+02, 2.56253290e+02, 3.62941623e+01, -1.23642432e+02],
#         [-2.79713263e+03, -2.24084396e+02, 2.56253290e+02, 4.76796803e+02, 3.78710322e+01, -4.35593824e+01],
#         [-2.20184400e+02, -5.86688258e+02, 3.62941623e+01, 3.78710322e+01, 9.94125242e+01, -5.88060853e+00],
#         [2.55975519e+02, 3.65194461e+01, -1.23642432e+02, -4.35593824e+01, -5.88060853e+00, 2.07310442e+01]
#     ])
#
#     temp_control = {"0.0":5.8, "0.5":5.9, "1.0":6.2, "1.5":6.4, "2.0":6.6, "3.0":6.8}
#
#     # create an experiment object
#     experiment = BatchReactorExperiment(XA0=0.4, temp_control=temp_control,
#                                         const_temp=False, doe_solve=True)
#
#     # Use a central difference, with step size 1e-3
#     fd_formula = "central"
#     step_size = 1e-2
#
#     # Use the determinant objective
#     objective_option = "pseudo_trace"
#
#     # Create the DesignOfExperiments object
#     # doe_obj = DesignOfExperiments(
#     #     experiment,
#     #     fd_formula=fd_formula,
#     #     step=step_size,
#     #     objective_option=objective_option,
#     #     prior_FIM=prior_FIM,
#     #     tee=True,)
#
#     doe_obj = DesignOfExperiments(
#         experiment=experiment,
#         fd_formula=fd_formula,
#         step=step_size,
#         use_grey_box_objective=True,  # Comment out if normal
#         scale_constant_value=1,
#         scale_nominal_param_value=True,
#         objective_option=objective_option,
#         prior_FIM=prior_FIM,
#         tee=True,
#         grey_box_tee=True,
#         )
#
#     doe_obj.run_doe()
#
#     solved_model = doe_obj.model.scenario_blocks[0]
#
#     # Extract time and temperature
#     t_vals = list(solved_model.t)
#     T_vals = [pyo.value(solved_model.T_reparam[t]) for t in t_vals]
#     XA_vals = [pyo.value(solved_model.XA[t]) for t in t_vals]
#     XB_vals = [pyo.value(solved_model.XB[t]) for t in t_vals]
#     XP_vals = [pyo.value(solved_model.XP[t]) for t in t_vals]
#     XE_vals = [pyo.value(solved_model.XE[t]) for t in t_vals]
#     XG_vals = [pyo.value(solved_model.XG[t]) for t in t_vals]
#
#     # If you want actual temperature instead of reparameterized (1/T):
#     T_actual = [1 / val for val in T_vals]
#
#     # Plot
#     fig, ax1 = plt.subplots()
#
#     # set the primary axis: temperature
#     ax1.plot(t_vals, T_actual, color='red', linewidth=3)
#     ax1.set_xlabel('Time (hr)', fontsize=14, fontweight='bold')
#     ax1.set_ylabel('Temperature (R)', color='red', fontsize=14, fontweight='bold')
#     ax1.tick_params(axis='x', labelsize=14, direction="in", top=True, right=True)
#     ax1.tick_params(axis='y', labelsize=14, labelcolor='red', direction="in", top=True, right=True)
#
#     # set the secondary axis: mass fractions
#     ax2 = ax1.twinx()
#
#     ax2.plot(t_vals, XA_vals, label='XA', color='blue')
#     ax2.plot(t_vals, XB_vals, label='XB', color='orange')
#     ax2.plot(t_vals, XP_vals, label='XP', color='cyan')
#     ax2.plot(t_vals, XE_vals, label='XE', color='purple')
#     ax2.plot(t_vals, XG_vals, label='XG', color='brown')
#
#     ax2.set_ylabel('Mass Fraction', fontsize=14, fontweight='bold')
#     ax2.tick_params(axis='y', labelsize=14, direction="in", top=True, right=True)
#
#     # add legend
#     plt.grid()
#     plt.show()


# # Sensitivity analysis
# # Range to evaluate designs
# design_ranges_plot = {"XA0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,],
#                       "tf": [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25]}
#
# # store the FIM, initial mass fraction of species A, and final time for mass fraction measurements
# FIM_results = []
# XA0_container = []
# tf_container = []
#
# count = 0
# # Grid search
# for XA0 in design_ranges_plot["XA0"]:
#     for tf in design_ranges_plot["tf"]:
#         count += 1
#         print("=======Iteration Number: {} =======".format(count))
#         print("Design variable values for this iteration: (XA0: {}, tf: {})".format(XA0, tf))
#         # update flowrates
#         XA0_container.append(XA0)
#         tf_container.append(tf)
#
#         # create MembraneExperiment_plot object
#         experiment = BatchReactorExperiment(XA0=XA0, sample_time=tf, temp_control=temp_control,
#                                             const_temp=False, doe_solve=True)
#
#         # Use a central difference, with step size 1e-3
#         fd_formula = "central"
#         step_size = 1E-2
#
#         # Use the determinant objective
#         objective_option = "determinant"
#
#         # Create the DesignOfExperiments object
#         doe_obj = DesignOfExperiments(
#             experiment,
#             fd_formula=fd_formula,
#             step=step_size,
#             objective_option=objective_option,
#             prior_FIM=None,
#             tee=True,)
#
#         # Compute the full factorial design with the sequential FIM calculation
#         FIM_results.append(doe_obj.compute_FIM(method="sequential"))
#
# # Extract criteria from FIM
# def get_FIM_metrics(result):
#     """Computes A, D, E, and ME optimalities
#
#     Argument:
#         result: FIM of individual experiments
#
#     Returns:
#         A_opt, D_opt, E_opt, and ME_opt: A, D, E, and ME optimality, respectively"""
#
#     eigenvalues, eigenvectors = np.linalg.eig(result)  # eigen decomposition of FIM
#     min_eig = min(eigenvalues)
#
#     # compute optimalities
#     A_opt = np.log10(np.trace(result))
#     D_opt = np.log10(np.linalg.det(result))
#     E_opt = np.log10(min_eig)
#     ME_opt = np.log10(np.linalg.cond(result))
#
#     return A_opt, D_opt, E_opt, ME_opt
#
# # computing optimalities of experiments
# FIM_metrics = []
# for i in FIM_results:  # loop through FIM of experiments
#     FIM_metrics.append(get_FIM_metrics(i))
#
# FIM_metrics_np = np.asarray(FIM_metrics)
#
# # Make heat map
# def plot_heatmap(data, title, y_label, x_label, colorbar_label):
#     # set heatmap x,y ranges
#     x_tick_labels_0 = np.sort(np.unique(data[:, 0]))
#     y_tick_labels_0 = np.sort(np.unique(data[:, 1]))[::-1]
#
#     x_tick_labels = [round(x, 1) for x in x_tick_labels_0]
#     y_tick_labels = [round(y, 1) for y in y_tick_labels_0]
#
#     # optimality-values
#     opt_vals = np.asarray(data[:, 2]).reshape(len(x_tick_labels), len(y_tick_labels))
#
#     # Plot the colormap
#     fig = plt.figure()
#
#     # Plotting options
#     ax = fig.add_subplot(111)
#     params = {"mathtext.default": "regular"}
#     plt.rcParams.update(params)
#
#     # Plotting data
#     plt.title(title, fontsize=14, fontweight='bold')
#     ax.set_yticks(range(len(y_tick_labels)))
#     ax.set_yticklabels(y_tick_labels)
#     ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
#
#     ax.set_xticks(range(len(x_tick_labels)))
#     ax.set_xticklabels(x_tick_labels)
#     ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
#
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=9))
#     ax.yaxis.set_major_locator(MaxNLocator(nbins=9))
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#
#     im = ax.imshow(opt_vals.T, cmap=plt.cm.hot_r)
#     ba = plt.colorbar(im)
#     ba.set_label(colorbar_label, fontsize=14, fontweight='bold')
#     ba.ax.tick_params(labelsize=14)
#     plt.tight_layout()
#     plt.savefig(f"{colorbar_label}_10_by_10.png")
#     plt.show()
#
# # X and Y axis labels
# x_label = "Sample Collection Time (hr)"
# y_label = "Initial Mass Fraction of A"
#
# # Draw A-optimality figure
# data_A = np.zeros((len(FIM_metrics), 3))
# data_A[:, 0] = tf_container
# data_A[:, 1] = XA0_container
# data_A[:, 2] = FIM_metrics_np[:, 0]
#
# plot_heatmap(data_A, "A-optimality", y_label, x_label, "log10(trace(FIM))")
#
# # Draw D-optimality figure
# data_D = np.zeros((len(FIM_metrics), 3))
# data_D[:, 0] = tf_container
# data_D[:, 1] = XA0_container
# data_D[:, 2] = FIM_metrics_np[:, 1]
#
# plot_heatmap(data_D, "D-optimality", y_label, x_label, "log10(det(FIM))")
#
# # Draw E-optimality figure
# data_E = np.zeros((len(FIM_metrics), 3))
# data_E[:, 0] = tf_container
# data_E[:, 1] = XA0_container
# data_E[:, 2] = FIM_metrics_np[:, 2]
#
# plot_heatmap(data_E, "E-optimality", y_label, x_label, "log10(min-eig(FIM))")
#
# # Draw ME-optimality figure
# data_ME = np.zeros((len(FIM_metrics), 3))
# data_ME[:, 0] = tf_container
# data_ME[:, 1] = XA0_container
# data_ME[:, 2] = FIM_metrics_np[:, 3]
#
# plot_heatmap(data_ME, "ME-optimality", y_label, x_label, "log10(cond(FIM))")