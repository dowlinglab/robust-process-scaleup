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

    def __init__(self, const_temp, XA0=None, data=None, temp_control=None, doe_solve=False,):

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
            self.model = reform_var_temp_doe_model()

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
                (m.XA[t], 0.001) for t in m.t
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
                (m.XA[t], 0.001) for t in meas_time_points
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


def reform_var_temp_doe_model():
    """
    Reformulates the variable temperature batch reactor model for parameter estimation

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
    model.alpha_1.fix(20.56)

    model.alpha_2 = pyo.Var(bounds=(0, None), )
    model.alpha_2.fix(9.85)

    model.alpha_3 = pyo.Var(bounds=(0, None), )
    model.alpha_3.fix(18.63)

    model.E1 = pyo.Var(bounds=(0, None), )
    model.E1.fix(108.83)

    model.E2 = pyo.Var(bounds=(0, None), )
    model.E2.fix(41.71)

    model.E3 = pyo.Var(bounds=(0, None), )
    model.E3.fix(94.43)

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

    # # fix the initial conditions
    t0 = model.t.first()
    model.XB_init = pyo.Constraint(expr=model.XB[t0] == 1 - model.XA[t0])
    model.XC[t0].fix(0.0)
    model.XE[t0].fix(0.0)
    model.XP[t0].fix(0.0)
    model.XG[t0].fix(0.0)

    # discretize the model
    disc = pyo.TransformationFactory("dae.finite_difference")
    disc.apply_to(model, nfe=90, scheme="BACKWARD")

    # initialize the temperature
    for t in model.t:
        model.T_reparam[t].set_value(1 / 6.3)

    # define the maximum temperature ramp rate:
    # 2 F/min = 120 R/hr
    max_temp_ramp = 1.2

    # define the temperature ramp-up and ramp-down rule
    def temp_ramp_up_rule(m, t):
        if t == m.t.first():
            return pyo.Constraint.Skip

        # get the previous time point
        t_prev = m.t.prev(t)

        # compute the time interval
        delta_t = t - t_prev

        # get the current and previous temperatures
        T_now = 1 / m.T_reparam[t]
        T_prev = 1 / m.T_reparam[t_prev]

        # limit the rate of temperature change between consecutive time points
        return T_now - T_prev <= max_temp_ramp * delta_t

    model.temp_ramp_up = pyo.Constraint(model.t, rule=temp_ramp_up_rule)

    def temp_ramp_down_rule(m, t):
        if t == m.t.first():
            return pyo.Constraint.Skip

        # get the previous time point
        t_prev = m.t.prev(t)

        # compute the time interval
        delta_t = t - t_prev

        # get the current and previous temperatures
        T_now = 1 / m.T_reparam[t]
        T_prev = 1 / m.T_reparam[t_prev]

        # limit the rate of temperature change between consecutive time points
        return T_prev - T_now <= max_temp_ramp * delta_t

    model.temp_ramp_down = pyo.Constraint(model.t, rule=temp_ramp_down_rule)

    return model


def main(doe_objective):
    """
    Solves the experimental design problem for XA(0) and T(t) for 0 <= t <= 3

    Parameters
    ----------
    doe_objective: str,
        Objective function for the experimental design formulation

    Returns
    -------
    solved_model: pyomo.ConcreteModel(),
        The solved Pyomo model from optimal experimental design
    """

    # define the FIM from the prior constant-temperature experiment
    prior_FIM = np.array([
        [50.708559, -360.895607, 365.820701, -7.512379, 53.466016, -54.195659],
        [-360.895607, 3428.341623, -3426.526122, 53.466016, -507.902463, 507.633500],
        [365.820701, -3426.526122,  3428.757428, -54.195659, 507.633500, -507.964063],
        [-7.512379, 53.466016, -54.195659, 1.112945, -7.920891, 8.028987],
        [53.466016, -507.902463, 507.633500, -7.920891, 75.244809, -75.204963],
        [-54.195659, 507.633500, -507.964063, 8.028987, -75.204963, 75.253935]
    ])

    experiment = BatchReactorExperiment(
        const_temp=False,
        doe_solve=True,
    )

    doe_obj = DesignOfExperiments(
        experiment=experiment,
        fd_formula="forward",
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

    # define the experimental design objective
    doe_objectives = ["condition_number"]

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
        results_df.to_csv(f".\{objective}_optimal_design.csv", index=False)

        # get the temperature
        temp_actual = [100 * (1 / val) for val in reparm_temp_vals]

        # Plot
        fig, ax1 = plt.subplots(figsize=(6.5, 5))

        # set the primary axis: mass fractions
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

        # set the secondary axis: temperature
        ax2 = ax1.twinx()
        ax2.plot(t_vals, temp_actual, "--", color='red', linewidth=3)
        ax2.set_ylabel(r'Temperature ($\mathbf{^{\circ}}$R)', color='red', fontsize=16, fontweight='bold')
        ax2.tick_params(axis='y', labelsize=16, labelcolor='red', direction="in", top=True, right=True)

        # add legend
        plt.grid()
        plt.tight_layout()
        plt.savefig(f".\design_profile_{objective}.png")
        plt.show()