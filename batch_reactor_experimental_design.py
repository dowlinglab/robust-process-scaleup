import pyomo.environ as pyo
import idaes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyomo.contrib.doe import DesignOfExperiments

from batch_experiment import BatchReactorExperiment


def main(doe_objective):
    """
    Solves the optimal experimental design (OED) problem for XA(t=0)
    and T(t) for 0 <= t <= 3

    Parameters
    ----------
    doe_objective: str,
        Objective function for the OED formulation

    Returns
    -------
    solved_model: pyomo.ConcreteModel(),
        The solved Pyomo model from OED
    """

    # define the FIM from the prior high-temperature experiment
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