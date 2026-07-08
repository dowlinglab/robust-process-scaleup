import pandas as pd
import numpy as np
import pyomo.contrib.parmest.parmest as parmest


def parameter_covariance_est(exp_list, const_temp=False):
    """Estimates the parameters and covariance matrix of the batch reactor
    model from the data

    Parameters
    ----------
    exp_list: list,
        List containing the BatchReactorExperiment objects
    const_temp: Boolean, optional
        Species if the batch reactor is a constant- or variable-temperature
        system. Default is False

    Returns
    -------
    obj: float or int,
        Value of the objective function at the estimated parameters
    theta: dict,
        Estimated values of the model parameters
    cov: pandas.DataFrame,
        Covariance matrix of the estimated model parameters
    """

    # create the Estimator object
    pest = parmest.Estimator(exp_list, obj_function="SSE_weighted", tee=True)

    # estimate the parameters
    obj, theta = pest.theta_est()

    # compute the covariance matrix of the estimated parameters
    try:
        if const_temp:
            cov = pest.cov_est(method="automatic_differentiation_kaug", step=1e-5)
        else:
            cov = pest.cov_est(method="automatic_differentiation_kaug")

        return obj, theta, cov
    except RuntimeError:
        return obj, theta, ["N/A"]


def ground_truth_data(measured_var):
    """Creates a pandas.DataFrame for the ground-truth
    simulation of mass fractions

    Parameters
    ----------
    measured_var: 2D numpy.array,
        Matrix containing the ground-truth simulation of
        mass fractions

    Returns
    -------
    outputs: pandas.DataFrame,
        Ground-truth simulation of mass fractions
    """

    # name of experimental outputs
    output_names = ["XA", "XB", "XC", "XP", "XE", "XG"]

    # outputs dataframe
    outputs = pd.DataFrame(measured_var, columns=output_names)

    print("\nThe output variables are:\n", outputs)

    return outputs


def generate_noisy_mass_fraction(ground_truth, species_cols, standard_dev, seed=10):
    """
    Adds Gaussian noise to the true values of the mass fractions

    Parameters
    ----------
    ground_truth: pandas.DataFrame,
        Dataframe containing the true values of the mass fractions
    species_cols: list,
        List containing the name of the species
    standard_dev: dict,
        Dictionary mapping species names to standard deviations
    seed: int, optional
        Random seed for reproducibility

    Returns
    -------
    noisy_mass_frac: pandas.DataFrame,
        Dataframe containing noisy mass fractions measurements
    """
    # generate random numbers
    random_number_gen = np.random.default_rng(seed)

    # copy the true mass fractions
    noisy_mass_frac = ground_truth.copy()

    # add a Gaussian noise
    for col in species_cols:
        # generate Gaussian noise
        noise = random_number_gen.normal(
            loc=0,
            scale=standard_dev[col],
            size=len(noisy_mass_frac)
        )

        # add noise to the ground truth values
        noisy_mass_frac[col] = noisy_mass_frac[col] + noise

    # enforce non-negative mass fractions
    noisy_mass_frac[species_cols] = noisy_mass_frac[species_cols].clip(lower=0)

    # get the sum of every row
    row_sums = noisy_mass_frac[species_cols].sum(axis=1)

    # divide every value in a row by its own row sum so they add to 1
    noisy_mass_frac[species_cols] = noisy_mass_frac[species_cols].div(
        row_sums,
        axis=0
    )

    return noisy_mass_frac


def synthetic_data_compiler(sample_time, mass_frac_data, csv_name):
    """Generates a csv file for the noisy mass fraction measurements

    Parameters
    ----------
    sample_time:
        Timepoints (in hr) that samples were collected for mass fraction measurements
    mass_frac_data: pandas.DataFrame,
        Dataframe containing the mass fraction measurements
    csv_name: str,
        File name for the csv

    Returns
    -------
    data: pandas.DataFrame,
        Noisy synthetic data that include the timepoints
    """
    # data columns
    column_names = ["Time (hr)", "XA", "XB", "XC", "XP", "XE", "XG"]

    # get the number of data columns
    n_columns = len(column_names)

    # get the number of time samples
    n_samples = len(sample_time)

    # define a matrix that contains the sample time and mass fractions
    data_matrix = np.zeros((n_samples, n_columns))
    data_matrix[:, 0] = sample_time
    data_matrix[:, 1] = mass_frac_data["XA"]
    data_matrix[:, 2] = mass_frac_data["XB"]
    data_matrix[:, 3] = mass_frac_data["XC"]
    data_matrix[:, 4] = mass_frac_data["XP"]
    data_matrix[:, 5] = mass_frac_data["XE"]
    data_matrix[:, 6] = mass_frac_data["XG"]

    # compile the full data
    data = pd.DataFrame(data_matrix, columns=column_names)
    print("The generated data is:\n", data)

    # save the full data to a csv file
    data.to_csv(f"{csv_name}.csv", index=False)

    return data
