from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
import pyomo.contrib.parmest.parmest as parmest
from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import RooneyBieglerExperiment
from scipy.stats import chi2


data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=["hour", "y"],
        )

# Create an experiment list
exp_list = []
for i in range(data.shape[0]):
    exp_list.append(
        RooneyBieglerExperiment(data.loc[i, :],)
    )
pest = parmest.Estimator(
    exp_list, obj_function="SSE", tee=True
)

SSE_min, theta_hat = pest.theta_est()
cov = pest.cov_est()

print("Estimated parameters:", theta_hat)
print("Covariance matrix of the estimated parameters:", cov)

theta1_grid = np.linspace(10, 40, 50)
theta2_grid = np.linspace(0.05, 2.5, 50)
theta_names = ['asymptote', 'rate_constant']

THETA1, THETA2 = np.meshgrid(theta1_grid, theta2_grid)

SSE_grid = np.zeros_like(THETA1)

for i in range(THETA1.shape[0]):
    for j in range(THETA1.shape[1]):
        theta = np.array([THETA1[i, j], THETA2[i, j]])
        theta_dict = {'asymptote':[theta[0]], 'rate_constant':[theta[1]]}

        theta_df = pd.DataFrame(theta_dict)
        print("Current parameters:", theta_df)

        obj_df = pest.objective_at_theta(theta_df)
        print("Objective at the current parameters:", obj_df)

        SSE_grid[i, j] = obj_df["obj"].iloc[0]

# -----------------------------
# 5. Likelihood-ratio statistic
# -----------------------------
n = 1
# LR_grid = n * np.log(SSE_grid / SSE_min)
LR_grid = n * SSE_grid / SSE_min

# -----------------------------
# 6. LR cutoff
# -----------------------------
confidence = 0.80
p = 2

cutoff = chi2.ppf(confidence, df=p)

print("LR cutoff =", cutoff)

# -----------------------------
# 7. Plot LR confidence region
# -----------------------------
plt.figure(figsize=(7, 5))

plt.contour(
    THETA1,
    THETA2,
    LR_grid,
    levels=[cutoff],
    colors="black",
    linewidths=2
)

plt.contourf(
    THETA1,
    THETA2,
    LR_grid,
    levels=[0, cutoff],
    alpha=0.3
)

plt.scatter(
    theta_hat['asymptote'],
    theta_hat['rate_constant'],
    color="red",
    marker="x",
    s=100,
    label="Best estimate"
)

plt.xlabel(r"Asymptote")
plt.ylabel(r"Rate constant")
plt.title("Likelihood-Ratio Confidence Region")
plt.legend()
plt.show()