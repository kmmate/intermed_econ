"""
A Monte Carlo simulation to compute size distortion in hypothesis testing in an OLS context
"""

import numpy as np
import matplotlib as plt

# generate artificial dataset (could use real data if available)
np.random.seed(0)
n = 4  # the higher is n, the less wrong we are about using +/-1.96 as critical values,
# the closer is the empirical size to 5%
sigma_u = 30
beta = 7
X = np.random.normal(0, 10, n).reshape((n, 1))
u = np.random.normal(0, sigma_u, n).reshape((n, 1))
y = X.dot(beta) + u

# monte carlo simulation
beta_zero = 5
XX_inv = np.linalg.inv(np.transpose(X).dot(X))
A = XX_inv.dot(np.transpose(X))
b = X.dot(beta_zero)
number_of_rejections = 0
reps = 100000
for m in range(1, reps + 1):
    u_m = np.random.normal(0, sigma_u, n).reshape((n, 1))
    y_m = b + u_m
    betahat_m = A.dot(y_m)
    uhat = y_m - X.dot(betahat_m)
    s2_m = np.transpose(uhat).dot(uhat) / (n - 1)
    t_m = (betahat_m - beta_zero) / np.sqrt(s2_m * XX_inv)
    number_of_rejections += abs(t_m) > 1.96
empirical_size = number_of_rejections / reps
print('The empirical size is %.10f' % empirical_size)
print('The size distortion is |theoretical-empirical|=|5%%-%.4f%%|=%.4f' % (empirical_size * 100,
                                                                            abs(5 - empirical_size * 100)))
