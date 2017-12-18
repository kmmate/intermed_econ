"""
Examples on how to use intermedecon.py for multivariate time series
"""

import numpy as np
import statsmodels.api as sm
import statsmodels.tsa as tsa
import statsmodels.tsa.api as tsa_api
import matplotlib.pyplot as plt
import intermedecon as icon


"""
Granger causality
"""
print('\n\n==========================================================================================================\n'
      '----------------------------------------------- Granger causality -----------------------------------------\n')
# generate data for Granger causality test for (y, x, z)' vector series at h=1 forecast horizon
nobs = 1000
y, x, z = np.zeros(nobs + 2), np.zeros(nobs + 2), np.random.randn(nobs + 2)
phiL1_z = np.array([0, 0, 0.2])
phiL2_z = np.array([0, 0, 0.1])  # nothing Granger causes z
phiL1_x = np.array([0, 0.02, 0.31])
phiL2_x = np.array([0, 0.01, 0.001])  # z Granger causes x; y does not Granger cause x
phiL1_y = np.array([0.23, 0.15, 0])
phiL2_y = np.array([0.08, 0.20, 0])  # x Granger causes y; z does not Granger cause y
eps_y, eps_x, eps_z = np.random.randn(nobs + 2), np.random.randn(nobs + 2), np.random.randn(nobs + 2)  # i.i.d. shocks
for t in range(2, nobs + 2):
    y[t] = phiL1_y.dot([y[t - 1], x[t - 1], z[t - 1]]) + phiL2_y.dot([y[t - 2], x[t - 2], z[t - 2]]) + eps_y[t]
    x[t] = phiL1_x.dot([y[t - 1], x[t - 1], z[t - 1]]) + phiL2_x.dot([y[t - 2], x[t - 2], z[t - 2]]) + eps_x[t]
    z[t] = phiL1_z.dot([y[t - 1], x[t - 1], z[t - 1]]) + phiL2_z.dot([y[t - 2], x[t - 2], z[t - 2]]) + eps_z[t]
order = 3
autoorder = 'bic'
# test whether x Granger causes z
print('H_0: x does NOT cause z. Results: ',
      icon.grangercausality(grangercaused=z, grangercause=x, covariates=y, order=order, autoorder=autoorder))
# test whether z Granger causes x
print('H_0: z does NOT cause x. Results: ',
      icon.grangercausality(grangercaused=x, grangercause=z, covariates=y, order=order, autoorder=autoorder))
# test whether y Granger causes x
print('H_0: y does NOT cause x. Results: ',
      icon.grangercausality(grangercaused=x, grangercause=y, covariates=z, order=order, autoorder=autoorder))
# test whether x Granger causes y
print('H_0: x does NOT cause y. Results: ',
      icon.grangercausality(grangercaused=y, grangercause=x, covariates=z,order=order, autoorder=autoorder))

"""
Cointegration
"""
print('\n\n==========================================================================================================\n'
      '----------------------------------------------- Cointegration -----------------------------------------\n')
# generate two I(1) process with one common stochastic trend
nobs = 500
w = np.zeros(shape=(nobs + 1, ))
for t in range(1, nobs + 1):
    w[t] = w[t - 1] + np.random.randn(1)  # common stochastic trend: random walk
beta = 3
constant = 4.5
y = [beta * wt + np.random.randn(1) for wt in w]
x = [constant + wt + np.random.randn(1) for wt in w]  # cointegrating vector for (y,x)' is (1,-beta)'
# augmented Engle-Granger test for cointegration (aeg). H_0: no cointegration,
coint_t, coint_pvalue, coint_critvalue = tsa.stattools.coint(y, x, trend='c', method='aeg',
                                                             maxlag=4, autolag='BIC', return_results=True)
print('\n Cointegration results. H_0: . t-stat: %.4f, p-value: %.4f,' % (coint_t, coint_pvalue),
      'critical values: \n', coint_critvalue, '\n test type: "c"')