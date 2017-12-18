"""
Examples on how to use intermedecon.py for univariate time series
"""

import numpy as np
import statsmodels.api as sm
import statsmodels.tsa as tsa
import statsmodels.tsa.api as tsa_api
import matplotlib.pyplot as plt
import intermedecon as icon


"""
ARMA models
"""
print('\n ===========================================================================================================\n'
      '------------------------------------------------- ARMA models -----------------------------------------------\n')
# model with known coeffs
phi = [-0.58, 0.6]
arma1 = icon.ARMA(phi_coeffs=phi)
print('We have just created the ARMA model: ', arma1.display())
print('Statement: the model is covariance stationary. Answer: ', arma1.isstable())
print('The model up to lag %d has the impulse response: ' % 3, arma1.irf(j_max=3))

# generate an AR(2) series
nobs = 3000
cons = 2
np.random.seed(2017)
y = np.zeros(shape=(nobs + 2, ))  # initial values
for t in range(2, nobs + 1):
    y[t] = cons + phi[0] * y[t - 2] + phi[1] * y[t -1] + np.random.randn(1)  # i.i.d. shocks, e
f, ax = plt.subplots()
ax.plot(y)
plt.show(block=False)
# built-in method
res = tsa.stattools.arma_order_select_ic(y, ic=['aic', 'bic'], trend='c')
print('Using built-in statsmodels function the optimal AIC order: ', res.aic_min_order)
print('Using built-in statsmodels function the optimal BIC order: ', res.bic_min_order)
# icon method
arma2 = icon.ARMA(endog=y)
opt_model = arma2.estimate(order={'ar': 4, 'ma': 5}, include_constant='c' ,autoorder='bic', verbose=True)
print('Using own function the BIC optimal model: ', opt_model.summary(alpha=0.05))

"""
Unit root process
"""
print('\n\n =========================================================================================================\n'
      '--------------------------------- Unit root processes -------------------------------------------------------\n')
# simulate a random walk
nobs = 500
y = np.zeros(shape=(nobs + 1, ))
for t in range(1, nobs + 1):
    y[t] = y[t - 1] + np.random.randn(1)
# testing for unit root
adftest = tsa.stattools.adfuller(y, maxlag=12, autolag='bic' ,regression='nc')
print('ADF test of y. Results: adf_stat: %.4f, p-value: %.4f, #lags used: %d, test type: "nc"' % adftest[0:3])
# try to estimate it, but do unit-root testing before
arma3 = icon.ARMA(endog=y)
opt_model = arma3.estimate(order={'ar': 1, 'ma': 0}, include_constant='nc', autoorder='off', pretest_unitroot=False,
               pretest_maxlag=3,verbose=True)
print('Using own function the BIC optimal model: ', opt_model.summary(alpha=0.05))
