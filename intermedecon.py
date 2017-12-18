"""
Module for Intermediate Econometrics course, CEU.

Features: ARMA modelling, VAR, unit root testing, cointegration test. Largely based on statsmodels package.
"""

import numpy as np
import statsmodels.tsa as tsa
import statsmodels.tsa.api as tsa_api
import scipy.stats as sp_stat
import warnings
from scipy.ndimage.interpolation import shift as sp_shift

__author__ = 'Mate Kormos'
__credits__ = 'Prof. Robert P. Lieli'

"""
=====================================================================================================================
Univariate time series
=====================================================================================================================
"""


class ARMA:
    """
    ARMA models of the form
    Y_t = c + phi_1 * y(t-1) + ... + phi_p * y(t-p) + theta_1 * e(t-1) + ... + theta_q * e(t-q) + e_t,
    with e_t white noise(0, sigma^2), phi_p != 0.
    """
    def __init__(self, endog=None, phi_coeffs=None, theta_coeffs=None, constant=None):
        """
        Setting up ARMA class.

        Either coefficients have to be specified on declaration, or data have to be given to estimate them.

        Parameters
        ----------
        :param endog: array-like, optional
                    data for estimation
        :param phi_coeffs: array-like, optional
                    coefficients on lagged y, phi[0]=phi_p,...,phi[p-1]=phi_1
        :param theta_coeffs: array-like, optional
                    coefficients on lagged e, theta[0]=theta_q,...,theta[q-1]=theta_1
        :param constant: float, optional
                    constant in the model

        Returns
        -------
        :return ARMA class instance
        """

        if endog is None and phi_coeffs is None and theta_coeffs is None and constant is None:
            raise ValueError('To initialise an ARMA class, either endog data or at least one of (phi_coeffs,'
                             ' theta_coeff, constant) has to be given.')
        self.ydata = endog
        self.phi = phi_coeffs
        self.theta = theta_coeffs
        self.con = constant
        if phi_coeffs is not None:
            self.p = len(phi_coeffs)
        else:
            self.p = 0
        if theta_coeffs is not None:
            self.q = len(theta_coeffs)
        else:
            self.q = 0
        if constant is not None:
            self.c = 1
        else:
            self.c = 0

    def display(self):
        """
        A visual representation of the model

        Returns
        -------
        :return: a string representation of the model
        """
        if self.ydata is not None:  # data given
            model_string_representation = 'ARMA class instance, initialised with data.'
        else:  # coeffs given
            if self.con is not None:
                con_str = str(self.con)
            else:
                con_str = ''
            if self.phi is not None:
                ar_str = '+'.join([str(self.phi[i]) + '*Y(t-' + str(self.p - i) + ')' for i in range(0, self.p)
                                   if self.phi[i] != 0])
            else:
                ar_str = ''
            if self.theta is not None:
                ma_str = '+'.join([str(self.theta[i]) + '*e(t-' + str(self.q - i) + ')' for i in range(0, self.q)
                                   if self.theta[i] != 0])
            else:
                ma_str = ''
            model_string_representation = ('Y(t) = ' + con_str + '+' + ar_str + '+' + ma_str +\
            '+e(t)').replace('++', '+').replace('+-', '-')
        return model_string_representation

    """
    Autoregressive properties
    """

    def ar_roots(self):
        """
        Roots of the characteristic polynomial of the autoregressive part

        Returns
        -------
        :return: the ar_roots of the characteristic polynomial
        """
        if self.phi is None:
            raise ValueError('phi_coeffs has to be given to compute AR roots.')
        else:
            polynomial = [-i for i in self.phi]
            polynomial.append(1)
            return np.roots(polynomial)

    def ar_lamba(self):
        """
        Returns the reciprocal of the ar_roots of the characteristic polynomial

        Returns
        -------
        :return: lambda_i=1/z_i
        """
        if self.phi is None:
            raise ValueError('phi_coeffs has to be given to compute AR lambdas.')
        else:
            roots = self.ar_roots()
            return [1 / root for root in roots]

    def isstable(self):
        """
        Checks whether the ARMA process is stable, provided that e_t is bounded.

        Returns
        -------
        :return: boolean
                    True iff stable
        """
        if self.phi is None:
            raise ValueError('phi_coeffs has to be given to check stability.')
        else:
            roots = self.ar_roots()
        return all(abs(root) > 1 for root in roots)

    """
    Moving average properties
    """

    def ma_roots(self):
        """
        Roots of the characteristic polynomial of the moving average part

        Returns
        -------
        :return: the ma_roots of the characteristic polynomial
        """
        if self.theta is None:
            raise ValueError('theta_coeffs has to be given to compute MA roots.')
        else:
            polynomial = [i for i in self.theta]
            polynomial.append(1)
        return np.roots(polynomial)

    def isinvertible(self):
        """
        Checks whether the ARMA process is invertible by means of MA(q), provided that e_t is bounded.

        Returns
        -------
        :return: boolean
                    True iff invertible
        """
        if self.theta is None:
            raise ValueError('theta_coeffs has to be given to check invertibility in MA-sense.')
        else:
            roots = self.ma_roots()
        return all(abs(root) > 1 for root in roots)

    """
    ARMA properties
    """

    def irf(self, j_max):
        """
        Impulse response function \partial y_{t+j} / (\partial e_{t}) for j=0,...,j_max

        Parameters
        ----------
        :param j_max: integer
                \partial y_{t+j} / (\partial e_{t}) for j=0,...,j_max

        Returns
        -------
        :return: array
                    of length j_max+1, with first element corresponding to j=0,...
        """
        if self.phi is None and self.theta is None and self.con is None:
            return ValueError('At least one of (phi_coeffs, theta_coeffs, constant) hos to be given to compute IRF.')
        elif self.phi is None and self.theta is None:
            return np.ones(shape=(j_max + 1, 1))
        else:
            # initialise y and e
            starttime = max(self.p, self.q)
            e = [0 for _ in range(0, starttime)]
            e.append(1)  # impulse
            dy = [0 for _ in range(0, starttime)]
            dy.append(1)
            # propagate impulse through time
            time = starttime + 1
            while time <= starttime + j_max:
                e.append(0)
                if self.theta is None:  # moving average component
                    ma_t = 0
                else:
                    ma_t = np.array(self.theta).dot(np.array(e[time - self.q:time])) + e[time]
                if self.phi is None:  # autoregressive component
                    ar_t = 0
                else:
                    ar_t = np.array(self.phi).dot(np.array(dy[time - self.p:time]))
                dy.append(ar_t + ma_t)
                time += 1
            return dy[starttime:starttime + j_max + 1]

    """
    Estimation
    """
    def estimate(self, order, autoorder='off', include_constant='c', method='css', pretest_unitroot=False,
                 pretest_maxlag=4, verbose=False):
        """
        ARMA parameter estimation

        Parameters
        ----------
        :param order: dictionary
                    with structure {'ar:' number_of_ar_lags, 'ma': number_of_ma_lags}, where number_of_lags is
                    integer.
        :param autoorder: string {'off', 'bic', 'aic'}
                    information criteria based on which order selection is done. Selection is done with
                    grid-search, starting downwards from the lags specified in 'order'. The selected model is
                    which minimises the info criterion 'bic' or 'aic'. If 'off', no automatic
                    model selection is carried out: the model is estimated with the lags given in 'order'.
        :param include_constant: string {'c', 'nc'}
                    if 'c', a constant is included in the regression; if 'nc', no constant is included
        :param method: string {'css-mle','mle','css'}
                    estimator. See http://www.statsmodels.org/stable/generated/
                    /statsmodels.tsa.arima_model.ARMA.fit.html#statsmodels.tsa.arima_model.ARMA.fit
        :param pretest_unitroot: boolean
                    iff True, tests the series with ADF for unit root, prints information
        :param pretest_maxlag: integer
                    maximum number of lags, from where downwards the ADF test automatically selects the optimal lag to
                    include in the ADF test, based on BIC
        :param verbose: boolean
                    if True, convergence and infro critarion information is printed, otherwise not
        Returns
        -------
        :return: statsmodels.tsa.arima_model.ARMAResults class
        """
        if self.ydata is None:
            raise ValueError('Data have to be given for estimation.')
        if pretest_unitroot:  # ADF unit root tests, check whether the series is I(0)
            pvalues = dict()
            for test_type in ['nc', 'c', 'ct']:
                adf = tsa.stattools.adfuller(self.ydata, maxlag=pretest_maxlag, regression=test_type, autolag='BIC')
                pvalues[test_type] = adf[1]
            if np.any([pvalues[pvalue] > 0.05 for pvalue in pvalues]):
                warnings.warn('The series may have unit root. Consider differentiating or other transformation to '
                              'achieve stationarity. Under unit root, the estimator properties like standard errors,'
                              ' and forecasting are not reliable.\n '
                              'statsmodels ADF test pvalues of H_0: "series is unit root" for different specifications '
                              '(for details see statsmodels documentation) are "nc": %.4f, "c": %.4f, "ct": %.4f'
                              % (pvalues['nc'], pvalues['c'], pvalues['ct']))
        if autoorder == 'off':  # fit model with specified lags
            arma_opt = tsa.arima_model.ARIMA(endog=self.ydata, order=(order['ar'], 0, order['ma']))\
                .fit(trend=include_constant, method=method, maxiter=100, disp=-1)
        else:  # grid-search for optimal lag values
            p_max, q_max = order['ar'], order['ma']
            p_opt, q_opt = 0, 0
            best_ic = np.inf
            np.random.seed([0])
            if verbose:
                print('\n----------------------- Automatic lag selection -----------------------------\n ')
            for p in range(p_max + 1):
                for q in range(q_max + 1):
                    try:
                        arma = tsa.arima_model.ARIMA(endog=self.ydata, order=(p, 0, q))\
                            .fit(trend=include_constant, method=method, maxiter=100, disp=-1)
                        arma_ic = arma.bic if autoorder == 'bic' else arma.aic
                        if verbose:
                            print(autoorder.upper() + '(p=%d, q=%d) = %.6f.' % (p, q, arma_ic))
                    except ValueError:
                        print('The model could not be fitted for p=%d and q=%d.' % (p, q))
                        arma_ic = np.inf
                        continue  # increase q
                    if arma_ic < best_ic:
                        best_ic = arma_ic
                        p_opt = p
                        q_opt = q
            print('---------------------------------------- Automatic lag selection results: ')
            print('The optimal lags are p_opt=%d, q_opt=%d' % (p_opt, q_opt))
            arma_opt = tsa.arima_model.ARIMA(endog=self.ydata, order=(p_opt, 0, q_opt))\
                .fit(trend=include_constant, method=method, maxiter=100, disp=-1)
        return arma_opt


"""
======================================================================================================================
Multivariate time series
======================================================================================================================
"""

"""
Functions for Granger causality
"""


def get_optimallag(endog, maxlags=None, criterion='bic'):
    """
    Finds the optimal lag order of a vector time series, using an info criterion

    Parameters
    ----------
    :param endog: array-like
                2-d array, vector time series
    :param maxlags: integer
                 the maximum number of lags, from which the automatic lag selection starts downwards.
                  if None, defaults to 12 * (nobs/100.)**(1./4)
    :param criterion: {'bic', 'aic', 'hqic'}
                information criterion
    Returns
    -------
    :return: int
                The optimal lag order
    """

    var = tsa.vector_ar.var_model.VAR(endog=endog)
    optimallag = var.select_order(maxlags=maxlags, verbose=False)
    return optimallag[criterion]


def ftest(ssr_ur, ssr_r, n, k, q, alpha=0.05):
    """
    F-test results

    Parameters
    ----------
    :param ssr_ur: float
                sum of squared residuals from the unrestricted model
    :param ssr_r: float
                sum of squared residuals from the restricted model
    :param n: integer
                sample size
    :param k: integer
                number of estimated parameters
    :param q: integer
                number of restrictions
    :param alpha: float in [0,1]
                significance level for the critical value
    Returns
    -------
    :return: dictionary
                with entries 'fstat', 'pvalue', 'critvalue' corresponding to the value of the F-statistic,
                the p-value of the test, and the critical value at the given significance level
    """

    dof1 = q
    dof2 = n - k
    fstat = ((ssr_r - ssr_ur) / dof1) / (ssr_ur / dof2)
    pvalue = 1 - sp_stat.f.cdf(fstat, dof1, dof2)
    critvalue = sp_stat.f.ppf(1 - alpha, dof1, dof2)
    fresults = {'fstat': fstat, 'pvalue': pvalue, 'critvalue': critvalue}
    return fresults


def ols_residuals(endog, exog):
    """
    Obtains OLS residuals from running endog on exog

    Parameters
    ----------
    :param endog: array-like
                      1-d array of dependent variable
    :param exog:  array-like
                      2-d array of independent variables
    Returns
    -------
    :return: array
                1-d array of residuals
    """
    xtx_inv = np.linalg.inv(exog.T.dot(exog))
    yhat = exog.dot(xtx_inv).dot(exog.T).dot(endog)
    residuals = endog - yhat
    return residuals


def lagmatrix(x, lags):
    """
    Returns a matrix of lagged arrays

    Parameters
    ----------
    :param x: array-like
                  array to lag. first observation is the earliest
    :param lags: array-like or range with numbers >=0
                  the lags to create
    Returns
    -------
    :return: array-like
                  2-d array of the lagged values
    """

    lags_number = len(lags)
    n = len(x)
    xa = np.asarray(x)
    lagged_matrix = np.zeros(shape=(n, lags_number))
    for col, lag in zip(range(lags_number), lags):
        lagged_matrix[:, col] = sp_shift(xa, lag, cval=np.NaN)
    return lagged_matrix


def grangercausality(grangercaused, grangercause, order, covariates=None, autoorder='bic', alpha=0.05):
    """
    Tests whether a time series Granger causes another at h=1 forecast horizon.

    The test is the sum of squared residuals based F-test. The null hypothesis is that grangercause does NOT
    Granger cause grangercaused.

    The test-statistics under the null hypothesis follows an
    F(dof1, dof2) distribution, where dof1=p, the number of lags included, and dof2=T - (1 + (2 + d) * p).

    Parameters
    ----------
    :param grangercaused: array-like
                              array of size (T, 1). time series which may be Granger caused by grangercause
    :param grangercause: array-like
                              array of size (T, 1). time series which may Granger cause grangercaused
    :param order: integer
                    number of lags to include in the VAR model. If autoorder='off', the model is estimated exactly with
                    the specified order. If autoorder!='off', then order is the largest number of lags from where
                    downwards automatic lag selection is done based on the info criterion specified by autoorder.
    :param covariates:  array-like
                              array of size (T, d), d>=1. other potntial time series, the lags of which shows up in the
                              equation of grangercaused
    :param autoorder: {'bic', 'aic', 'hqic', 'off'}
                        information criterion based on which automatic lag-selection is done. If 'off', the model is
                        estimated exactly with the number of lags specified by order. If not 'off', then automatic lag
                        selection is done downwards from order, based on the info criterion specified by autoorder.
    :param alpha: float in [0, 1]
                        significance level to return the corresponding critical value of the F distribution

    Returns
    -------
    :return: dictionary
                with entries 'fstat', 'pvalue', 'critvalue' corresponding to the value of the F-statistic,
                the p-value of the test, and the critical value at the given significance level

    Notes
    -----
    When bivariate Granger causality is tested (i.e. covariates=None) with this function at h=1 forecast horizon,
     then not rejecting H_0: no Granger causality implies that there is no Granger causality at h>=1 forecast horizons.

    However, when a third series is present (i.e. covariates != None), then not rejecting H_0: no Granger causality
    does not necassarily imply that there is no Granger causality at higher h>=1 forecast horizons.
    """
    T = len(grangercause)
    y = np.asarray(grangercaused)
    x = np.asarray(grangercause)
    if covariates is not None:  # other covariates are given
        z = np.asarray(covariates)
        try:  # z has more than one time series
            all_series = np.hstack((y[:, None], x[:, None], z))
            d = np.size(z, axis=1)
        except ValueError:  # z has one time series
            all_series = np.hstack((y[:, None], x[:, None], z[:, None]))
            d = 1
    else:  # no other covariates
        all_series = np.hstack((y[:, None], x[:, None]))
        d = 0

    if autoorder == 'off':  # no automatic lag selection
        optimal_lag = order
    else:  # automatic lag selection
        optimal_lag = get_optimallag(endog=all_series, maxlags=order, criterion=autoorder)
    print('Based on criterion "%s" the optimal largest lag to include is %d.' % (autoorder, optimal_lag))
    # constructing design matrices with lags
    ylags = lagmatrix(y, range(1, optimal_lag + 1))[optimal_lag:, :]
    xlags = lagmatrix(x, range(1, optimal_lag + 1))[optimal_lag:, :]
    constant = np.ones(shape=(T - optimal_lag, 1))
    if covariates is not None:
        zlags = lagmatrix(z, range(1, optimal_lag + 1))[optimal_lag:, :]
        regressors_unrestricted = np.hstack((constant, ylags, xlags, zlags))
        regressors_restricted = np.hstack((constant, ylags, zlags))
    else:
        regressors_unrestricted = np.hstack((constant, ylags, xlags))
        regressors_restricted = np.hstack((constant, ylags))

    # obtain residuals from (un)restricted regressions
    residuals_unrestricted = ols_residuals(endog=y[optimal_lag:], exog=regressors_unrestricted)
    residuals_restricted = ols_residuals(endog=y[optimal_lag:], exog=regressors_restricted)
    ssr_unrestricted = residuals_unrestricted.T.dot(residuals_unrestricted)
    ssr_restricted = residuals_restricted.T.dot(residuals_restricted)

    # ftest
    granger_testresult = ftest(ssr_ur=ssr_unrestricted, ssr_r=ssr_restricted, n=T, k=1 + (2 + d) * optimal_lag,
                               q=optimal_lag, alpha=alpha)
    return granger_testresult