"""
Intermediate Econometrics Problem set 1 (due: 31 October 2017)
"""

import numpy as np


class ARMA:
    """
    ARMA models of the form
    Y_t = c + phi_1 * y(t-1) + ... + phi_p * y(t-p) + theta_1 * e(t-1) + ... + theta_q * e(t-q) + e_t,
    with e_t white noise(0, sigma^2).
    """
    def __init__(self, phi_coeffs, theta_coeffs=[0.0], constant=[0.0]):
        """
        Setting up ARMA class
        :param phi_coeffs: coefficients on lagged y, array-like. phi[0]=phi_p,...,phi[p-1]=phi_1
        :param theta_coeffs: coefficients on lagged e, array-like. theta[0]=theta_q,...,theta[q-1]=theta_1
        :param constant: constant in the model
        """
        self.phi = phi_coeffs
        self.p = len(phi_coeffs)
        self.theta = theta_coeffs
        self.q = len(theta_coeffs)
        self.c = constant

    def display(self):
        """
        Prints out a visual representation of the model
        :return:
        """
        c_str = str(self.c[0])
        ar_str = '+'.join([str(self.phi[i]) + '*Y(t-' + str(self.p - i) + ')' for i in range(0, self.p)])
        ma_str = '+'.join([str(self.theta[i]) + '*e(t-' + str(self.q - i) + ')' for i in range(0, self.q)]) + '+e(t)'
        return 'Y(t) = ' + c_str + '+' + ar_str + '+' + ma_str

    """
    Autoregressive properties
    """

    def ar_roots(self):
        """
        Roots of the characteristic polynomial of the autoregressive part
        :return: the ar_roots of the characteristic polynomial
        """
        polynomial = [-i for i in self.phi]
        polynomial.append(1)
        return np.roots(polynomial)

    def ar_lamba(self):
        """
        Returns the reciprocal of the ar_roots of the characteristic polynomial
        :return: lambda_i=1/z_i
        """
        roots = self.ar_roots()
        return [1 / root for root in roots]

    def isstable(self):
        """
        Checks whether the ARMA process is stable, provided that e_t is bounded.
        :return: boolean, True iff stable
        """
        roots = self.ar_roots()
        return all(abs(root) > 1 for root in roots)

    """
    Moving average properties
    """

    def ma_roots(self):
        """
        Roots of the characteristic polynomial of the moving average part
        :return: the ma_roots of the characteristic polynomial
        """
        polynomial = [i for i in self.theta]
        polynomial.append(1)
        return np.roots(polynomial)

    def ma_lamba(self):
        """
        Returns the reciprocal of the ma_roots of the characteristic polynomial of the moving average part
        :return: lambda_i=1/z_i
        """
        roots = self.ma_roots()
        return [1 / root for root in roots]

    def isinvertible(self):
        """
        Checks whether the ARMA process is invertible by means of MA(q), provided that e_t is bounded.
        :return: boolean, True iff invertible
        """
        roots = self.ma_roots()
        if len(roots) != 0:
            return all(abs(root) > 1 for root in roots)
        else:
            return 'There are no MA terms.'

    """
    ARMA properties
    """

    def irf(self, j):
        """
        Impulse response function \partial y_{t+j} / (\partial e_{t})
        :param j: \partial y_{t+j} / (\partial e_{t})
        :return:
        """
        # initialise y and e
        starttime = max(self.p, self.q)
        e = [0 for i in range(0, starttime)]
        e.append(1)  # impulse
        dy = [0 for i in range(0, starttime)]
        dy.append(1)
        # propagate impulse through time
        time = starttime + 1
        while time <= starttime + j:
            e.append(0)
            ma_t = np.array(self.theta).dot(e[time-self.q:time])+e[time]  # moving average component
            ar_t = np.array(self.phi).dot(dy[time-self.p:time])  # autoregressive component
            dy.append(ar_t + ma_t)
            time += 1
        return dy[starttime+j]


# let's see how it works
arma1 = ARMA(phi_coeffs=[-0.58, 0.6])#, theta_coeffs=[0.1, 0.2, 0.1, 0.3])
print('We have just created the ARMA model: ', arma1.display())
print('\n---------- AR properties ---------------')
print('The root(s) of the AR characteristic polynomial: ', arma1.ar_roots())
print('The AR lambda(s) for factorisation (reciprocal of root(s)): ', arma1.ar_lamba())
print('Claim: the ARMA is stable. Answer: ', arma1.isstable())
print('\n----------- MA properties --------------')
print('The root(s) of the MA characteristic polynomial: ', arma1.ma_roots())
print('Claim: the ARMA is invertible in MA(q) sense. Answer: ', arma1.isinvertible())
print('\n-----------Impulse response -------------')
print('The impulse response function for lag %d is %.4f' % (1, arma1.irf(1)))
