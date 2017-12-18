"""
Intermediate Econometrics Problem set 1 (due: 31 October 2017)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# figure settings


def figsize(scale):
    """
    Figure scaling for LaTeX
    :param scale:
    :return:
    """
    fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


    # figure format settings
pgf_with_rc_fonts = {'font.family': 'serif', 'figure.figsize': figsize(0.9), 'pgf.texsystem': 'pdflatex'}
mpl.rcParams.update(pgf_with_rc_fonts)
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
plt.rc('font', family='serif')
plt.close()  # clean-up previous figures


class ARMA:
    """
    ARMA models of the form
    Y_t = c + phi_1 * y(t-1) + ... + phi_p * y(t-p) + theta_1 * e(t-1) + ... + theta_q * e(t-q) + e_t,
    with e_t white noise(0, sigma^2).
    """
    def __init__(self, phi_coeffs, theta_coeffs=list(), constant=list()):
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
        self.con = constant
        self.c = len(constant)

    def display(self):
        """
        A visual representation of the model

        Returns
        -------
        :return: a string representation of the model
        """
        if self.c != 0:
            c_str = str(self.con[0])
        ar_str = '+'.join([str(self.phi[i]) + '*Y(t-' + str(self.p - i) + ')' for i in range(0, self.p)
                           if self.phi[i] != 0])
        ma_str = '+'.join([str(self.theta[i]) + '*e(t-' + str(self.q - i) + ')' for i in range(0, self.q)
                           if self.theta[i] != 0])
        if self.c != 0:  # there is constant
            if self.q != 0: # there are MA term(s)
                return 'Y(t) = ' + c_str + '+' + ar_str + '+' + ma_str + '+e(t)'
            else:  # no MA terms
                return 'Y(t) = ' + c_str + '+' + ar_str + '+e(t)'
        else:  # no constant
            if self.q != 0:  # there are MA term(s)
                return 'Y(t) = ' + ar_str + '+' + ma_str + '+e(t)'
            else:  # no MA terms
                return 'Y(t) = ' + ar_str + '+e(t)'


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

    def irf(self, j_max):
        """
        Impulse response function \partial y_{t+j} / (\partial e_{t}) for j=0,...,j_max
        :param j_max: \partial y_{t+j} / (\partial e_{t}) for j=0,...,j_max
        :return: array of length j_max+1, with first element corresponding to j=0,...
        """
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
            ma_t = np.array(self.theta).dot(e[time-self.q:time])+e[time]  # moving average component
            ar_t = np.array(self.phi).dot(dy[time-self.p:time])  # autoregressive component
            dy.append(ar_t + ma_t)
            time += 1
        return dy[starttime:starttime + j_max + 1]


# let's see how it works
arma1 = ARMA(phi_coeffs=[-0.58, 0.6])
print('We have just created the ARMA model: ', arma1.display())
print('\n---------- AR properties ---------------')
print('The root(s) of the AR characteristic polynomial: ', arma1.ar_roots())
print('The AR lambda(s) for factorisation (reciprocal of root(s)): ', arma1.ar_lamba())
print('Claim: the ARMA is stable. Answer: ', arma1.isstable())
print('\n----------- MA properties --------------')
print('The root(s) of the MA characteristic polynomial: ', arma1.ma_roots())
print('Claim: the ARMA is invertible in MA(q) sense. Answer: ', arma1.isinvertible())
print('\n-----------Impulse response -------------')
print('The impulse response function up until lag %d is ' % 2, arma1.irf(2))


"""
Problem 2, numerical approximation
"""

print('\n-------------------------- PROBLEM 2 ------------------------------------')


def gen_phi(n):
    """
    Generates a n-sized random i.i.d. sample from the uniform distribution U(-1,1).
    Uses inverse cdf transformation.
    :param n: size of the sample
    :return: a n-length array
    """
    #cdf_x = np.random.rand(n)
    #x = 2 * cdf_x - 1
    x = np.random.chisquare(2, n)
    return x


np.random.seed(0)
for p in [1, 2, 3, 4, 5, 6, 7, 8]:
    b_sum = 0
    effective_m = 0
    for m in np.arange(1, 100000):
        phi = gen_phi(p)
        if phi[-1] != 0:
            arma = ARMA(phi_coeffs=phi)
            #if arma.isstable():
            b = -phi[0] * np.product(arma.ar_roots()) * (- 1) ** p
            b_sum += b
            effective_m += 1
    b_mean = b_sum / effective_m  # mean conditional on phi_p != 0
    print('The average (of %d numbers ) for p=%d is ' % (effective_m, p), b_mean)


"""
Problem 3
"""

print('\n-------------------------- PROBLEM 3 ------------------------------------')
arma = ARMA(phi_coeffs=[0.1, 0.3])
print('We have just created the ARMA model: ', arma.display())
print('\n---------- AR properties ---------------')
print('The root(s) of the AR characteristic polynomial: ', arma.ar_roots())
print('The AR lambda(s) for factorisation (reciprocal of root(s)): ', arma.ar_lamba())
print('Claim: the ARMA is stable. Answer: ', arma.isstable())


"""
Problem 4
"""

models = [{'ar': [-0.58, 0.6], 'ma': [], 'constant': []},  # a
          {'ar': [0.84, 0.25], 'ma': [], 'constant': []},  # b
          {'ar': [-0.1, -0.2, 1.3], 'ma': [], 'constant': []},  # c
          {'ar': [-0.1, -0.2, 1.3], 'ma': [], 'constant': [0.05]},  # d
          {'ar': [0.7], 'ma': [0.6, 0, 0, 0], 'constant': []}]  # e

f, axes = plt.subplots(nrows=3, ncols=2, sharex='all')

for model, axis in zip(models, axes.reshape(-1)):
    arma = ARMA(phi_coeffs=model['ar'], theta_coeffs=model['ma'], constant=model['constant'])
    axis.plot(arma.irf(j_max=30), label='IRF(j)', color='red', linewidth=2)
    axis.legend()
    axis.set_title(arma.display() + '\n $|$AR roots$|$: ' + '('+
                   ', '.join(['%.2f' % abs(i) for i in arma.ar_roots()]) + ')', size=10)
    axis.set_xlabel('$j$', size=10)
    plt.setp(axis.get_xticklabels(), size=10, ha='center')
    plt.setp(axis.get_yticklabels(), size=10)

for l in axes[1, 1].get_xaxis().get_majorticklabels():
    l.set_visible(True)
f.delaxes(axes[2, 1])
plt.tight_layout()


f.set_size_inches(figsize(1.1)[0], figsize(0.9)[1])
# save to pgf
plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Fall\IntEcon\ps1\Problem4.pgf', bbox_inches='tight')
# save to png
plt.savefig(r'C:\Users\Máté\Dropbox\CEU\2017 Fall\IntEcon\ps1\Problem4.png', bbox_inches='tight')
plt.show(block=False)
