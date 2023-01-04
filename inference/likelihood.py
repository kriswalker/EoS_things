import sys
import os
import numpy as np
import bilby
from scipy.special import logsumexp
from contextlib import contextmanager


@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


class EOSLikelihood(bilby.Likelihood):
    """
    An hyperlikelihood that samples equation of state hyper-parameters
    given a chosen parametrization.

    """

    def __init__(self, likelihood_list, axes_list, logZ_list, pressure_array,
                 maximum_mass, maximum_speed_of_sound, parametrization,
                 parameters):
        bilby.Likelihood.__init__(self, parameters=parameters)

        self.axes_list = axes_list
        self.likelihood_list = likelihood_list
        self.logZ_list = logZ_list
        self.pressure_array = pressure_array
        self.maximum_mass = maximum_mass
        self.maximum_speed_of_sound = maximum_speed_of_sound
        self.parametrization = parametrization
        self.mass_array_list = self.create_mass_array()

        dA_list = []
        for axes in axes_list:
            maxes = axes[0]
            dA = np.diff(maxes[0])[0] * np.diff(maxes[1])[0]
            dA_list.append(dA)
        self.dA_list = dA_list

    def create_mass_array(self):
        mass_array_list = []
        for axes in self.axes_list:
            maxes = axes[0]
            mgrid = np.array(np.meshgrid(*maxes)).reshape(2, len(maxes[0])**2)
            dict_tmp = dict(chirp_mass=mgrid[0],
                            mass_ratio=mgrid[1])
            tmp = bilby.gw.conversion.\
                convert_to_lal_binary_neutron_star_parameters(dict_tmp)

            m1 = tmp[0]['mass_1']
            m2 = tmp[0]['mass_2']
            mass_array_list.append(np.array([m1, m2]))

        return mass_array_list

    def eos_model(self, mass_array):
        """

        """

        Lambda, max_mass = self.parametrization.Lambda_of_mass(
            self.pressure_array, mass_array, **self.parameters,
            maximum_mass_limit=self.maximum_mass)

        if type(Lambda) == float:
            return None, None
        else:
            return Lambda[0], Lambda[1]

    def get_marg_likelihood(self, likelihood, axes, lambda1, lambda2):

        maxes, laxes = axes
        cm_ax, q_ax = maxes
        l1_ax, l2_ax = laxes
        m_likelihood, l_likelihood = likelihood

        cond1 = min(l1_ax) < min(lambda1) < max(l1_ax)
        cond2 = min(l1_ax) < max(lambda1) < max(l1_ax)
        cond3 = min(l2_ax) < min(lambda2) < max(l2_ax)
        cond4 = min(l2_ax) < max(lambda2) < max(l2_ax)
        if (cond1 or cond2) and (cond3 or cond4):
            marg_l = []
            k = 0
            for i in range(len(q_ax)):
                for j in range(len(cm_ax)):
                    if (min(l1_ax) <= lambda1[k] <= max(l1_ax)):
                        l1_ind = np.argmin(np.abs(lambda1[k] - l1_ax))
                    else:
                        marg_l.append(-200)
                        continue
                    if (min(l2_ax) <= lambda2[k] <= max(l2_ax)):
                        l2_ind = np.argmin(np.abs(lambda2[k] - l2_ax))
                    else:
                        marg_l.append(-200)
                        continue
                    marg_l.append(m_likelihood[j, i] +
                                  l_likelihood[l1_ind, l2_ind])
                    k += 1
        else:
            marg_l = [-200] * len(q_ax) * len(cm_ax)

        return np.array(marg_l)

    def mass_prior(self, axes):

        return np.ones(np.shape(axes[0][0]))

    def integrate_likelihood(self, likelihood, axes, measure, logZ,
                             mass_array):
        """

        """
        l1, l2 = self.eos_model(mass_array)

        if l1 is not None:
            marg_likelihood = self.get_marg_likelihood(likelihood, axes,
                                                       l1, l2)
            # marg_likelihood += np.log(self.mass_prior(axes))
            return logsumexp(marg_likelihood) + logZ + np.log(measure)
        else:
            return -np.inf

    def log_likelihood(self):

        try:
            with suppress_stderr():
                max_sound_speed = self.parametrization.maximum_speed_of_sound(
                    **self.parameters)
        except RuntimeError:
            return -np.inf

        if max_sound_speed <= self.maximum_speed_of_sound:
            integral = list()
            for i, likelihood in enumerate(self.likelihood_list):
                integral.append(self.integrate_likelihood(
                    likelihood, self.axes_list[i], self.dA_list[i],
                    self.logZ_list[i], self.mass_array_list[i]))
            print(integral)
            logL = np.double(sum(np.array(integral)))
            return logL
        else:
            return -np.inf
