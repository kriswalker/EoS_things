import numpy as np
import bilby
from scipy.special import logsumexp


class EOSLikelihood(bilby.Likelihood):
    """
    An hyperlikelihood that samples equation of state hyper-parameters
    given a chosen parametrization.

    """

    def __init__(self, likelihood_list, axes_list, pressure_array,
                 maximum_mass, maximum_speed_of_sound, parametrization,
                 parameters):
        bilby.Likelihood.__init__(self, parameters=parameters)

        self.axes_list = axes_list
        self.likelihood_list = likelihood_list
        self.pressure_array = pressure_array
        self.maximum_mass = maximum_mass
        self.maximum_speed_of_sound = maximum_speed_of_sound
        self.parametrization = parametrization

    def eos_model(self, axes):
        """

        """

        dict_tmp = dict(chirp_mass=axes[0],
                        mass_ratio=axes[1])
        tmp = \
            bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters(
                dict_tmp)

        m1 = tmp[0]['mass_1']
        m2 = tmp[0]['mass_2']
        mass_array = np.array([m1, m2])
        Lambda, max_mass = self.parametrization.Lambda_of_mass(
            self.pressure_array, mass_array, **self.parameters,
            maximum_mass_limit=self.maximum_mass)

        if type(Lambda) == float:
            return None, None
        else:
            return Lambda[0], Lambda[1]

    def get_marg_likelihood(self, likelihood, axes, lambda1, lambda2):

        cm_ax, q_ax, l1_ax, l2_ax = axes
        m_likelihood, l_likelihood = likelihood

        marg_l = []
        for i in range(len(cm_ax)):
            if (min(l1_ax) <= lambda1[i] <= max(l1_ax)):
                l1_ind = np.argmin(np.abs(lambda1[i] - l1_ax))
            else:
                marg_l.append(-200)
                continue
            if (min(l2_ax) <= lambda2[i] <= max(l2_ax)):
                l2_ind = np.argmin(np.abs(lambda2[i] - l2_ax))
            else:
                marg_l.append(-200)
                continue
            marg_l.append(m_likelihood[i, i] + l_likelihood[l1_ind, l2_ind])

        return np.array(marg_l)

    def mass_prior(self, axes):

        return np.ones(np.shape(axes[0]))

    def integrate_likelihood(self, likelihood, axes, measure):
        """

        """

        l1, l2 = self.eos_model()

        if l1 is not None:
            marg_likelihood = self.get_marg_likelihood(likelihood, axes,
                                                       l1, l2)
            marg_likelihood += np.log(self.mass_prior(axes))
            return logsumexp(marg_likelihood) + np.log(measure)
        else:
            return -np.inf

    def log_likelihood(self):

        max_speed_of_sound = self.parametrization.maximum_speed_of_sound(
            **self.parameters)

        if max_speed_of_sound <= self.maximum_speed_of_sound:
            integral = list()
            for ii, likelihood in enumerate(self.likelihood_list):
                axes = self.axes_list[ii]
                dA = np.diff(axes[0])[0] * np.diff(axes[1])[0]
                integral.append(self.integrate_likelihood(
                    likelihood, axes, dA))
            return np.double(sum(np.array(integral)))
        else:
            return -np.inf
