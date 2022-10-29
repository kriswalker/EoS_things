import numpy as np
from bilby.core.likelihood import Likelihood
from scipy.ndimage import gaussian_filter1d


def calculate_symmetry_energy_pdf(profile, noise_level, smooth=True,
                                  log=False):
    """
    Calculates the probability distribution
    """
    if smooth:
        profile = gaussian_filter1d(profile, noise_level)
    if np.shape(noise_level) == ():
        max_power = np.max(profile)
    else:
        max_power = np.max(profile, axis=1).reshape((len(profile), 1))
        noise_level = noise_level.reshape((len(noise_level), 1))
    if log:
        return np.log(1/(noise_level * np.sqrt(2*np.pi))) + \
            -0.5 * ((profile - max_power) / noise_level)**2
    else:
        return 1/(noise_level * np.sqrt(2*np.pi)) * \
            np.exp(-0.5 * ((profile - max_power) / noise_level)**2)


class SuperLikelihood(Likelihood):
    """
    Bilby-compatible class for subclassing likelihood functions.

    """

    def __init__(self, y, func, params, **kwargs):

        super(SuperLikelihood, self).__init__(params)
        self.y = y
        self._func = func
        self.kwargs = kwargs

    @property
    def func(self):
        return self._func

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        if isinstance(y, int) or isinstance(y, float):
            y = np.array([y])
        self._y = y

    @property
    def model(self):
        return self.func(self.parameters, **self.kwargs)


class SymmetryEnergyLikelihood(SuperLikelihood):
    def __init__(self, symmetry_energy_model, params, profiles, x, sigma,
                 **kwargs):

        super(SymmetryEnergyLikelihood, self).__init__(
            y=np.zeros(len(profiles)), func=symmetry_energy_model,
            params=params, **kwargs)

        self.profiles = profiles
        self.x = x
        self.sigma = sigma

    def log_likelihood(self):

        if 'lnefac' in self.parameters.keys():
            lnefac = self.parameters['lnefac']
            equad = self.parameters['equad']
            noise = np.sqrt((self.sigma * np.exp(lnefac))**2 + equad**2)
        else:
            noise = self.sigma

        se_prob = calculate_symmetry_energy_pdf(self.profiles, noise,
                                                smooth=False, log=True)
        integral = np.sum(np.exp(se_prob[..., :-1]) *
                          np.diff(self.x, axis=1), axis=1)
        integral = integral.reshape((len(integral), 1))
        se_prob_norm = se_prob - np.log(integral)

        model_x = self.model

        like = np.zeros(len(self.x))
        outside = np.argwhere(
            (model_x > np.max(self.x, axis=1)) |
            (model_x < np.min(self.x, axis=1))).flatten()
        inside = np.argwhere(
            (model_x < np.max(self.x, axis=1)) &
            (model_x > np.min(self.x, axis=1))).flatten()
        like[outside] = -200  # for model x outside profile x range

        model_x = model_x[inside].reshape(
            (len(model_x[inside]), 1))
        inds = np.argmin(np.abs(self.x[inside] - model_x), axis=1)
        like[inside] = se_prob_norm[inside, inds]

        return np.sum(like)
