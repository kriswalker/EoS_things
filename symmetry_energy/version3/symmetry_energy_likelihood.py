import numpy as np
import bilby
from scipy.optimize import least_squares
import toast.piecewise_polytrope as piecewise_polytrope
import symmetry_energy
from misc_models import pressure_of_density


class SymmetryEnergyLikelihood(bilby.Likelihood):

    def __init__(self, likelihood, density, polytrope_axes, parameters):
        bilby.Likelihood.__init__(self, parameters=parameters)

        self.density = density
        self.logp_ax, self.Gamma1_ax, self.Gamma2_ax = polytrope_axes
        self.likelihood = likelihood

    def get_polytrope_parameters(self):

        pressure = symmetry_energy.pressure_expansion(self.density,
                                                      self.parameters)
        if np.any(np.isnan(pressure)):
            return False

        def residual(p):
            r = pressure - pressure_of_density(self.density, *p, Gamma_3=2.71)
            return r

        return least_squares(residual, [33.72, 3.62, 2.87]).x

    def log_likelihood(self):

        params = self.get_polytrope_parameters()
        if type(params) == bool:
            return -np.inf

        b1 = (params[0] > self.logp_ax[0]) & (params[0] < self.logp_ax[-1])
        b2 = (params[1] > self.Gamma1_ax[0]) & (params[1] < self.Gamma1_ax[-1])
        b3 = (params[2] > self.Gamma2_ax[0]) & (params[2] < self.Gamma2_ax[-1])

        if b1 and b2 and b3:
            idx1 = np.argmin(np.abs(self.logp_ax - params[0]))
            idx2 = np.argmin(np.abs(self.Gamma1_ax - params[1]))
            idx3 = np.argmin(np.abs(self.Gamma2_ax - params[2]))
            return self.likelihood[idx1, idx2, idx3]
        else:
            return -np.inf
