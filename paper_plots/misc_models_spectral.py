import numpy as np
from scipy.interpolate import interp1d
from spectral_decomposition import density_of_pressure, \
    energy_density_of_pressure


def pressure_of_density(density, gamma_0, gamma_1, gamma_2, gamma_3):

    px = 10**np.linspace(30, 36, 100)
    f = interp1d(density_of_pressure(
        px, gamma_0, gamma_1, gamma_2, gamma_3), px)
    return f(density)


def dimensionless_energy_per_baryon_of_density(density, gamma_0, gamma_1,
                                               gamma_2, gamma_3):
    c = 299792458
    pressure = pressure_of_density(density, gamma_0, gamma_1, gamma_2, gamma_3)
    eps = energy_density_of_pressure(
        pressure, gamma_0, gamma_1, gamma_2, gamma_3)
    return eps / (density * c**2) - 1
