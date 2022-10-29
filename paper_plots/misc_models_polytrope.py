import numpy as np
from scipy.interpolate import interp1d
from toast.piecewise_polytrope import density_of_pressure, \
    energy_density_of_pressure
from toast.eos_by_name import density_of_pressure as density_of_pressure_eos, \
    energy_density_of_pressure as energy_density_of_pressure_eos


def pressure_of_density(density, log_p, Gamma_1, Gamma_2, Gamma_3):

    # this *should* work, but fails below rho ~ 2.5e17 kg/m^3
    # rho1, rho2 = 10**17.7, 10**18
    # pressure = []
    # for d in density:
    #     if d < rho1:
    #         K = 10**log_p / rho1**Gamma_1
    #         pressure.append(K * d**Gamma_1)
    #     elif d < rho2:
    #         K = 10**log_p / rho1**Gamma_2
    #         pressure.append(K * d**Gamma_2)
    #     else:
    #         K2 = 10**log_p / rho1**Gamma_2
    #         K = K2 * rho2**(Gamma_2 - Gamma_3)
    #         pressure.append(K * d**Gamma_3)
    # return np.array(pressure)

    px = 10**np.linspace(30, 36, 100)
    f = interp1d(density_of_pressure(px, log_p, Gamma_1, Gamma_2, Gamma_3), px)
    return f(density)


def pressure_of_density_model(density, model):
    px = 10**np.linspace(31.3, 35, 1000)
    f = interp1d(density_of_pressure_eos(px, model), px)
    return f(density)


def dimensionless_energy_per_baryon_of_density(density, log_p, Gamma_1,
                                               Gamma_2, Gamma_3):
    c = 299792458
    pressure = pressure_of_density(density, log_p, Gamma_1, Gamma_2, Gamma_3)
    eps = energy_density_of_pressure(
        pressure, log_p, Gamma_1, Gamma_2, Gamma_3)
    return eps / (density * c**2) - 1


def dimensionless_energy_per_baryon_of_density_model(density, model):
    c = 299792458
    pressure = pressure_of_density_model(density, model)
    eps = energy_density_of_pressure_eos(pressure, model)

    return eps / (density * c**2) - 1
