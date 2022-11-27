import numpy as np
from scipy.interpolate import interp1d
from toast.piecewise_polytrope import density_of_pressure
from toast.eos_by_name import density_of_pressure as density_of_pressure_eos


def pressure_of_density(density, log_p, Gamma_1, Gamma_2, Gamma_3):

    rho1, rho2 = 10**17.7, 10**18
    pressure = []
    K1 = 10**log_p / rho1**Gamma_1
    K2 = K1 * rho1**(Gamma_1 - Gamma_2)
    K3 = K2 * rho2**(Gamma_2 - Gamma_3)
    for d in density:
        if d < rho1:
            pressure.append(K1 * d**Gamma_1)
        elif d < rho2:
            pressure.append(K2 * d**Gamma_2)
        else:
            pressure.append(K3 * d**Gamma_3)
    pressure = np.array(pressure)

    # pressure_sly = pressure_of_density_eos(density, 'SLY4')
    # intersection = np.argmin(np.abs(pressure - pressure_sly))
    # pressure[:intersection] = pressure_sly[:intersection]

    return pressure


def pressure_of_density_2(density, log_p, Gamma_1, Gamma_2, Gamma_3):

    px = 10**np.linspace(31, 35.2, 300)
    f = interp1d(density_of_pressure(px, log_p, Gamma_1, Gamma_2, Gamma_3), px)
    return f(density)


def pressure_of_density_eos(density, eos_name):

    px = 10**np.linspace(31, 35.2, 300)
    f = interp1d(density_of_pressure_eos(px, eos_name), px)
    return f(density)
