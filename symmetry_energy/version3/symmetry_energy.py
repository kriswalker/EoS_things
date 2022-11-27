import numpy as np
from scipy.optimize import newton

c = 299792458  # m / s
hbar = 1.05457182e-34  # m^2 kg / s
M_proton = 1.67262192e-27  # kg
M_neutron = 1.67492749e-27  # kg
JtoMeV = 6.242e12
rho_0 = 2.85e17  # kg / m^3


def neutron_fraction(rhox, params):
    F = 64 * symmetry_energy_model(rhox, params)**3 / \
        (3 * np.pi**2 * (hbar * c)**3 * rhox)

    def func(f_n):
        _ = (1 - f_n) / (2 * f_n - 1)**2 - F * average_baryon_mass(f_n)
        return _

    sol = newton(func, 0.95 * np.ones(len(rhox)))

    if (np.any(sol < 0)) or (np.any(sol > 1)):
        return np.inf
    else:
        return sol


def average_baryon_mass(f_n):
    # return M_proton / (1 + f_n * (M_proton / M_neutron - 1))
    return M_proton + f_n * (M_neutron - M_proton)


def isospin_asymmetry(rhox, params):
    f_n = neutron_fraction(rhox, params)
    return 2 * f_n - 1


def symmetry_energy_model(rhox, params):
    x = (rhox - rho_0) / (3 * rho_0)
    Esym = params['S_0'] + params['L_0'] * x + params['K_sym'] * x**2 / 2 \
        + params['J_sym'] * x**3 / 6
    return Esym / JtoMeV


def pressure_expansion(rhox, params):

    x = (rhox - rho_0) / (3 * rho_0)
    snm = params['K_0'] * x + params['J_0'] * x**2 / 2
    sym = params['L_0'] + params['K_sym'] * x + params['J_sym'] * x**2 / 2

    delta = isospin_asymmetry(rhox, params)
    p = snm + sym * delta**2

    f_n = (delta + 1) / 2
    fac = rhox**2 / (3 * average_baryon_mass(f_n) * rho_0)

    return fac * p / JtoMeV
