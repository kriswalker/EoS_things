import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as lalsim
import bilby
from scipy.optimize import newton
from misc_models import dimensionless_energy_per_baryon_of_density, \
    dimensionless_energy_per_baryon_of_density_model
from utils import calc_confidence_interval

plt.rcParams.update({
    "backend": 'ps',
    "text.usetex": True,
})

model_names = lalsim.SimNeutronStarEOSNames  # list of available lal models

c = 299792458  # m / s
hbar = 1.05457182e-34  # m^2 kg / s
M_proton = 1.67262192e-27  # kg
M_neutron = 1.67492749e-27  # kg
JtoMeV = 6.242e12

###############################################################################

from_file = True

###############################################################################

logrho_min, logrho_max, res = 17, 18, 100
rhox = 10**np.linspace(logrho_min, logrho_max, res)
rho_0 = 2.85e17
x = (rhox - rho_0) / (3 * rho_0)

home_dir = '/home/kris/Documents/research/3GEoS_project/'
r = 'outdir_n800_eos_12loudest'
results_file = home_dir + 'results/{}/dynesty_result.json'.format(r)

if from_file:
    infile = open('data/symmetry_energy_CI_100likes.txt', 'r')
    ci = []
    for line in infile:
        ci.append([float(xi) for xi in line.split()])
    ci = np.array(ci)
    infile.close()
    infile = open('data/symmetry_energy_curve_samples_100likes.txt', 'r')
    curve_samples = []
    for line in infile:
        curve_samples.append([float(xi) for xi in line.split()])
    curve_samples = np.array(curve_samples)
    infile.close()
else:
    ndraws = 100
    ci, ys = calc_confidence_interval(
        results_file, rhox, dimensionless_energy_per_baryon_of_density,
        ndraws=ndraws)
    curve_samples = ys[0]
    outfile = open('symmetry_energy_CI.txt', 'w')
    for cii in ci:
        outfile.write('{} {}\n'.format(cii[0], cii[1]))
    outfile.close()
    outfile = open('symmetry_energy_curve_samples.txt', 'w')
    for yi in curve_samples:
        outfile.write(('{} ' * len(yi))[:-1].format(*yi) + '\n')
    outfile.close()

fig = plt.subplots(figsize=(4, 3))
fig[1].fill_between(rhox, ci[:, 0], ci[:, 1], color='C2',
                    alpha=0.5, label=r'$m$-$\Lambda$ ${\rm inference}$')


def neutron_fraction(xarr, S_0, L_0, K_sym, J_sym):
    F = 64 * symmetry_energy_model(xarr, S_0, L_0, K_sym, J_sym)**3 / \
        (3 * np.pi**2 * (hbar * c)**3 * rhox)

    def func(f_n):
        _ = (1 - f_n) / (2 * f_n - 1)**2 - F * average_baryon_mass(f_n)
        return _

    return newton(func, 0.95 * np.ones(len(rhox)))


def average_baryon_mass(f_n):
    return M_proton / (1 + f_n * (M_proton / M_neutron - 1))


def isospin_asymmetry(xarr, S_0, L_0, K_sym, J_sym):
    f_n = neutron_fraction(xarr, S_0, L_0, K_sym, J_sym)
    return 2 * f_n - 1


def symmetric_energy_model(xarr, E_0, K_0, J_0):
    Esnm = E_0 + K_0 * xarr**2 / 2 + J_0 * xarr**3 / 6
    return Esnm / JtoMeV


def symmetry_energy_model(xarr, S_0, L_0, K_sym, J_sym):
    Esym = S_0 + L_0 * xarr + K_sym * xarr**2 / 2 + J_sym * xarr**3 / 6
    return Esym / JtoMeV


def dimensionless_energy_per_baryon_model(xarr, E_0, K_0, J_0,
                                          S_0, L_0, K_sym, J_sym):
    delta = isospin_asymmetry(xarr, S_0, L_0, K_sym, J_sym)
    epb = symmetric_energy_model(xarr, E_0, K_0, J_0) + \
        symmetry_energy_model(xarr, S_0, L_0, K_sym, J_sym) * delta**2
    f_n = (delta + 1) / 2
    return epb / (average_baryon_mass(f_n) * c**2)


results_file = \
    'data/outdir_symmetry_energy_100likes_sigma4.0/dynesty_result.json'
results = bilby.result.read_in_result(filename=results_file)
param_vals = {}
param_errs = {}
for n, key in enumerate(list(results.posterior)[:8]):
    val = results.get_one_dimensional_median_and_error_bar(key)
    param_vals[key] = val.median
    param_errs[key] = (val.plus + val.minus) / 2
    print(key + ' = {} +/- {} MeV'.format(param_vals[key], param_errs[key]))

ci, ys = calc_confidence_interval(results_file, x,
                                  dimensionless_energy_per_baryon_model,
                                  ndraws=1000)

# for y in ys[0]:
#     fig[1].plot(rhox, y, color='grey', alpha=0.3, linewidth=1, zorder=0)
fig[1].fill_between(rhox, ci[:, 0], ci[:, 1], color='grey',
                    alpha=0.5, label=r'$E_{\rm sym}$ ${\rm inference}$')

rhox_smooth = 10**np.linspace(logrho_min, logrho_max, 100)
model = dimensionless_energy_per_baryon_of_density_model(rhox_smooth, 'SLy')
fig[1].plot(rhox_smooth, model, color='k', label=r'${\rm SLy}$')
fig[1].axvline(rho_0, color='k', linewidth=0.5,
               label=r'${\rm Saturation\,density},\,\rho_0$')
fig[1].set_xlim(min(rhox), max(rhox))
# fig[1].set_ylim(-0.01, 0.14)
# fig[1].set_xscale('log')
fig[1].set_xlabel(r'$\rho\,({\rm kg\,\rm m}^{-3})$')
fig[1].set_ylabel(r'$E(\rho)/(Mc^2)=\varepsilon(\rho)/(\rho c^2) - 1$')
fig[1].legend(loc='upper left')
fig[0].tight_layout()
fig[0].savefig('figures/symmetry_energy_vs_density.pdf')
