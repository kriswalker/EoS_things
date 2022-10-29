import numpy as np
import matplotlib.pyplot as plt
import bilby
from scipy.optimize import newton
from misc_models import dimensionless_energy_per_baryon_of_density, \
    dimensionless_energy_per_baryon_of_density_model
from utils import calc_confidence_interval, kde1D
from symmetry_energy_likelihood import SymmetryEnergyLikelihood

c = 299792458  # m / s
hbar = 1.05457182e-34  # m^2 kg / s
M_proton = 1.67262192e-27  # kg
M_neutron = 1.67492749e-27  # kg
JtoMeV = 6.242e12

save_data, from_file = False, True
run_inference = True
outdir = 'outdir_symmetry_energy_100likes_sigma4.0_limitconstraint'

logrho_min, logrho_max, res = 17, 18, 100
rhox = 10**np.linspace(logrho_min, logrho_max, res)  # kg / m^3
rho_0 = 2.85e17  # kg / m^3
x = (rhox - rho_0) / (3 * rho_0)

home_dir = '/home/kris/Documents/research/3GEoS_project/'
r = 'outdir_n800_eos_12loudest'
results_file = home_dir + 'results/{}/dynesty_result.json'.format(r)

if from_file:
    infile = open('symmetry_energy_CI_100likes.txt', 'r')
    ci = []
    for line in infile:
        ci.append([float(xi) for xi in line.split()])
    ci = np.array(ci)
    infile.close()
    infile = open('symmetry_energy_curve_samples_100likes.txt', 'r')
    curve_samples = []
    for line in infile:
        curve_samples.append([float(xi) for xi in line.split()])
    curve_samples = np.array(curve_samples)
    infile.close()
else:
    ndraws = 10000
    ci, ys = calc_confidence_interval(
        results_file, rhox, dimensionless_energy_per_baryon_of_density,
        ndraws=ndraws)
    curve_samples = ys[0]

if save_data:
    curve_samples = ys[0]
    outfile = open('symmetry_energy_CI.txt', 'w')
    for cii in ci:
        outfile.write('{} {}\n'.format(cii[0], cii[1]))
    outfile.close()
    outfile = open('symmetry_energy_curve_samples.txt', 'w')
    for yi in curve_samples:
        outfile.write(('{} ' * len(yi))[:-1].format(*yi) + '\n')
    outfile.close()

# fig = plt.subplots(figsize=(6, 5))
# fig[1].fill_between(rhox, ci[:, 0], ci[:, 1], color='C2',
#                     alpha=0.5, label=r'1$\sigma$ confidence interval')
# model = dimensionless_symmetry_energy_of_density_model(rhox, 'SLy')
# fig[1].plot(rhox, model, color='k', label='SLy')
# fig[1].axvline(rho_0, color='r', linewidth=1, label=r'$\rho_0$')
# fig[1].set_xlim(min(rhox), max(rhox))
# fig[1].set_xlabel(r'$\rho$ (kg m$^{-3}$)')
# fig[1].set_ylabel(r'$\varepsilon/(\rho c^2) - 1$')
# fig[1].legend(loc='lower right')
# fig[0].tight_layout()
# fig[0].savefig('symmetry_energy_vs_density.png', dpi=300)
# plt.close(fig[0])

profiles = []
xaxes = []
for i, dist in enumerate(curve_samples.T):
    xi, yi = kde1D(dist.reshape(-1, 1), 'gaussian', 0.01 * np.ptp(dist), 60)

    # fhist = plt.subplots(figsize=(5, 5))
    # fhist[1].hist(dist, bins=40, alpha=0.8, density=True)
    # fhist[1].plot(xi, yi, color='C1')
    # fhist[1].set_xlabel(r'$\varepsilon/(\rho c^2) - 1$')
    # fhist[0].tight_layout()
    # fhist[0].savefig('likelihood_plots/{}_newrange.png'.format(i), dpi=300)
    # plt.close(fhist[0])

    profiles.append(yi / max(yi))
    xaxes.append(xi)
profiles = np.array(profiles)
xaxes = np.array(xaxes)


def neutron_fraction(params):
    F = 64 * symmetry_energy_model(params)**3 / \
        (3 * np.pi**2 * (hbar * c)**3 * rhox)

    def func(f_n):
        _ = (1 - f_n) / (2 * f_n - 1)**2 - F * average_baryon_mass(f_n)
        return _

    sol = newton(func, 0.95 * np.ones(len(rhox)))

    if (np.any(sol) < 0) or (np.any(sol) > 1):
        return 1e100
    else:
        return sol


def average_baryon_mass(f_n):
    return M_proton / (1 + f_n * (M_proton / M_neutron - 1))


def isospin_asymmetry(params):
    f_n = neutron_fraction(params)
    return 2 * f_n - 1


def symmetric_energy_model(params):
    Esnm = params['E_0'] + params['K_0'] * x**2 / 2 + params['J_0'] * x**3 / 6
    return Esnm / JtoMeV


def symmetry_energy_model(params):
    Esym = params['S_0'] + params['L_0'] * x + params['K_sym'] * x**2 / 2 \
        + params['J_sym'] * x**3 / 6
    return Esym / JtoMeV


def dimensionless_energy_per_baryon_model(params):
    delta = isospin_asymmetry(params)
    epb = symmetric_energy_model(params) + \
        symmetry_energy_model(params) * delta**2
    # epb += -params['K_0']/18 + params['J_0']/162 +\
    #     (-params['S_0'] + params['L_0']/2 -
    #      params['K_sym']/18 + params['J_sym']/162) * delta
    f_n = (delta + 1) / 2
    return epb / (average_baryon_mass(f_n) * c**2)


if run_inference:

    priors = dict()

    priors["E_0"] = bilby.prior.Uniform(-25, 0, name='E_0',
                                        latex_label="$E_0$")
    priors["K_0"] = bilby.prior.Uniform(150, 300, name='K_0',
                                        latex_label="$K_0$")
    priors["J_0"] = bilby.prior.Uniform(-200, 200, name='J_0',
                                        latex_label="$J_0$")

    priors["S_0"] = bilby.prior.Uniform(20, 40, name='S_0',
                                        latex_label="$S_0$")
    priors["L_0"] = bilby.prior.Uniform(30, 80, name='L_0',
                                        latex_label="$L_0$")
    priors["K_sym"] = bilby.prior.Uniform(-200, 0, name='K_sym',
                                          latex_label=r"$K_{\rm sym}$")
    priors["J_sym"] = bilby.prior.Uniform(-200, 200, name='J_sym',
                                          latex_label=r"$J_{\rm sym}$")

    likelihood = SymmetryEnergyLikelihood(
        dimensionless_energy_per_baryon_model, priors, profiles, xaxes,
        4.0*np.ones(len(profiles)))

    results = bilby.core.sampler.run_sampler(likelihood, priors=priors,
                                             sampler='dynesty',
                                             label='dynesty',
                                             npoints=200, verbose=False,
                                             resume=False, check_point=True,
                                             check_point_delta_t=60,
                                             outdir=home_dir +
                                             'code/symmetry_energy/'+outdir)
    results.plot_corner()
