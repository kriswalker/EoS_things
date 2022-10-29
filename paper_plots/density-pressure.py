import sys
import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as lalsim
from scipy.interpolate import interp1d
from utils import calc_confidence_interval, calc_model

from misc_models_polytrope import pressure_of_density as \
    pressure_of_density_poly, pressure_of_density_model as \
        pressure_of_density_eos
from misc_models_spectral import pressure_of_density as \
    pressure_of_density_spect

plt.rcParams.update({
    "backend": 'ps',
    "text.usetex": True,
})

model_names = lalsim.SimNeutronStarEOSNames  # list of available lal models

###############################################################################

norm, model_name = False, 'SLy'
plot_draws = False

###############################################################################

logrho_min, logrho_max, res = 16.9, 18.2, 100
rhox = 10**np.linspace(logrho_min, logrho_max, res)

rhos = 2.85e17, 10**17.7, 10**18

models = ['SLy']
ys_models = calc_model(rhox, pressure_of_density_eos, models)

home_dir = '/home/kris/Documents/research/3GEoS_project/'
rs = ['outdir_n800_eos_12loudest', 'outdir_n100_spectral_12loudest']
funcs = [pressure_of_density_poly, pressure_of_density_spect]
colors = ['C2', 'C0']
labels = [r'${\rm Piecewise\,polytrope}$', r'${\rm Spectral\,decomposition}$']
fig = plt.subplots(figsize=(4, 3))
for i, r in enumerate(rs):
    results_file = home_dir + 'results/{}/dynesty_result.json'.format(r)

    ndraws = 1000
    means = []
    stds = []
    ci, ys = calc_confidence_interval(results_file, rhox,
                                      funcs[i], ndraws=ndraws)
    # outfile = open('density-pressure_CI.txt', 'w')
    # for cii in ci:
    #     outfile.write('{} {}\n'.format(cii[0], cii[1]))
    # outfile.close()

    fig[1].fill_between(rhox, ci[:, 0], ci[:, 1], color=colors[i], alpha=0.5,
                        label=labels[i])
    for rho in rhos:
        fig[1].axvline(rho, color='k', linewidth=0.5)
    if plot_draws:
        for j, y in enumerate(ys[0]):
            fig[1].plot(rhox, ys_models['SLy'], color='grey', alpha=0.3,
                        zorder=0, label='draws' if j == 0 else None)

for name in ys_models.keys():
    fig[1].plot(rhox, ys_models[name], color='k',
                label=r'${{\rm {}}}$'.format(name))
# fig[1].set_xlim(10**logp_min, 10**logp_max)
# fig[1].set_xlim(min(ys_models['SLy']), max(ys_models['SLy']))
fig[1].set_xlim(min(rhox), max(rhox))
# fig[1].set_ylim(0, 4)
fig[1].set_xscale('log')
fig[1].set_yscale('log')
ylabel = r'$p/p_\mathrm{SLy}$' if norm else r'$p\,({\rm Pa})$'
fig[1].set_ylabel(ylabel)
fig[1].set_xlabel(r'$\rho\,({\rm kg}\,{\rm m}^{-3})$')
fig[1].legend(loc='upper left')
fig[0].tight_layout()
fig[0].savefig(home_dir +
               'code/paper_plots/figures/pressure_vs_density_overlay.pdf')
