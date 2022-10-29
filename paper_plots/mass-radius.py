import sys
import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as lalsim
from scipy.interpolate import interp1d
from utils import calc_confidence_interval, calc_model

from toast.piecewise_polytrope import radius_of_mass as radius_of_mass_poly
from spectral_decomposition import radius_of_mass as radius_of_mass_spect
from toast.eos_by_name import radius_array_of_mass as radius_of_mass_eos

plt.rcParams.update({
    "backend": 'ps',
    "text.usetex": True,
})

model_names = lalsim.SimNeutronStarEOSNames  # list of available lal models

###############################################################################

radius_of_mass = radius_of_mass_spect
from_file = False
norm, model_name = False, 'SLy'
plot_draws = False
save = True

###############################################################################

m_min, m_max, res = 0.5, 1.97, 100
mx = np.linspace(m_min, m_max, res)

models = ['SLy']
ys_models = calc_model(mx, radius_of_mass_eos, models)

home_dir = '/home/kris/Documents/research/3GEoS_project/'
rs = ['outdir_n800_eos_12loudest', 'outdir_n100_spectral_12loudest']
funcs = [radius_of_mass_poly, radius_of_mass_spect]
colors = ['C2', 'C0']
labels = [r'${\rm Piecewise\,polytrope}$', r'${\rm Spectral\,decomposition}$']
fig = plt.subplots(figsize=(4, 3))
for i, r in enumerate(rs):

    def RofM(mass, **params):
        return funcs[i](mass, **params)[1]

    results_file = home_dir + 'results/{}/dynesty_result.json'.format(r)

    ndraws = 1000
    means = []
    stds = []
    if from_file:
        infile = open('mass-radius_CI.txt', 'r')
        ci = []
        for line in infile:
            ci.append([float(x) for x in line.split()])
        ci = np.array(ci)
        infile.close()
    else:
        ci, ys = calc_confidence_interval(results_file, mx, RofM,
                                          ndraws=ndraws)
        # outfile = open('mass-radius_CI_spectral.txt', 'w')
        # for cii in ci:
        #     outfile.write('{} {}\n'.format(cii[0], cii[1]))
        # outfile.close()

    model = ys_models[model_name] if norm else np.ones(res)

    fig[1].fill_between(mx, ci[:, 0]/model, ci[:, 1]/model,
                        color=colors[i], alpha=0.4,
                        label=labels[i])
                        # label=r'1$\sigma\,{\rm confidence}\,{\rm interval}$')
    if plot_draws:
        for j, y in enumerate(ys[0]):
            fig[1].plot(mx, y, color='grey', alpha=0.3, zorder=0,
                        linewidth=0.5, label='draws' if j == 0 else None)

for name in ys_models.keys():
    fig[1].plot(mx, ys_models[name]/model, color='k',
                label=r'${{\rm {}}}$'.format(name))
fig[1].set_xlim(m_min, m_max)
# fig[1].set_ylim(0.8, 1.2)
fig[1].set_xlabel(r'$m\,({\rm M}_\odot)$')
ylabel = r'$R/R_\mathrm{SLy}$' if norm else r'$R\,({\rm km})$'
fig[1].set_ylabel(ylabel)
fig[1].legend()
fig[0].tight_layout()
if save:
    fig[0].savefig('figures/radius_vs_mass_overlay.pdf')
