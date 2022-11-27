import sys
import numpy as np
import bilby
import matplotlib.pyplot as plt
from toast.eos_by_name import density_of_pressure
from symmetry_energy_likelihood import SymmetryEnergyLikelihood
from symmetry_energy import pressure_expansion

# homedir = '/fred/oz170/kwalker/projects/3G_EoS/'
homedir = '/home/kris/Documents/research/3GEoS_project/'
datafile = homedir + 'data/kdes/polytrope_kde.npz'
npoints = 200
extension = '_S0fixed'

data = np.load(datafile)
kde = data['kde']
axes = data['axes']

priors = dict()
# priors["E_0"] = bilby.prior.Uniform(-25, 0, name='E_0',
#                                     latex_label="$E_0$")
priors["K_0"] = bilby.prior.Uniform(150, 300, name='K_0',
                                    latex_label="$K_0$")
priors["J_0"] = bilby.prior.Uniform(-200, 200, name='J_0',
                                    latex_label="$J_0$")

# priors["S_0"] = bilby.prior.Uniform(20, 40, name='S_0',
#                                     latex_label="$S_0$")
priors["S_0"] = bilby.prior.DeltaFunction(32, name='S_0',
                                          latex_label="$S_0$")
priors["L_0"] = bilby.prior.Uniform(30, 80, name='L_0',
                                    latex_label="$L_0$")
priors["K_sym"] = bilby.prior.Uniform(-200, 0, name='K_sym',
                                      latex_label=r"$K_{\rm sym}$")
priors["J_sym"] = bilby.prior.Uniform(-200, 200, name='J_sym',
                                      latex_label=r"$J_{\rm sym}$")

rho_0 = 2.85e17
logrho_min, logrho_max, res = np.log10(rho_0), np.log10(3 * rho_0), 100
rhox = 10**np.linspace(logrho_min, logrho_max, res)

likelihood = SymmetryEnergyLikelihood(kde, rhox, axes, priors)

results = bilby.core.sampler.run_sampler(
    likelihood, priors=priors, sampler='dynesty', label='dynesty',
    npoints=npoints, verbose=False, resume=False, check_point_delta_t=60,
    outdir='outdir_symmetryenergy_n{0}{1}'.format(npoints, extension))
results.plot_corner()
post = results.posterior

fig, ax = plt.subplots(figsize=(6, 6))
ndraws = 100
for _ in range(ndraws):
    draw = post.sample().to_dict('records')[0]
    p = pressure_expansion(rhox, draw)
    label = r'${\rm EoS\,\,draws}$' if _ == 0 else None
    ax.plot(rhox, p, color='C0', alpha=0.5, label=label)
maxl_params = dict(post[priors.keys()].iloc[post.log_likelihood.idxmax()])
p = pressure_expansion(rhox, maxl_params)
ax.plot(rhox, p, color='k', alpha=0.8,
        label=r'${\rm max\,\,likelihood\,\,EoS}$')
ax.plot(density_of_pressure(p, 'sly'), p, color='r', alpha=0.8,
        label=r'SLy')
ax.set_xlim(min(rhox), max(rhox))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel(r'$p\,({\rm Pa})$')
ax.set_xlabel(r'$\rho\,({\rm kg}\,{\rm m}^{-3})$')
ax.legend()
fig.tight_layout()
