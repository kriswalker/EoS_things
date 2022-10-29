import sys
import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as lalsim
from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters
from scipy.interpolate import interp1d
from utils import calc_confidence_interval, calc_model, read_samples
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset

from toast.piecewise_polytrope import Lambda_of_mass
from toast.eos_by_name import Lambda_array_of_mass as Lambda_of_mass_eos

plt.rcParams.update({
    "backend": 'ps',
    "text.usetex": True,
})

model_names = lalsim.SimNeutronStarEOSNames  # list of available lal models

###############################################################################

plot_draws = False

###############################################################################

m_min, m_max, res = 1.0, 2.0, 100
mx = np.linspace(m_min, m_max, res)
pressure = np.logspace(np.log10(4e32), np.log10(2.5e35), 100)

models = ['SLy']
ys_models = calc_model(mx, Lambda_of_mass_eos, models)


def LofM(mass, log_p, Gamma_1, Gamma_2, Gamma_3):
    return Lambda_of_mass(pressure, mass, log_p, Gamma_1, Gamma_2, Gamma_3)[0]


home_dir = '/home/kris/Documents/research/3GEoS_project/'
outdir = home_dir + 'code/paper_plots/figures/'
r = 'outdir_n800_eos_12loudest'
results_file = home_dir + 'results/{}/dynesty_result.json'.format(r)

ndraws = 1000
means = []
stds = []
# ci, ys = calc_confidence_interval(results_file, mx, LofM, ndraws=ndraws)

for event in range(4,5):
    fig = plt.subplots(figsize=(4, 3))
    # fig[1].fill_between(mx, ci[:, 0], ci[:, 1], color='C2', alpha=.5,
    #                     label=r'1$\sigma\,{\rm confidence}\,{\rm interval}$')
    # if plot_draws:
    #     for j, y in enumerate(ys[0]):
    #         fig[1].plot(mx, y, color='grey', alpha=0.3, zorder=0,
    #                     label='draws' if j == 0 else None)
    
    fig[1].plot(mx, ys_models['SLy'], color='k', linewidth=1,
                label=r'${\rm SLy}$')

    samples_file = home_dir + 'data/samples/event_{}_pesummary.dat'.format(event)
    params = ['chirp_mass_source', 'mass_ratio', 'lambda_1', 'lambda_2']
    samples, params = read_samples(samples_file, params)
    cm, q, l1, l2 = samples.T
    
    dict_tmp = dict(chirp_mass=cm,
                    mass_ratio=q)
    tmp = convert_to_lal_binary_neutron_star_parameters(dict_tmp)
    m1 = tmp[0]['mass_1']
    m2 = tmp[0]['mass_2']
    
    selection = np.random.randint(0, len(m1), 1000)
    
    fig[1].scatter(m1[selection], l1[selection], alpha=0.6, marker='.', s=0.1)
    fig[1].scatter(m2[selection], l2[selection], alpha=0.6, marker='.', s=0.1)
    fig[1].set_xlim(1.1, 1.8)
    fig[1].set_ylim(0, 1000)
    fig[1].set_xlabel(r'$m\,({\rm M}_\odot)$')
    fig[1].set_ylabel(r'$\Lambda$')
    
    inset = plt.axes([0,0,1,1])
    ip = InsetPosition(fig[1], [0.55, 0.55, 0.4, 0.4])
    inset.set_axes_locator(ip)
    mark_inset(fig[1], inset, loc1=2, loc2=4, fc="none", ec='0.5')
    
    # left, bottom, width, height = [0.6, 0.6, 0.4, 0.4]
    # inset = fig[0].add_axes([left, bottom, width, height])
    inset.scatter(m2, l2, alpha=0.6, marker='.', color='C1', s=0.1)
    inset.plot(mx, ys_models['SLy'], color='k', linewidth=1,
               label=r'${\rm SLy}$')
    inset.set_xlim(1.232, 1.24)
    inset.set_ylim(675, 725)
    inset.get_xaxis().set_ticks([1.234, 1.238])
    inset.get_yaxis().set_ticks([680, 720])
    # inset.set_xlabel(r'$m\,({\rm M}_\odot)$')
    # inset.set_ylabel(r'$\Lambda$')
    
    # to make legend visible
    # fig[1].scatter(10, 0, s=3, alpha=0.8, color='C0', marker='.',
    #                label=r'${\rm NS}\,1$')
    # fig[1].scatter(10, 0, s=3, alpha=0.8, color='C1', marker='.',
    #                label=r'${\rm NS}\,2$')
    
    fig[1].legend(loc='lower left')
    fig[0].tight_layout()
    fig[0].savefig(outdir + 'Lambda_vs_mass_event{}_inset.pdf'.format(event),
                   bbox_inches='tight')
