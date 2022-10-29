import sys
import numpy as np
import matplotlib.pyplot as plt
import lalsimulation as lalsim
from scipy.interpolate import interp1d
from bilby import result
from tqdm import tqdm
from utils import calc_confidence_interval, calc_model

from toast.piecewise_polytrope import maximum_mass
from toast.eos_by_name import maximum_mass as maximum_mass_eos

plt.rcParams.update({
    "backend": 'ps',
    "text.usetex": True,
})

model_names = lalsim.SimNeutronStarEOSNames  # list of available lal models

max_mass_model = maximum_mass_eos('SLy')

home_dir = '/home/kris/Documents/research/3GEoS_project/'
r = 'outdir_n800_eos_12loudest'
results_file = home_dir + 'results/{}/dynesty_result.json'.format(r)

ndraws = 1000
results = result.read_in_result(filename=results_file)
# maxl_params = dict(results.posterior.iloc[
#     results.posterior.log_likelihood.idxmax()])
# maxl_params.pop('log_likelihood')
# maxl_params.pop('log_prior')

ys = []
ys_prior = []
for _ in tqdm(range(ndraws)):
    draw = results.posterior.sample().to_dict('records')[0]
    draw.pop('log_likelihood')
    draw.pop('log_prior')
    ys.append(maximum_mass(**draw))
    
    draw_prior = {'log_p': np.random.uniform(32.6, 34.4),
                  'Gamma_1': np.random.uniform(2, 4.5),
                  'Gamma_2': np.random.uniform(1.1, 4.5),
                  'Gamma_3': np.random.uniform(1.1, 4.5)}
    ys_prior.append(maximum_mass(**draw_prior))
ys, ys_prior = np.array(ys), np.array(ys_prior)
ymean = np.mean(ys)
ystd = np.std(ys)

fig = plt.subplots(figsize=(4, 3))
fig[1].hist(ys, alpha=0.6, bins='auto')#, density=True)
fig[1].hist(ys_prior, alpha=0.6, bins='auto')#, density=True)
fig[1].axvline(ymean, color='k')
fig[1].axvline(ymean-ystd, color='r', linestyle='--', linewidth=0.8)
fig[1].axvline(ymean+ystd, color='r', linestyle='--', linewidth=0.8)
fig[1].set_xlim(1.9, 3.2)
fig[1].set_xlabel(r'${\rm Maximum}\,{\rm mass}\,({\rm M}_\odot)$')
fig[0].tight_layout()
fig[0].savefig(home_dir + 'code/paper_plots/figures/maximum_mass_with_prior.pdf')
