import sys
import numpy as np
from bilby import result
import bilby

import toast.piecewise_polytrope as piecewise_polytrope

import spectral_decomposition
from likelihood import EOSLikelihood

# homedir = '/fred/oz170/kwalker/projects/3G_EoS/'
homedir = '/home/kris/Documents/research/3GEoS_project/'
events_loudest = [6, 12, 5, 4, 8, 0, 2, 1, 3]

# =============================================================================
#  data options
# =============================================================================

ns = [3, 6, 9]
n = 3#ns[int(sys.argv[1])]

datafile = homedir + 'data/kdes/event_kdes_100px.npz'
events = events_loudest[:n]

# =============================================================================
#  physics options
# =============================================================================

eos = 'polytrope'
pressure_array = np.logspace(np.log10(4e32), np.log10(2.5e35), 50)
maximum_mass = 1.97
maximum_speed_of_sound = 1.1

# =============================================================================
#  sampler options
# =============================================================================

sampler = 'dynesty'
npoints = 50
npool = 1
verbose = False
resume = False

bilby_label = 'eos'
extension = '{}_{}loudest_test'.format(eos, n)
outdir = homedir + 'results/outdir_n{0}_{1}'.format(npoints, extension)

# =============================================================================
# =============================================================================
# =============================================================================

data = np.load(datafile)
inds = np.where(np.in1d(data['event_ids'], events))[0]
kde_list = data['kdes'][inds]
axes_list = data['axes'][inds]
events = data['event_ids'][inds]

logZ_list = []
for i, event in enumerate(events):
    results_file = homedir + \
        'data/json_files/inj_{}_data0_100-0_analysis'.format(event) + \
        '_CECESET1ET2ET3_dynesty_result.json'
    results = result.read_in_result(filename=results_file,
                                    outdir=None, label=None,
                                    extension='json', gzip=False)
    logZ_list.append(results.log_evidence)

priors = dict()
if eos == 'polytrope':
    priors["log_p"] = bilby.prior.Uniform(32.6, 34.4, name='log_p',
                                          latex_label="$\\log p_1$")
    priors["Gamma_1"] = bilby.prior.Uniform(2, 4.5, name='Gamma_1',
                                            latex_label="$\\Gamma_1$")
    priors["Gamma_2"] = bilby.prior.Uniform(1.1, 4.5, name='Gamma_2',
                                            latex_label="$\\Gamma_2$")
    priors["Gamma_3"] = bilby.prior.Uniform(1.1, 4.5, name='Gamma_3',
                                            latex_label="$\\Gamma_3$")
    parametrization = piecewise_polytrope

elif eos == 'spectral':
    priors["gamma_0"] = bilby.prior.Uniform(0.2, 2.0, name='gamma_0',
                                            latex_label="$\\gamma_0$")
    priors["gamma_1"] = bilby.prior.Uniform(-1.6, 1.7, name='gamma_1',
                                            latex_label="$\\gamma_1$")
    priors["gamma_2"] = bilby.prior.Uniform(-0.6, 0.6, name='gamma_2',
                                            latex_label="$\\gamma_2$")
    priors["gamma_3"] = bilby.prior.Uniform(-0.02, 0.02, name='gamma_3',
                                            latex_label="$\\gamma_3$")
    parametrization = spectral_decomposition

likelihood = EOSLikelihood(kde_list, axes_list, logZ_list, pressure_array,
                           maximum_mass, maximum_speed_of_sound,
                           parametrization, priors)

results = bilby.core.sampler.run_sampler(
    likelihood, priors=priors, sampler=sampler, label=bilby_label,
    npoints=npoints, npool=npool, verbose=verbose, resume=resume,
    outdir=outdir)
results.plot_corner()
