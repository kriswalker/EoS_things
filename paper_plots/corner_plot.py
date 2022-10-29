import matplotlib.pyplot as plt
from bilby import result

plt.rcParams.update({
    "backend": 'ps',
    "text.usetex": True,
})

home_dir = '/home/kris/Documents/research/3GEoS_project/'

eos_results_file = home_dir + \
    'results/outdir_n800_eos_12loudest/dynesty_result.json'
# results = result.read_in_result(filename=eos_results_file, outdir=None,
#                                 label=None, extension='json', gzip=False)
# results.plot_corner(filename='figures/corner_12loudest.pdf',
#                     labels=[r'$\log p_1$', r'$\Gamma_1$', r'$\Gamma_2$',
#                             r'$\Gamma_3$'])

gw_results_file = home_dir + \
    'data/json_files/inj_4_data0_100-0_analysis_CECESET1ET2ET3_dynesty_result.json'
results = result.read_in_result(filename=gw_results_file, outdir=None,
                                label=None, extension='json', gzip=False)
results.plot_corner(filename='figures/corner_event4.pdf',
                    parameters=['chirp_mass', 'mass_ratio', 'lambda_1',
                                'lambda_2'])
                    # labels=[r'$\log p_1$', r'$\Gamma_1$', r'$\Gamma_2$',
                            # r'$\Gamma_3$'])