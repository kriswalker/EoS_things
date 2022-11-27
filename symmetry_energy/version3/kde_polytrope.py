import sys
import numpy as np
import matplotlib.pyplot as plt
from bilby import result
from scipy.stats import gaussian_kde
from utils import kde2D, kdeND

home_dir = '/home/kris/Documents/research/3GEoS_project/'
r = 'outdir_n800_eos_12loudest'
results_file = home_dir + 'results/{}/dynesty_result.json'.format(r)

results = result.read_in_result(filename=results_file)
data = results.samples[:, :3]
kde, axes = kdeND(data, 'epanechnikov', 0.1, 100)
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.pcolormesh(axes[0], axes[2], np.sum(np.exp(kde), axis=0))
# ax.scatter(data[:, 0], data[:, 1], alpha=0.5, marker='.')
# ax.set_ylabel(r'$p\,({\rm Pa})$')
# ax.set_xlabel(r'$\rho\,({\rm kg}\,{\rm m}^{-3})$')
# fig.tight_layout()

np.savez(home_dir + 'data/kdes/polytrope_kde.npz', kde=kde, axes=axes)
