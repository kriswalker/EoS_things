import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo

from utils import read_samples, kdeND, invert

homedir = '/home/kris/Documents/research/3GEoS_project/'
datadir = homedir + 'data/'
savefile = datadir + 'kdes/event_kdes_res50_new.npz'

densities = []

logz = []
res = 100
min_max_values_list = []
m_axs = []
l_axs = []

events = np.arange(0, 13, 1)[:]
inds = np.arange(0, 13, 1)[:]

kde_list = []
axes_list = []
for i, event in enumerate(events):
    j = inds[i]

    samples_file = datadir + 'samples/event_{}_pesummary.dat'.format(event)
    samples, _ = read_samples(samples_file,
                              params=['chirp_mass_source', 'mass_ratio',
                                      'lambda_1', 'lambda_2'])
    m_samples = samples[:, :2]
    l_samples = samples[:, 2:]

    # lumdist_samples, _ = read_samples(samples_file,
    #                                   params=['luminosity_distance'])

    # fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    # ax.hist(lumdist_samples, bins='auto', density=True)
    # ax.set_xlabel(r'$D_{\rm lum}$')
    # fig.tight_layout()

    # lum_dist = lambda z: cosmo.luminosity_distance(z).value
    # z_samples = np.array([invert(s, lum_dist, 0.1) for s in lumdist_samples])
    # m_samples[:, 0] /= (1 + z_samples)

    mkde, maxes = kdeND(m_samples, 'gaussian', 0.03, res)
    lkde, laxes = kdeND(l_samples, 'tophat', 0.03, res)
    kde_list.append((mkde, lkde))
    axes_list.append((maxes, laxes))

    ax1 = plt.subplot(121)
    ax1.pcolormesh(maxes[0], maxes[1], np.exp(mkde))
    inds = np.random.choice(np.arange(0, np.shape(samples)[0]), 1000)
    ax1.scatter(m_samples[inds, 0], m_samples[inds, 1], color='r',
                alpha=0.5, marker='.')
    ax1.set_xlabel(r'$\mathcal{M}$')
    ax1.set_ylabel(r'$q$')

    ax2, ax3 = plt.subplot(222), plt.subplot(224)
    ax2.hist(m_samples[:, 0], bins='auto', density=True)
    ax2.plot(maxes[0], np.sum(np.exp(mkde), axis=0) * np.diff(maxes[1])[0])
    ax2.set_xlabel(r'$\mathcal{M}$')
    ax3.hist(m_samples[:, 1], bins='auto', density=True)
    ax3.plot(maxes[1], np.sum(np.exp(mkde), axis=1) * np.diff(maxes[0])[0])
    ax3.set_xlabel(r'$q$')
    plt.tight_layout()
    plt.show()

    ax1 = plt.subplot(121)
    ax1.pcolormesh(laxes[0], laxes[1], np.exp(lkde))
    ax1.scatter(l_samples[inds, 0], l_samples[inds, 1], color='r',
                alpha=0.5, marker='.')
    ax1.set_xlabel(r'$\Lambda_1$')
    ax1.set_ylabel(r'$\Lambda_2$')

    ax2, ax3 = plt.subplot(222), plt.subplot(224)
    ax2.hist(l_samples[:, 0], bins='auto', density=True)
    ax2.plot(laxes[0], np.sum(np.exp(lkde), axis=0) * np.diff(laxes[1])[0])
    ax2.set_xlabel(r'$\Lambda_1$')
    ax3.hist(l_samples[:, 1], bins='auto', density=True)
    ax3.plot(laxes[1], np.sum(np.exp(lkde), axis=1) * np.diff(laxes[0])[0])
    ax3.set_xlabel(r'$\Lambda_2$')
    plt.tight_layout()
    plt.show()

np.savez(savefile, kdes=np.array(kde_list), axes=np.array(axes_list),
         event_ids=events)
