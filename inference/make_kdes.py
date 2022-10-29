import numpy as np
from kde_posterior import read_samples, interpolate
import matplotlib.pyplot as plt

homedir = '/home/kris/Documents/research/3GEoS_project/'
datadir = homedir + 'data/'
savefile = datadir + 'kdes/event_kdes_res50_new.npz'

densities = []

logz = []
res = 50
factors = np.array([(1, 0.05), (1, 0.05), (1, 0.05),
                    (1, 0.05), (1, 0.05), (1, 0.05),
                    (1, 0.05), (1, 0.05), (1, 0.05),
                    (1, 0.05), (1, 0.05), (1, 0.05),
                    (1, 0.05)
                    ])
min_max_values_list = []
m_axs = []
l_axs = []
events = np.arange(0, 13, 1)
inds = np.arange(0, 13, 1)
ncols = 4
fm = plt.subplots(int(np.ceil(len(events)/ncols)), ncols, figsize=(12, 9))
fm_ = plt.subplots(int(np.ceil(len(events)/ncols)), 2*ncols, figsize=(24, 9))
fl = plt.subplots(int(np.ceil(len(events)/ncols)), ncols, figsize=(12, 9))
fl_ = plt.subplots(int(np.ceil(len(events)/ncols)), 2*ncols, figsize=(24, 9))
for i, event in enumerate(events):
    j = inds[i]

    samples_file = datadir + 'samples/event_{}_pesummary.dat'.format(event)
    density, m_ax, l_ax = interpolate(samples_file, res=res, bw=factors[i],
                                      plot=False, sup_data=None, event=event)
    densities.append(density)
    m_axs.append(m_ax)
    l_axs.append(l_ax)
    cm = m_ax[0]
    q = m_ax[1]
    l1 = l_ax[0]
    l2 = l_ax[1]

    mmvl = {}
    mmvl["min_cm"] = min(cm)
    mmvl["max_cm"] = max(cm)
    mmvl["min_q"] = min(q)
    mmvl["max_q"] = max(q)
    mmvl["min_lambda_1"] = min(l1)
    mmvl["max_lambda_1"] = max(l1)
    mmvl["min_lambda_2"] = min(l2)
    mmvl["max_lambda_2"] = max(l2)
    min_max_values_list.append(mmvl)

    data, params = read_samples(samples_file)
    cm_samp = data[:, np.argwhere((params == 'chirp_mass_source'))[0]]
    q_samp = data[:, np.argwhere((params == 'mass_ratio'))[0]]
    l1_samp = data[:, np.argwhere((params == 'lambda_1'))[0]]
    l2_samp = data[:, np.argwhere((params == 'lambda_2'))[0]]

    row, col = i // ncols, i % ncols
    row_, col_ = (2 * i) // (2 * ncols), (2 * i) % (2 * ncols)
    lindensity = (np.exp(density[0]), np.exp(density[1]))
    fm[1][row][col].pcolormesh(m_ax[0], m_ax[1], lindensity[0])
    fm[1][row][col].scatter(cm_samp, q_samp, alpha=0.5)
    fl[1][row][col].pcolormesh(l_ax[0], l_ax[1], lindensity[1])
    fl[1][row][col].scatter(l1_samp, l2_samp, alpha=0.5)

    fm_[1][row_][col_].hist(cm_samp, bins='auto', density=True, alpha=0.7)
    fm_[1][row_][col_].plot(m_ax[0], np.sum(lindensity[0], axis=0) *
                            np.diff(m_ax[1])[0], alpha=0.7)
    fm_[1][row_][col_+1].hist(q_samp, bins='auto', density=True, alpha=0.7)
    fm_[1][row_][col_+1].plot(m_ax[1], np.sum(lindensity[0], axis=1) *
                              np.diff(m_ax[0])[0], alpha=0.7)

    fl_[1][row_][col_].hist(l1_samp, bins='auto', density=True, alpha=0.7)
    fl_[1][row_][col_].plot(l_ax[0], np.sum(lindensity[1], axis=0) *
                            np.diff(l_ax[1])[0], alpha=0.7)
    fl_[1][row_][col_+1].hist(l2_samp, bins='auto', density=True, alpha=0.7)
    fl_[1][row_][col_+1].plot(l_ax[1], np.sum(lindensity[1], axis=1) *
                              np.diff(l_ax[0])[0], alpha=0.7)
for f in [(fm, 'mass_2d_res50_new'), (fl, 'lambda_2d_res50_new'),
          (fm_, 'mass_1d_res50_new'), (fl_, 'lambda_1d_res50_new')]:
    f[0][0].tight_layout()
    f[0][0].savefig(homedir + 'plots/kde_plots/' + f[1] + '.png')

np.savez(savefile, densities=np.array(densities),
         m_axs=np.array(m_axs), l_axs=np.array(l_axs))
