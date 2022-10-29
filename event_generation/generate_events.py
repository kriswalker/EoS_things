from toast.eos_by_name import Lambda_of_mass
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import tqdm
from tabulate import tabulate
import bilby
import lal

from utils import ce_noise, redshift_sample, antennas, get_waveform, snrfunc, \
    mass_to_chirp_mass, invert, horizon_distance, merger_rate_density, \
    redshift_pdf, integrate, lum_dist, com_dist

###############################################################################
# OPTIONS
###############################################################################

COSMIC_EXPLORER = {'det_coords': np.array([[0, 0]]),
                   'det_angles': np.array([[0, np.pi/2]]),
                   'det_noises': [ce_noise],
                   'det_lengths': np.array([40e3]),
                   'label': 'CE',
                   }

detector_list = [COSMIC_EXPLORER]

num_events = int(1e1)  # number of events to generate

from_inj_file, inj_infile = False, './data/inj_file_1.dat'
from_z_file, z_infile = True, './data/z_samples.txt'
write_inj_file, inj_outfile, ninj = False, './data/inj_file.dat', 5

###############################################################################


if not from_inj_file:
    # sample angles
    theta_rand = np.arccos(np.random.uniform(low=-1, high=1, size=num_events))
    phi_rand = np.random.uniform(low=0, high=2*np.pi, size=num_events)
    psi_rand = np.random.uniform(low=0, high=2*np.pi, size=num_events)
    iota_rand = np.arccos(np.random.uniform(low=-1, high=1, size=num_events))
    event_coords = np.vstack([theta_rand,
                              phi_rand,
                              psi_rand,
                              iota_rand]).T

    # sample redshifts
    z_sample = []
    if from_z_file:
        z_file = open(z_infile, 'r')
        for line in z_file:
            z_sample.append(float(line.split()[0]))
        z_sample = np.random.choice(z_sample, size=num_events)
    else:
        z_sample = redshift_sample(nsamp=num_events, bounds=[0, 25])
        
    z_sample = 0.01 * np.ones(num_events)

    # sample masses
    lower, upper = 1.0, 2.35  # NS mass bounds
    mean, std = 1.33, 0.11  # mean mass and standard deviation
    X = st.truncnorm((lower-mean)/std, (upper-mean)/std, loc=mean, scale=std)
    masses = X.rvs(2 * num_events)
    masses = np.reshape(masses, (num_events, 2))  # sort into binary pairs

else:
    datafile = open(inj_infile, 'r')
    data = []
    for i, line in enumerate(datafile):
        if i > 0:
            data.append([float(x) for x in line.split()])
        else:
            headers = line.split()
    data = np.array(data)
    num_events = np.shape(data)[0]

    chirp_mass = data[:, 0]
    q = data[:, 1]
    ld = data[:, 8]
    dec = data[:, 9]
    ra = data[:, 10]
    cos_theta_jn = data[:, 11]
    psi = data[:, 12]
    geocent_time = data[:, 16]

    num_source = np.shape(data)[0]

    gmst = []
    for t in geocent_time:
        gmst.append(lal.GreenwichMeanSiderealTime(int(t)))
    gmst = np.array(gmst)

    theta_rand, phi_rand = bilby.core.utils.ra_dec_to_theta_phi(ra, dec, gmst)
    phi_rand = phi_rand % 2*np.pi
    psi_rand = psi
    iota_rand = np.arccos(cos_theta_jn)
    event_coords = np.vstack([theta_rand,
                              phi_rand,
                              psi_rand,
                              iota_rand]).T

    z_sample = []
    for d in ld:
        z_sample.append(invert(d, lum_dist, 1))
    z_sample = np.array(z_sample)

    alpha = (q**3 / (1 + q))**(1/5)
    m2 = chirp_mass / alpha
    m1 = q * m2
    masses = np.hstack((m1, m2))
    masses = np.reshape(masses, (int(len(masses)/2), 2))

# input event parameters
for detector in detector_list:

    det_coords = detector['det_coords']
    det_angles = detector['det_angles']

    aa = (1+np.cos(iota_rand)**2) / 2
    bb = 1j * np.cos(iota_rand)
    hplus = aa * np.cos(2*psi_rand) + bb * np.sin(2*psi_rand)
    hcross = aa * np.sin(2*psi_rand) - bb * np.cos(2*psi_rand)

    detector['det_plus'], detector['det_cross'] = antennas(event_coords,
                                                           det_coords,
                                                           det_angles)
    detector['event_extinctions'] = (
        np.einsum('...a, ...', detector['det_plus']**2, np.abs(aa)**2) +
        np.einsum('...a, ...', detector['det_cross']**2, np.abs(bb)**2))
    detector['snrs'] = np.zeros(num_events)

    detector['event_draws'] = np.zeros((num_events, 7))
    detector['event_draws'][:, 0] = masses[:, 0]
    detector['event_draws'][:, 1] = masses[:, 1]
    detector['event_draws'][:, 2] = z_sample
    detector['event_draws'][:, 3] = theta_rand
    detector['event_draws'][:, 4] = phi_rand
    detector['event_draws'][:, 5] = psi_rand
    detector['event_draws'][:, 6] = iota_rand

# calculate snrs
zz0 = 0.001
with tqdm.tqdm(range(num_events*len(detector_list))) as pbar:
    for i in range(num_events):
        ff, mywf = get_waveform(m1=masses[i, 0], m2=masses[i, 1],
                                distance=lum_dist(zz0))
        for detector in detector_list:
            snr_i = np.squeeze(snrfunc(z_sample[i], zz0, ff, mywf,
                                       detector['event_extinctions'][i],
                                       detector['det_noises']))
            detector['snrs'][i] = snr_i
            pbar.update()

for detector in detector_list:

    # cumulative snr
    snrs = detector['snrs']
    cum_snr = [snrs[0]]
    for i, snr in enumerate(snrs[1:]):
        cum_snr.append(np.sqrt(cum_snr[i]**2 + snr**2))
    cum_snr = np.array(cum_snr)
    detector['cum_snr'] = cum_snr

    nobs = np.linspace(1, num_events, num_events)  # observation number

    # fit a line in log space
    p = np.polyfit(np.log10(nobs), np.log10(cum_snr), 1)
    m = (max(np.log10(cum_snr)) - min(np.log10(cum_snr))) / \
        (max(np.log10(nobs)) - min(np.log10(nobs)))
    b = max(np.log10(cum_snr)) - m * max(np.log10(nobs))
    print('m = {0}, 10^b = {1}'.format(round(m, 2), round(10**b, 2)))

    # plot cumulative snr
    fig1, ax1 = plt.subplots()
    ax1.scatter(nobs, cum_snr, marker='.')
    ax1.plot(nobs, (10**b)*(nobs**m), color='r')
    ax1.set_xlabel('No. of observations')
    ax1.set_ylabel('cumulative snr')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    fig1.tight_layout()
    plt.show()

    if num_events >= 100:  # plot histogram of snrs
        fig2, ax2 = plt.subplots()
        lim = 30
        snrs_ = snrs[np.argwhere(snrs <= lim)].flatten()
        ax2.hist(snrs, bins=lim*5, density=True)
        ax2.set_xlim(0, lim)
        ax2.set_xlabel('snr')
        ax2.set_yscale('log')
        fig2.tight_layout()
        plt.show()

    event_draws_ = np.insert(detector['event_draws'], 0, snrs, axis=1)
    event_draws_ = np.insert(event_draws_, 0, np.arange(0, num_events), axis=1)
    summary = event_draws_[np.flip(np.argsort(snrs), axis=0)]
    print(tabulate(summary[:20],  headers=['ID', 'snr', 'm1', 'm2', 'z',
                                           'theta', 'phi', 'psi', 'iota']))

    if write_inj_file:
        inj = open(inj_outfile, 'w')
        inj.write('chirp_mass mass_ratio a_1 a_2 tilt_1 tilt_2 phi_12 phi_jl' +
                  ' luminosity_distance dec ra cos_theta_jn psi phase' +
                  ' lambda_1 lambda_2 geocent_time\n')

        event_draws = detector['event_draws']
        m_ordered = np.flip(np.sort(event_draws[:ninj, 0:2], axis=1), axis=1)
        m1, m2 = m_ordered[:, 0], m_ordered[:, 1]
        cm = mass_to_chirp_mass(m1, m2)
        q = m2 / m1

        ld = lum_dist(event_draws[:ninj, 2])

        theta, phi = event_draws[:ninj, 3], event_draws[:ninj, 4]
        ra, dec = bilby.core.utils.theta_phi_to_ra_dec(theta, phi, 100)

        psi = event_draws[:ninj, 5]
        iota = event_draws[:ninj, 6]
        phase = np.random.uniform(0, 2*np.pi, size=ninj)

        lambda1 = np.array([Lambda_of_mass(m1i, 'SLy') for m1i in m1])
        lambda2 = np.array([Lambda_of_mass(m2i, 'SLy') for m2i in m2])

        z = []
        for dist in ld:
            z.append(invert(dist, lum_dist, 0.01))
        z = np.array(z)

        inj_params = np.array([cm*(1+z),
                               q,
                               np.zeros(ninj),
                               np.zeros(ninj),
                               np.zeros(ninj),
                               np.zeros(ninj),
                               np.zeros(ninj),
                               np.zeros(ninj),
                               ld,
                               dec,
                               ra % (2 * np.pi),
                               np.cos(iota),
                               psi,
                               phase,
                               lambda1,
                               lambda2,
                               100*np.ones(ninj).astype(int)]).T

        nparams = np.shape(inj_params)[1]
        for params in inj_params:
            injstr = (nparams * '{} ')[:-1].format(*params)
            injstr += '\n'
            inj.write(injstr)
        inj.close()

zhorizon = horizon_distance(COSMIC_EXPLORER, masses=[2.3, 2.3])
dhorizon = com_dist(zhorizon) / 1000
print('Horizon redshift = {0}\nHorizon distance = {1} Gpc'.format(
    zhorizon, dhorizon))

zz = np.linspace(0, zhorizon, 100)
mrd = merger_rate_density(zz)
dVdz = redshift_pdf(zz)[1]
horizon_vol = integrate(dVdz, zz)
print('Horizon volume =', horizon_vol)
print(4 * np.pi * dhorizon**3 / 3)

mergers_per_year = integrate(mrd * dVdz / (1 + zz), zz)
print('merger rate = {} / year'.format(mergers_per_year))

print('mrd at z=0 =', merger_rate_density(0))
