import numpy as np
import scipy.constants as scc
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.interpolate import interp1d
import scipy.stats as st

from astropy import constants as const
from astropy.cosmology import Planck15 as cosmo
from pycbc import waveform as wf

# Noise functions
f_ce_nu, strain_ce_nu = np.loadtxt('./data/ce_wb.dat').T
f_ce, strain_ce = np.loadtxt('./data/ce_wb_unfsr.dat').T
f_et, strain_et = np.loadtxt('./data/et_d.dat').T
f_voy, strain_voy = np.loadtxt('./data/voyager.dat').T
f_aligo, strain_aligo = np.loadtxt('./data/aligo.dat').T

def ce_noise(ff):
    return np.interp(ff, f_ce, strain_ce, left=1e-19, right=np.Inf)

def ce_noise_nu(ff):
    return np.interp(ff, f_ce_nu, strain_ce_nu, left=1e-19, right=np.Inf)

def et_noise(ff):
    return np.interp(ff, f_et[f_et>2], strain_et[f_et>2], left=np.Inf, right=np.Inf)

def voy_noise(ff):
    return np.interp(ff, f_voy, strain_voy, left=1e-19, right=np.Inf)

def aligo_noise(ff):
    return np.interp(ff, f_aligo, strain_aligo, left=1e-19, right=np.Inf)

det_noises = [aligo_noise, voy_noise, et_noise, ce_noise]
det_noises_nu = [aligo_noise, voy_noise, et_noise, ce_noise_nu]

mSun = const.M_sun.value

lum_dist = lambda z: cosmo.luminosity_distance(z).value
com_dist = lambda z: cosmo.comoving_distance(z).value
comov_vol = lambda z: cosmo.comoving_volume(z).value

def get_waveform(m1=30, m2=30, phic=0, distance=1e3):
    fisco = scc.c**3/(scc.G*6**1.5*2*np.pi*(m1+m2)*mSun)
    df = 2**(np.max([np.floor(np.log(fisco/1024)/np.log(2)), -4]))
    mywf_p, mywf_c = wf.get_fd_waveform(
                        approximant='IMRPhenomPv2_NRTidal',
                        mass1=m1,
                        mass2=m2,
                        spin1z=0,
                        spin2z=0,
                        coa_phase=phic,
                        distance=distance,
                        delta_f = df,
                        f_lower = 0.5,
                        f_final = 0,
        )
    ff = np.array(mywf_p.sample_frequencies)
    mywfp = np.array(mywf_p.data)[ff>=1]
    #mywfc = np.array(mywf_c.data)[ff>=2]
    ff = ff[ff>=1]
    return ff, mywfp#, mywfc

def antennas(source_coords, det_coords, det_angles, freq=None):
    '''Return the plus and cross antenna patterns for GW sources incident on a detector network.
    Frequency dependence is not currently implemented.
    
    Arguments:
        source_coords (array): array of coordinates for GWs, each specified as [theta, phi, psi, iota],
            where theta is the colatitude, phi is the longitude, and psi is the polarization angle
        det_coords (array): array of coordinates for detectors, each specified as [theta, phi]
        det_angles (array): array of angles describing detector arms, each specified as [rho, alpha],
            where rho is the angle of the X-arm wrt due east, and alpha is the detector opening angle
        freq (scalar): frequency at which to compute the antenna pattern (Essick et al. 2017)
        
    Returns:
        det_plus (array): plus antenna pattern
        det_cross (array): cross antenna pattern
    
    '''
    
    source_theta, source_phi, source_psi, _ = source_coords.T
    
    # Construct GW direction vectors
    source_n = np.array([
            np.sin(source_theta)*np.cos(source_phi),
            np.sin(source_theta)*np.sin(source_phi),
            np.cos(source_theta),
        ]).T
    source_ex = np.array([
            np.sin(source_phi) * np.cos(source_psi) - np.cos(source_theta) * np.cos(source_phi) * np.sin(source_psi),
            -np.cos(source_phi) * np.cos(source_psi) - np.cos(source_theta) * np.sin(source_phi) * np.sin(source_psi),
            np.sin(source_theta) * np.sin(source_psi),
        ]).T
    source_ey = np.array([
            -np.sin(source_phi) * np.sin(source_psi) - np.cos(source_theta) * np.cos(source_phi) * np.cos(source_psi),
            np.cos(source_phi) * np.sin(source_psi) - np.cos(source_theta) * np.sin(source_phi) * np.cos(source_psi),
            np.sin(source_theta) * np.cos(source_psi),
        ]).T
    
    # Construct detector direction vectors
    det_theta, det_phi = det_coords.T
    det_rho, det_alpha = det_angles.T
    
    det_n = np.array([
            np.sin(det_theta)*np.cos(det_phi),
            np.sin(det_theta)*np.sin(det_phi),
            np.cos(det_theta),
        ]).T
    det_ex = np.array([
            np.sin(det_phi) * np.cos(-det_rho) - np.cos(det_theta) * np.cos(det_phi) * np.sin(-det_rho),
            -np.cos(det_phi) * np.cos(-det_rho) - np.cos(det_theta) * np.sin(det_phi) * np.sin(-det_rho),
            np.sin(det_theta) * np.sin(-det_rho),
        ]).T
    det_ey = np.array([
            np.sin(det_phi) * np.cos(-det_rho-det_alpha) - np.cos(det_theta) * np.cos(det_phi) * np.sin(-det_rho-det_alpha),
            -np.cos(det_phi) * np.cos(-det_rho-det_alpha) - np.cos(det_theta) * np.sin(det_phi) * np.sin(-det_rho-det_alpha),
            np.sin(det_theta) * np.sin(-det_rho-det_alpha),
        ]).T
    
    # Construct source tensors
    source_plus = np.einsum('...a,...b', source_ex, source_ex) - np.einsum('...a,...b', source_ey, source_ey)
    source_cross = np.einsum('...a,...b', source_ex, source_ey) + np.einsum('...a,...b', source_ey, source_ex)

    # Construct detector freq resp
    if freq is None:
        Dx = 0.5*np.ones((source_n.shape[0], det_ex.shape[0]))
        Dy = 0.5*np.ones((source_n.shape[0], det_ex.shape[0]))
    else:
        LL = det_dict['det_lengths'][0]
        nx = np.einsum('ab,cb -> ac', source_n, det_ex)
        ny = np.einsum('ab,cb -> ac', source_n, det_ey)
        Dx = (scc.c / (8j * np.pi * freq * LL)
              * ((1 - np.exp(-2j*np.pi*freq*(1-nx)*LL/scc.c)) / (1 - nx)
              - np.exp(-4j*np.pi*freq*LL/scc.c) * (1 - np.exp(2j*np.pi*freq*(1+nx)*LL/scc.c)) / (1 + nx)))
        Dy = (scc.c / (8j * np.pi * freq * LL)
              * ((1 - np.exp(-2j*np.pi*freq*(1-ny)*LL/scc.c)) / (1 - ny)
              - np.exp(-4j*np.pi*freq*LL/scc.c) * (1 - np.exp(2j*np.pi*freq*(1+ny)*LL/scc.c)) / (1 + ny)))
    
    Dx2 = np.einsum('ab,bc,bd->abcd', Dx, det_ex, det_ex)
    Dy2 = np.einsum('ab,bc,bd->abcd', Dy, det_ey, det_ey)

    #Construct detector tensor
    det_tensor = Dx2 - Dy2
    
    # Construct detector antenna responses
    det_plus = np.einsum('abc, adbc -> ad', source_plus, det_tensor)
    det_cross = np.einsum('abc, adbc -> ad', source_cross, det_tensor)
    
    det_plus = np.einsum('abc, adbc -> ad', source_plus, det_tensor)
    det_cross = np.einsum('abc, adbc -> ad', source_cross, det_tensor)
    
    return det_plus, det_cross

h0 = 1e-24

def snr_simple(ff, det_noises, source_extinctions):
    det_noises_sq_arr = np.array([np.square(det_noises[kk](ff)) for kk in range(len(det_noises))])
    weights = np.einsum('ba,ac->bc', h0**2 * source_extinctions, np.reciprocal(det_noises_sq_arr))
    return weights

lum_dist = lambda z: cosmo.luminosity_distance(z).value

def snrfunc(zz, zz0, ff0, waveform0, source_extinctions, det_noises):
    lum_dist0 = lum_dist(zz0)
    ff = ff0/(1+zz)
    det_noises_sq_arr = np.array([np.square(det_noises[kk](ff)) for kk in range(len(det_noises))])
    weights = np.einsum('a,ab', source_extinctions, np.reciprocal(det_noises_sq_arr))
    intgnd = waveform0*lum_dist0/lum_dist(zz)*(1+zz)**2
    snrsq = 4 * np.trapz(weights*(np.square(np.real(intgnd))+np.square(np.imag(intgnd))), ff)
    return np.sqrt(np.max([snrsq, 0]))

def minfunc(zz, zz0, ff0, waveform0, source_extinctions, det_noises):
    lum_dist0 = lum_dist(zz0)
    ff = ff0/(1+zz)
    det_noises_sq_arr = np.array([np.square(det_noises[kk](ff)) for kk in range(len(det_noises))])
    weights = np.einsum('a,ab', source_extinctions, np.reciprocal(det_noises_sq_arr))
    intgnd = waveform0*lum_dist0/(1e-3+lum_dist(zz))*(1+zz)**2
    snrsq = 4 * np.trapz(weights*(np.square(np.real(intgnd))+np.square(np.imag(intgnd))), ff)
    return np.square(np.max([snrsq, 0]) - 64)

def horizon_distance(det_dict, masses, horizon_snr=8, zguess=20):
    zz0 = 0.001
    ff, mywf = get_waveform(m1=masses[0], m2=masses[1], distance=lum_dist(zz0))
    source_coords = np.vstack([np.array([0]), np.array([0]), np.array([0]), np.array([0])]).T
    det_plus, det_cross = antennas(source_coords, det_dict['det_coords'], det_dict['det_angles'])
    source_extinctions = (np.einsum('...a, ...', det_plus**2, np.array([1])) 
                          + np.einsum('...a, ...', det_cross**2, np.array([1])))
    def min_func(z, hsnr):
        snr = snrfunc(z, zz0, ff, mywf, source_extinctions[0],
                                          det_dict['det_noises'])
        return abs(hsnr - snr)
    
    sol = minimize(min_func, zguess, (horizon_snr))
    return sol.x[0]
    
def integrate(y, x):
    return np.sum(y[:-1] * np.diff(x))

def invert(y, func, xguess):
    def min_func(x):
        return abs(y - func(x))
    sol = minimize(min_func, xguess)
    return sol.x[0]

def normalize(y, x, norm=None):
    if norm is None:
        integral = integrate(y, x)
    else:
        integral = norm
    return y / integral
    
def merger_rate_density(redshift, zmin=25, lam=-2e-3, tmin=0.1, gamma=-1):
    
    SFR = lambda z : 1e9 * 0.015 * (1 + z)**2.7 / (1 + ((1 + z) / 2.9)**5.6) # Msol/Gpc/yr
    DTD = lambda z2, z1, t, gam : (cosmo.lookback_time(z2).value - 
                                        cosmo.lookback_time(z1).value - t)**gam
    dtdz = lambda z : -1 / ((1 + z) * cosmo.H(z).value)
    
    integrand = lambda zp, z, t, gam : lam * DTD(z, zp, t, gam) * SFR(zp) * dtdz(zp)
    
    if type(redshift) == int or type(redshift) == float:
        return quad(integrand, zmin, redshift, args=(redshift, tmin, gamma))[0]
    else:
        merger_rate = []
        for zz in redshift:
            merger_rate.append(quad(integrand, zmin, zz, args=(zz, tmin, gamma))[0])
        return np.array(merger_rate)
    
def redshift_pdf(z, norm=None):
    dVdz = 1e-12 * 4 * np.pi * const.c.value * lum_dist(z)**2 / ((1 + z)**2 * cosmo.H(z).value) # Gpc^3
    return normalize((dVdz / (1 + z)) * merger_rate_density(z, zmin=25, lam=-2e-3,
                                   tmin=0.1, gamma=-1), z, norm), dVdz

def redshift_sample(nsamp=1000, bounds=[0, 25]):
    
    z_coarse = np.linspace(bounds[0], bounds[1], 1000)
    zpdf_coarse = redshift_pdf(z_coarse)[0]
    zpdf_interp = interp1d(z_coarse, zpdf_coarse)
    
    class z_pdf(st.rv_continuous):
        def _pdf(self, z):
            return zpdf_interp(z)
    
    pdf = z_pdf(a=bounds[0], b=bounds[1])
    return pdf.rvs(size=nsamp)

def mass_to_chirp_mass(m1, m2):
    return (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
    
    
    
    