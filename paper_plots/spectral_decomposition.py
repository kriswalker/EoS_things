import lalsimulation as lalsim
from lal import MSUN_SI, G_SI, C_SI
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters
from bilby.core.utils import logger


def Lambda_of_central_pressure(central_pressure, gamma_0, gamma_1, gamma_2,
                               gamma_3):
    '''
    Parameters
    ----------
    central_pressure = float
        central pressure in SI units
    gamma_0, gamma_1, gamma_2, gamma_3 = float
        spectral decomposition parameters in SI units

    Returns
    -------
    Mass in solar masses and the dimensionless tidal deformability
    '''
    polytrope = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(
            gamma_0, gamma_1, gamma_2, gamma_3)
    radius, mass, k2 = lalsim.SimNeutronStarTOVODEIntegrate(central_pressure,
                                                            polytrope)

    Lambda = (2/3) * k2 * (C_SI**2 * radius / (G_SI*mass))**5
    mass /= MSUN_SI
    return mass, Lambda


def Lambda_array_of_central_pressure(
        central_pressure_array, gamma_0, gamma_1, gamma_2, gamma_3,
        maximum_mass_limit=1.97):
    '''
    This function returns Lambda, mass and the maximum mass
    given an EoS.
    If the maximum mass is below maximum_mass_limit, we set
    Lambda and mass to np.nan because we usually only care about
    EoS with maximum masses above maximum_mass_limit

    Parameters
    ----------
    pressure_array: numpy array
        array of central pressures in SI units
    gamma_0, gamma_1, gamma_2, gamma_3 = float
        spectral decomposition parameters
    maximum_mass_limit: float
        maximum mass of the EoS, set by default to 1.97
        consistent with pulsar observations.
        If maximum_mass<maximum_mass, Lambda=np.nan and
        mass=np.nan

    Returns
    -------
    Mass: np array
        mass in solar masses and the
    Lambda: numpy array
        dimensionless tidal deformability
    max_mass: float
        neutron star maximum mass
    '''
    tmp = np.array(
        [Lambda_of_central_pressure(pp, gamma_0, gamma_1, gamma_2, gamma_3) for
         pp in central_pressure_array]
        )
    mass = tmp[:, 0]
    Lambda = tmp[:, 1]

    arg_maximum_mass = np.argmax(mass)
    max_mass = mass[arg_maximum_mass]

    if max_mass >= maximum_mass_limit:
        # Choose masses between 1. and maximum mass
        args = np.argwhere((mass >= 1.)).flatten()
        mass = mass[args[0]:arg_maximum_mass]
        Lambda = Lambda[args[0]:arg_maximum_mass]
    else:
        # We do not consider maximum masses below 1.6 msun.
        # Usually, EoSs with maximum masses<=1.97 msun are removed
        # in the prior
        mass = np.nan
        Lambda = np.nan

    return mass, Lambda, max_mass


def maximum_mass(
        gamma_0, gamma_1, gamma_2, gamma_3):
    '''
    Parameters
    ----------
    gamma_0, gamma_1, gamma_2, gamma_3 = float
        spectral decomposition hyper-parameters in SI units

    Returns
    -------
    Maximum mass in solar masses
    '''
    polytrope = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(
        gamma_0, gamma_1, gamma_2, gamma_3)
    polytrope_family = lalsim.CreateSimNeutronStarFamily(polytrope)
    max_mass = lalsim.SimNeutronStarMaximumMass(polytrope_family)/MSUN_SI

    return max_mass


def maximum_speed_of_sound(
        gamma_0, gamma_1, gamma_2, gamma_3):
    '''
    Parameters
    ----------
    gamma_0, gamma_1, gamma_2, gamma_3 = spectral decomposition
    hyper parameters

    Returns
    -------
    Maximum speed of sound divided by the speed of light
    '''
    polytrope = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(
        gamma_0, gamma_1, gamma_2, gamma_3)
    max_enthalpy = lalsim.SimNeutronStarEOSMaxPseudoEnthalpy(polytrope)
    max_speed_of_sound = lalsim.SimNeutronStarEOSSpeedOfSound(max_enthalpy,
                                                              polytrope)

    return max_speed_of_sound/C_SI


def maximum_pressure(
        gamma_0, gamma_1, gamma_2, gamma_3):
    '''
    Parameters
    ----------
    gamma_0, gamma_1, gamma_2, gamma_3 = float
        spectral decomposition hyper-parameters in SI units

    Returns
    -------
    returns the maximum pressure in SI units
    '''
    polytrope = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(
        gamma_0, gamma_1, gamma_2, gamma_3)
    max_pressure = lalsim.SimNeutronStarEOSMaxPressure(polytrope)

    return max_pressure


def Lambda_of_mass(
        central_pressure_array, mass, gamma_0, gamma_1, gamma_2, gamma_3,
        maximum_mass_limit=1.97):
    '''
    We interpolate the Lambda_array_from_central_pressure function
    to obtain Lambda,  given a mass, an array of central pressures
    and an EoS model.

    If the maximum_mass is below the maximum_mass_limit supported by
    pulsar observations, we set Lambda=np.nan. These values
    should be ruled out by the prior
    If mass > maximum_mass the neutron star collapses to a
    black hole, therefore Lambda=0

    Parameters
    ----------
    central_pressure_array = numpy array
        central pressure array in SI units
    mass= numpy array
        mass of a neutron star in solar masses
    gamma_0, gamma_1, gamma_2, gamma_3 = float
        spectral decomposition hyper-parameters in SI units
    maximum_mass_limit: maximum mass of the EoS, set by default to 1.97
        consistent with pulsar observations

    Returns
    -------
    Lambda: numpy array or float
        Dimensionless tidal deformability
    max_mass: float
        maximum mass of an EoS model
    '''
    mass_tmp, Lambda_tmp, max_mass = Lambda_array_of_central_pressure(
        central_pressure_array, gamma_0, gamma_1, gamma_2, gamma_3,
        maximum_mass_limit)

    if max_mass >= maximum_mass_limit:
        if hasattr(mass, '__len__'):
            if mass.shape == (len(mass),):  # i.e. np.array([mass_1])
                interpolated_Lambda = interp1d(
                    mass_tmp, Lambda_tmp, fill_value='extrapolate')

                Lambda = interpolated_Lambda(mass)
                args = np.argwhere(Lambda >= 0).flatten()
                args_2 = np.argwhere(Lambda < 0).flatten()

                # We append zeros after a NS collapses, i.e, for masses > M_tov
                Lambda = np.append(Lambda[args], np.zeros(len(args_2)))

            elif mass.shape == (2, len(mass[0])):
                interpolated_Lambda = interp1d(
                    mass_tmp, Lambda_tmp, fill_value='extrapolate')
                Lambda = np.zeros([2, len(mass[0])])
                for ii in range(2):
                    Lambda_tmp_2 = interpolated_Lambda(mass[ii])
                    args = np.argwhere(Lambda_tmp_2 >= 0).flatten()
                    args_2 = np.argwhere(Lambda_tmp_2 < 0).flatten()

                    # append zeros after a NS collapses, i.e for masses > M_tov
                    Lambda[ii] = np.append(Lambda_tmp_2[args],
                                           np.zeros(len(args_2)))

        else:
            # if mass is a float
            interpolated_Lambda = interp1d(
                mass_tmp, Lambda_tmp, fill_value='extrapolate')
            Lambda = interpolated_Lambda(mass)
            if Lambda < 0:
                Lambda = 0

    else:
        # if the maximum_mass is below the maximum mass supported by
        # pulsar observations, we set Lambda=np.nan. These values
        # should be ruled out by the prior
        Lambda = np.nan

    return Lambda, max_mass


def density_of_pressure(pressure, gamma_0, gamma_1, gamma_2, gamma_3):
    '''
    Parameters
    ----------
    pressure_array = numpy array
        pressure array in SI units
    gamma_0, gamma_1, gamma_2, gamma_3 = float
        polytrope hyper parameters in SI units

    Returns
    -------
    density in kg/m^3
    '''
    pol = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(
        gamma_0, gamma_1, gamma_2, gamma_3)
    enthalpy = np.array([
        lalsim.SimNeutronStarEOSPseudoEnthalpyOfPressure(pp, pol)
        for pp in pressure])
    density = np.array([
        lalsim.SimNeutronStarEOSRestMassDensityOfPseudoEnthalpy(ee, pol)
        for ee in enthalpy])

    return density


def radius_of_mass(mass, gamma_0, gamma_1, gamma_2, gamma_3):
    '''
    Parameters
    ----------
    mass = numpy array
            mass in solar masses
    gamma_0, gamma_1, gamma_2, gamma_3 = float
        polytrope hyperparameters in SI units

    Returns
    -------
    mass: numpy array
        mass in solar masses
    radius: numpy array
         radius in km
    '''
    m1 = mass*MSUN_SI
    polytrope = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(
            gamma_0, gamma_1, gamma_2, gamma_3)
    EoS_family = lalsim.CreateSimNeutronStarFamily(polytrope)

    m_tmp = list()
    r_tmp = list()
    for mm in m1:
        try:
            r_tmp.append(lalsim.SimNeutronStarRadius(mm, EoS_family))
            m_tmp.append(mm)
        except RuntimeError:
            pass
    return np.array(m_tmp)/MSUN_SI, np.array(r_tmp)/1000


def energy_density_of_pressure(pressure, gamma_0, gamma_1, gamma_2, gamma_3):
    '''
    Parameters
    ----------
    pressure_array = numpy array
        pressure array in SI units
    gamma_0, gamma_1, gamma_2, gamma_3 = float
        polytrope hyper parameters in SI units

    Returns
    -------
    energy density in J/m^3
    '''
    pol = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(
        gamma_0, gamma_1, gamma_2, gamma_3)

    energy_density = np.array([
        lalsim.SimNeutronStarEOSEnergyDensityOfPressure(pp, pol)
        for pp in pressure])

    return energy_density


def generate_bns_parameters_from_piecewise_polytrope_posteriors(
        posterior_samples, maximum_speed_of_sound=1.15,
        max_mass=1.90, check_eos_constraints=False,
        pressure=np.logspace(np.log10(4e32), np.log10(2.5e35), 70)):
    '''
    Parameters
    ----------
    posterior_samples = pandas dataframe
        posterior samples containing spectral decomposition posteriors
    maximum_speed_of_sound = float
        maximum speed of sound
    max_mass = float
        maximum mass allowed by the EoS
    check_eos_constraints = bool
        check if an EoS satifsifes maximum mass and speed of sound constraints
    Returns
    -------
    posterior_samples = pandas dataframe
        posteriors samples including all bns parameters, including lambda_1
        and lambda_2
    '''

    bns_posteriors = convert_to_lal_binary_neutron_star_parameters(
        posterior_samples)[0]

    logger.info("Generating lambda_1 samples")
    Lambda_1_list = list()
    for ii in tqdm(range(len(bns_posteriors['mass_1']))):
        Lambda_tmp = Lambda_of_mass(
            pressure, bns_posteriors['mass_1'][ii],
            bns_posteriors['gamma_0'][ii],
            bns_posteriors['gamma_1'][ii],
            bns_posteriors['gamma_2'][ii],
            bns_posteriors['gamma_3'][ii])
        Lambda_1_list.append(Lambda_tmp)
    Lambda_1 = np.array(Lambda_1_list)

    logger.info("Generating lambda_2 samples")
    Lambda_2_list = list()
    for ii in tqdm(range(len(bns_posteriors['mass_2']))):
        Lambda_tmp = Lambda_of_mass(
            pressure, bns_posteriors['mass_2'][ii],
            bns_posteriors['gamma_0'][ii],
            bns_posteriors['gamma_1'][ii],
            bns_posteriors['gamma_2'][ii],
            bns_posteriors['gamma_3'][ii])
        Lambda_2_list.append(Lambda_tmp)
    Lambda_2 = np.array(Lambda_2_list)

    bns_posteriors['lambda_1'] = Lambda_1
    bns_posteriors['lambda_2'] = Lambda_2

    if check_eos_constraints:
        logger.info('Checking causality and maximum mass constraints')
        max_mass_list = list()
        max_sound_list = list()
        for ii in tqdm(range(len(bns_posteriors['gamma_0']))):
            max_sound = maximum_speed_of_sound(
                bns_posteriors['gamma_0'][ii],
                bns_posteriors['gamma_1'][ii],
                bns_posteriors['gamma_2'][ii],
                bns_posteriors['gamma_3'][ii])
            if max_sound > maximum_speed_of_sound:
                max_sound_list.append(ii)
            else:
                max_mass_tmp = maximum_mass(
                    bns_posteriors['gamma_0'][ii],
                    bns_posteriors['gamma_1'][ii],
                    bns_posteriors['gamma_2'][ii],
                    bns_posteriors['gamma_3'][ii])
                if max_mass_tmp < max_mass:
                    max_mass_list.append(ii)

        delete_indices = np.append(max_mass_list, max_sound_list)
        final_posteriors = bns_posteriors.drop(
            delete_indices).reset_index(drop=True)
        logger.info(f'Number of samples removed: {len(delete_indices)}')
    else:
        final_posteriors = bns_posteriors

    return final_posteriors


def pressure_density_posterior(
        bilby_result, central_pressure=np.logspace(np.log10(3e31),
                                                   np.log10(90e34), 100)):
    '''
    Parameters
    ----------
    bilby_result: bilby result object

    Returns
    -------
    density: numpy array
        density in units of g/cm^3
    pressure: numpy array
        pressure in units of dyne/cm^2
    '''
    gamma_0 = bilby_result.posterior['gamma_0']
    gamma_1 = bilby_result.posterior['gamma_1']
    gamma_2 = bilby_result.posterior['gamma_2']
    gamma_3 = bilby_result.posterior['gamma_3']
    pressure_list = list()
    density_list = list()

    for i in tqdm(range(len(gamma_1))):
        density_tmp = density_of_pressure(central_pressure, gamma_0[i],
                                          gamma_1[i], gamma_2[i], gamma_3[i])
        for j in range(len(density_tmp)):
            pressure_list.append(central_pressure[j])
            density_list.append(density_tmp[j])

    return np.array(density_list)/1000, np.array(pressure_list)*10


def mass_radius_posterior(bilby_result, nsamples=200,
                          mass_array=np.linspace(0.5, 3, 100)):
    '''
    Parameters
    ----------
    bilby_result: bilby result object

    Returns
    -------
    mass: numpy array
        mass in units of solar masses
    radius: numpy array
        radius in units of km
    '''

    gamma_0 = bilby_result.posterior['gamma_0']
    gamma_1 = bilby_result.posterior['gamma_1']
    gamma_2 = bilby_result.posterior['gamma_2']
    gamma_3 = bilby_result.posterior['gamma_3']
    mass_list = list()
    radius_list = list()

    samples = np.random.randint(0, len(gamma_0), nsamples)

    # for i in tqdm(range(len(gamma_0))):
    for i in tqdm(samples):
        m_tmp, r_tmp = radius_of_mass(mass_array, gamma_0[i], gamma_1[i],
                                      gamma_2[i], gamma_3[i])
        mass_list.append(m_tmp)
        radius_list.append(r_tmp)

    new_mass_list = list()
    new_radius_list = list()

    for i in range(len(mass_list)):
        m_tmp = mass_list[i]
        r_tmp = radius_list[i]
        for j in range(len(m_tmp)):
            new_mass_list.append(m_tmp[j])
            new_radius_list.append(r_tmp[j])

    return np.array(new_mass_list), np.array(new_radius_list)


def mass_radius_confidence_intervals(bilby_result, quantiles=[0.05, 0.5, 0.95],
                                     mass_array=np.linspace(0.5, 3, 100),
                                     nsamples=200):
    '''
    Parameters
    ----------
    result_dataframe: pandas dataframe
        a dataframe containing radius and mass posteriors
    quantiles: list
        quantiles. Should be a list of lenght=3, by default returns
        the 90% confidence interval and the median
    mass_array: numpy array
        MUST be the same value used in
        toast.piecewise_polytrope.mass_radius_posterior

    Returns
    -------
    min_radius: numpy array
        min radius encompased by the confidence interval defined by quantiles
    max_radius: numpy array
        max radius encompased by the confidence interval defined by quantiles
    median_radius: numpy array
        median radius
    mass: numpy array

    '''

    # mass_posteriors = result_dataframe['mass'].values
    # radius_posteriors = result_dataframe['radius'].values
    mass_posteriors, radius_posteriors = mass_radius_posterior(
        bilby_result, nsamples, mass_array)

    # mass_tmp = np.linspace(0.5,3,100)
    mass_tmp = mass_array
    min_radius_list = list()
    max_radius_list = list()
    median_radius_list = list()
    mass_list = list()
    for mm in mass_tmp:
        try:
            args = np.argwhere(mass_posteriors == mm).flatten()
            r_min, r_median, r_max = np.quantile(radius_posteriors[args],
                                                 quantiles)
            min_radius_list.append(r_min)
            max_radius_list.append(r_max)
            median_radius_list.append(r_median)
            mass_list.append(mm)
        except IndexError:
            pass
    min_radius = np.array(min_radius_list)
    max_radius = np.array(max_radius_list)
    median_radius = np.array(median_radius_list)
    mass = np.array(mass_list)

    return min_radius, max_radius, median_radius, mass


def pressure_density_confidence_intervals(result_dataframe,
                                          quantiles=[0.05, 0.5, 0.95],
                                          central_pressure_array_length=100):
    '''
    Parameters
    ----------
    result_dataframe: pandas dataframe
        a dataframe containing pressure and density posteriors
    quantiles: list
        quantiles. Should be a list of lenght=3, by default returns
        the 90% confidence interval and the median
    central_pressure_array_length: int
        MUST be the same lenght of the central pressure array used in
        toast.piecewise_polytrope.pressure_density_posterior, defaults
        to 100

    Returns
    -------
    min_density: numpy array
        min density encompased by the confidence interval defined by quantiles
    max_density: numpy array
        max density encompased by the confidence interval defined by quantiles
    median_density: numpy array
        median density
    density: numpy array
        density array

    '''
    # after central_pressure_array_lenght, the values just repeat
    pressure = result_dataframe['pressure'].values[
        :central_pressure_array_length]
    min_density = list()
    max_density = list()
    median_density = list()

    for pp in pressure:
        args = np.argwhere(
            result_dataframe['pressure'].values == pp).flatten()
        density = result_dataframe['density'].values[args]
        minn, median, maxx = np.quantile(density, quantiles)
        min_density.append(minn)
        max_density.append(maxx)
        median_density.append(median)

    min_density = np.array(min_density)
    max_density = np.array(max_density)
    median_density = np.array(median_density)

    return min_density, max_density, median_density, pressure
