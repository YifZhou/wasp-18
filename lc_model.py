import numpy as np
from ackBar2 import ackBar2
from lmfit import Parameters, Model
import pandas as pd
import matplotlib.pyplot as plt


def lcModel(crate1, slope1, crate2, slope2, amp, phase,
            dTrap_f, dTrap_s, dTrap_f1, dTrap_s1, trap_pop_s, trap_pop_f,
            tExp, expTime, scanDirect):
    """wfc3 transit observation model, free parameters are transit mid
    time, transit period, transit depth (in term of planet radii),
    limb darkending.

    for two directional scanning

    And ackbar model parameters: number of traps, trapping coeeficient
    and trap life time

    transit light curves are firstly produced by batman transit model
    as intrinsic light curve of the transit. Then the light curve is
    fed into ackbar model to simulate the ramp effect
    """
    upIndex, = np.where(scanDirect == 0)
    downIndex, = np.where(scanDirect == 1)
    dTrap_f_list = np.ones(18) * dTrap_f
    dTrap_f_list[12] = dTrap_f1
    dTrap_s_list = np.ones(18) * dTrap_s
    dTrap_s_list[12] = dTrap_s1
    T = 81342.402336  # seconds, the orbital period of wasp18
    # a sinusoidal curve -- the phase
    cRates = amp * np.cos(tExp / T * 2 * np.pi + phase)
    # two slopes for two scanning direction
    cRates[upIndex] = (cRates / expTime + crate1 * (
        1 + tExp * slope1/1e7) / expTime)[upIndex]
    cRates[downIndex] = (cRates / expTime + crate2 * (
        1 + tExp * slope2/1e7) / expTime)[downIndex]
    obsCounts = ackBar2(
        cRates,
        tExp,
        expTime,
        trap_pop_s,
        trap_pop_f,
        dTrap_f=dTrap_f_list,
        dTrap_s=dTrap_s_list,
        lost=0,
        mode='scanning')
    return obsCounts


def fitLCModel(params, tExp, f0, scanDirect, weights):
    lc = Model(lcModel, independent_vars=['tExp', 'expTime', 'scanDirect'])
    expTime = 111.756798
    fitResult = lc.fit(f0,
                       tExp=tExp,
                       expTime=expTime,
                       scanDirect=scanDirect,
                       weights=weights,
                       params=params,
                       method='powell')
    return fitResult


def calRamp(p, tExp, scanDirect):
    """calcualte ramp profiles"""

    dTrap_f_list = np.ones(18) * p['dTrap_f']
    dTrap_f_list[12] = p['dTrap_f1']
    dTrap_s_list = np.ones(18) * p['dTrap_s']
    dTrap_s_list[12] = p['dTrap_s1']
    expTime = 111.756798
    upIndex, = np.where(scanDirect == 0)
    downIndex, = np.where(scanDirect == 1)
    cRates = np.ones_like(tExp)
    cRates[upIndex] = (p['crate1'] * (
        1 + tExp * p['slope1']/1e7) / expTime)[upIndex]
    cRates[downIndex] = (p['crate2'] * (
        1 + tExp * p['slope2']/1e7) / expTime)[downIndex]
    ramp = ackBar2(
        cRates,
        tExp,
        expTime,
        p['trap_pop_s'],
        p['trap_pop_f'],
        dTrap_f=dTrap_f_list,
        dTrap_s=dTrap_s_list,
        lost=0,
        mode='scanning')
    return ramp


if __name__ == '__main__':
    p = Parameters()
    count = 24850.829223225974
    p.add('crate1', value=count, vary=True)
    p.add('crate2', value=count, vary=True)
    p.add('slope1', value=0, min=-1, max=1, vary=True)
    p.add('slope2', value=0, min=-1, max=1, vary=True)
    p.add('amp', value=0.001 * count, vary=True)
    p.add('phase', value=-1.15, min=-np.pi, max=np.pi, vary=True)
    p.add('trap_pop_s', value=0, min=0, max=300, vary=True)
    p.add('trap_pop_f', value=0, min=0, max=100, vary=True)
    p.add('dTrap_f', value=0, min=0, max=150, vary=True)
    p.add('dTrap_s', value=0, min=0, max=120, vary=True)
    # for additional 3 points
    p.add('dTrap_f1', value=100, min=0, max=150, vary=True)
    p.add('dTrap_s1', value=100, min=0, max=120, vary=True)
    df = pd.read_csv('w18.csv')
    t = df['time'].values
    orbit = df['orbit'].values
    scanD = df['scanD'].values
    f0 = df['flux0'].values
    fitResult = fitLCModel(p, t, f0, scanD,
                           df['weights'].values)

    plt.figure()
    plt.plot(t[scanD == 0], f0[scanD == 0], 'o', mfc='none', mec='C0')
    plt.plot(t[scanD == 0], fitResult.best_fit[scanD == 0], color='C0')

    plt.plot(t[scanD == 1], f0[scanD == 1], 'o', mfc='none', mec='C1')
    plt.plot(t[scanD == 1], fitResult.best_fit[scanD == 1], color='C1')
    ramp = calRamp(fitResult.best_values, t, scanD)
    plt.figure()
    plt.plot(t, f0 / ramp, '.')
    plt.show()
