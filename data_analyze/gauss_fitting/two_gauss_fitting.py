from scipy.optimize import leastsq
import numpy as np
import peakutils

def _two_gauss_func(par, t):
    return abs(par[4])*1/np.sqrt(2*np.pi)/abs(par[0])*np.exp(-np.power(t-abs(par[1]), 2)/(2*np.power(par[0], 2))) + \
        abs(1-par[4])*1/np.sqrt(2*np.pi)/abs(par[2])*np.exp(-np.power(t-abs(par[3]), 2)/(2*np.power(par[2], 2)))

def _err_two_gauss_func(par, t, expect_output):
    err = expect_output - _two_gauss_func(par, t)
    return err

def _get_first_two_peak_index(density):
    peak_indices = peakutils.indexes(density['y'], min_dist=3)
    peak_indices.sort()
    return peak_indices[:2]

def _get_sigma(density, two_peak_index):
    xs = density['x']
    ys = density['y']
    s_est = []
    valley_indices = peakutils.indexes(-ys, min_dist=3)
    if valley_indices.size == 0:
        valley_indices = np.append(valley_indices, int(np.average(two_peak_index)))
    valleies = xs[valley_indices]
    valleies = np.append(valleies, min(xs))
    valleies = np.append(valleies, max(xs))
    mu_est = xs[two_peak_index]
    is_second_peak_higher = ys[two_peak_index[0]] < ys[two_peak_index[1]]
    if is_second_peak_higher:
        mu_est = xs[[two_peak_index[1], two_peak_index[0]]]
    for mu in mu_est:
        dis = abs(valleies-mu)
        s_est.append(min(dis)/2)
        valleies = np.delete(valleies, np.argmin(dis))
    if is_second_peak_higher:
        return [s_est[1], s_est[0]]
    return s_est

def _get_two_gauss_p0(density, last_two_gauss_p):
    xs = density['x']
    ys = density['y']
    two_peak_index = _get_first_two_peak_index(density)
    mu_est = xs[two_peak_index]
    s_est = _get_sigma(density, two_peak_index)
    p0 = [s_est[0], mu_est[0], s_est[1], mu_est[1], 0.5]
    return p0

def _get_threshold(p_est, isHighFreq):
    if isHighFreq:
        return max(p_est[3]+2*p_est[2], min(p_est[1], p_est[3])+50)
    return max(p_est[1]+2*p_est[0], min(p_est[1], p_est[3])+50)

def fitting(density, isHighFreq):
    p0 = _get_two_gauss_p0(density, [])
    p_est = leastsq(_err_two_gauss_func, p0, args=(density['x'], density['y']))
    density['param0'] = p0
    density['y0_est'] = _two_gauss_func(p0, density['x'])
    density['param'] = p_est[0]
    density['y_est'] = _two_gauss_func(p_est[0], density['x'])
    density['threshold'] = _get_threshold(p_est[0], isHighFreq)
    return density
