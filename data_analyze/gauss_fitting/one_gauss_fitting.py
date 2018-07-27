from scipy.optimize import leastsq
import numpy as np
import peakutils

def _gauss_func(par, t):
    return 1/np.sqrt(2*np.pi)/abs(par[0])*np.exp(-np.power(t-abs(par[1]), 2)/(2*np.power(par[0], 2)))

def _err_func(par, t, expect_output):
    err = expect_output - _gauss_func(par, t)
    return err

def _two_gauss_func(par, t):
    return (par[0]/2)*np.exp(-np.power(t-par[2], 2)/(2*np.power(par[1], 2))) + \
            (par[3]/2)*np.exp(-np.power(t-par[5], 2)/(2*np.power(par[4], 2)))

def _err_two_gauss_func(par, t, expect_output):
    err = expect_output - _two_gauss_func(par, t)
    return err

def _get_p0(density):
    xs = density['x']
    ys = density['y']
    indices = peakutils.indexes(-ys, thres=0.1)
    valleies = xs[indices]
    valleies = np.append(valleies, min(xs))
    valleies = np.append(valleies, max(xs))
    mu_est = xs[np.argmax(ys)]
    dis = abs(valleies-mu_est)
    s_est = (min(dis)/2)
    p0 = [s_est, mu_est]
    return p0

def _get_threshold(p_est):
    mu = p_est[1]
    s = abs(p_est[0])
    return max(mu+2*s, mu+50)

def fitting(density):
    p0 = _get_p0(density)
    p_Est = leastsq(_err_func, p0, args=(density['x'], density['y']))
    density['param0'] = p0
    density['y0_est'] = _gauss_func(p0, density['x'])
    density['param'] = p_Est[0]
    density['y_est'] = _gauss_func(p_Est[0], density['x'])
    density['threshold'] = _get_threshold(p_Est[0])
    return density

