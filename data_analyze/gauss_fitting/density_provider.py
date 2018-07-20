import numpy as np
from scipy import stats
import peakutils

def get_density(origin):
    #sorted_data = origin.sort_values(ascending=True)
    #xmin = origin[4]
    #xmax = origin[120-5]
    xmin = origin.min()
    xmax = origin.max()
    if xmax - xmin < 50:
        return xmin
    if int(xmin) == 0:
        xmin -= 100
    xs = np.arange(int(xmin*0.5), int(xmax*1.2), 1)
    density_func = stats.gaussian_kde(origin)
    ys = density_func(xs)
    density = {"x":xs, "y":ys}
    return density

def is_one_more_peak(density):
    return len(peakutils.indexes(density['y'], min_dist=3)) > 1
