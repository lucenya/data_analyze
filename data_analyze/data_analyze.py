import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal, fftpack 
from gauss_fitting import density_provider
from gauss_fitting import one_gauss_fitting
from gauss_fitting import two_gauss_fitting

DATA_DIR = ".\\data\\"
#DETECTED_CAT = ["duration_P75", "duration_P95", "duration_P75_fft", "duration_P95_fft"]
DETECTED_CAT = ["duration_P75", "duration_P95", "duration_P75_butter", "duration_P95_butter"]
CHART_COLUMNS = 2
CHART_ROWS = 5

def _save_chart(density, origin, anomaly, file_name):
    fig = plt.figure(figsize=(20, 7*CHART_ROWS), dpi=80)
    i = 0
    for category in DETECTED_CAT:
        threshold = density[category]['threshold']    
        plt.subplot(CHART_ROWS, CHART_COLUMNS, i*CHART_COLUMNS+1)
        plt.plot(density[category]['x'], density[category]['y'], 'b')
        plt.plot(density[category]['x'], density[category]['y_est'], 'y')
        plt.plot([threshold, threshold], [0, max(density[category]['y'])], 'r')
        plt.title(density[category]['param'])
        plt.subplot(CHART_ROWS, CHART_COLUMNS, i*CHART_COLUMNS+2)
        b,a = signal.butter(3, 0.15, 'low')
        sf = signal.filtfilt(b, a, origin[category])
        plt.plot(origin.startDayHour, origin[category], 'b')
        plt.plot(anomaly[category]['startDayHour'], anomaly[category][category], 'ro')
        plt.plot(origin.startDayHour, sf, 'y')
        plt.title(category + str.replace(file_name, ".png", ""))
        i += 1
    plt.subplot(CHART_ROWS, CHART_COLUMNS, CHART_COLUMNS*CHART_ROWS)
    plt.plot(origin.startDayHour, origin.numSamples)
    fig.savefig(file_name)
    plt.close(fig)

def _check_anomaly(origin, density, file_name):
    anomaly = {}
    for category in DETECTED_CAT:
        anomaly[category] = origin[origin[category] > density[category]['threshold']]
    _save_chart(density, origin, anomaly, file_name)
    
def _start_detect(origin, file_name):
    densities = {} 
    for category in DETECTED_CAT:
        perf = origin[category]
        #b,a = signal.butter(3, 0.15, 'low')
        #sf = signal.filtfilt(b, a, origin[category])
        density = density_provider.get_density(perf)
        if not isinstance(density, dict):
            print('The perf data is stable at {}'.format(density))
            return
        if density_provider.is_one_more_peak(density):
            density = two_gauss_fitting.fitting(density)
        else:
            density = one_gauss_fitting.fitting(density)
        densities[category] = density    
    _check_anomaly(origin, densities, file_name)


file_list = (os.listdir(DATA_DIR))

for file in file_list:
    if os.path.isdir(DATA_DIR+file):
        continue    
    data = pd.read_csv(DATA_DIR+file) 
    data = data.drop([119])
    print(file)
    data['startDayHour'] = pd.to_datetime(data.startDayHour)
    b,a = signal.butter(3, 0.15, 'low')
    data['duration_P75_butter'] = signal.filtfilt(b, a, data['duration_P75'])
    data['duration_P95_butter'] = signal.filtfilt(b, a, data['duration_P95'])
    #data['duration_P75_fft'] = fftpack.rfft(data['duration_P75'])
    #data['duration_P75_fft'] = np.sqrt(np.power(data['duration_P75_fft'].imag,2)+np.power(data['duration_P75_fft'].real,2))
    #data['duration_P95_fft'] = fftpack.rfft(data['duration_P95'])
    #data['duration_P95_fft'] = np.sqrt(np.power(data['duration_P95_fft'].imag,2)+np.power(data['duration_P95_fft'].real,2))
    #data['duration_P75_fft'] = fftpack.dst(data['duration_P75'])
    #data['duration_P95_fft'] = fftpack.dst(data['duration_P95'])
    _start_detect(data, str.replace(file, ".csv", ".png"))
    