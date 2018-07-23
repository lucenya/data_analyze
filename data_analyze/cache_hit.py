import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal, fftpack 
from gauss_fitting import density_provider
from gauss_fitting import one_gauss_fitting
from gauss_fitting import two_gauss_fitting

DATA_DIR = ".\\cache_hit\\"
#DETECTED_CAT = ["duration_P75", "duration_P95", "duration_P75_fft", "duration_P95_fft"]
DETECTED_CAT = ["ratio", "sampleNum", "ratio_butter", "sampleNum_butter"]
CATEGORY = ["ratio", "sampleNum"]
CHART_COLUMNS = 2
CHART_ROWS = 2

def _save_chart(density, origin, anomaly, file_name):
    fig = plt.figure(figsize=(20, 7*CHART_ROWS), dpi=80)
    i = 0
    for category in CATEGORY:
        threshold = density[category]['threshold']
        plt.subplot(CHART_ROWS, CHART_COLUMNS, i*CHART_COLUMNS+1)
        plt.plot(density[category]['x'], density[category]['y'], 'b')
        plt.plot(density[category]['x'], density[category]['y_est'], 'y')
        plt.plot([threshold, threshold], [0, max(density[category]['y'])], 'r')
        plt.title(density[category]['param'])
        plt.subplot(CHART_ROWS, CHART_COLUMNS, i*CHART_COLUMNS+2)
        plt.plot(origin.startDayHour, origin[category], 'b')
        plt.plot(anomaly[category]['startDayHour'], anomaly[category][category], 'ro')
        plt.plot(origin.startDayHour, origin[category+"_butter"], 'y')
        plt.title(category + str.replace(file_name, ".png", ""))
        i += 1
    fig.savefig(file_name)
    plt.close(fig)

def _check_anomaly(origin, density, file_name):
    anomaly = {}
    for category in CATEGORY:
        anomaly[category] = origin[origin[category] > density[category]['threshold']]
    _save_chart(density, origin, anomaly, file_name)
    
def _start_detect(origin, file_name):
    densities = {} 
    for category in DETECTED_CAT:
        perf = origin[category]
        density = density_provider.get_density(perf)
        densities[category] = density

    for category in CATEGORY:
        butter_cat = category+"_butter"
        if not isinstance(densities[category], dict):
            print('The perf data is stable at {}'.format(densities[category]))
            return
        if density_provider.is_one_more_peak(densities[category]):
            isHighFreq = True
            if (isinstance(densities[butter_cat], dict) and density_provider.is_one_more_peak(densities[butter_cat])):
                isHighFreq = False
            density = two_gauss_fitting.fitting(densities[category], isHighFreq)
        else:
            density = one_gauss_fitting.fitting(densities[category])
        
    _check_anomaly(origin, densities, file_name)


file_list = (os.listdir(DATA_DIR))

for file in file_list:
    if os.path.isdir(DATA_DIR+file):
        continue    
    data = pd.read_csv(DATA_DIR+file) 
    data = data.drop([119])
    print(file)
    data['startDayHour'] = pd.to_datetime(data.DateKey, format="%Y%m%d").apply(lambda x: x.date())
    data['ratio'] = data.apply((lambda x: 1 if x.MissCount+x.HitCount == 0 else x.MissCount/(x.MissCount+x.HitCount)), axis=1)*1000
    data['sampleNum'] = data.MissCount+data.HitCount
    b,a = signal.butter(3, 0.15, 'low')
    data['ratio_butter'] = signal.filtfilt(b, a, data['ratio'])
    data['sampleNum_butter'] = signal.filtfilt(b, a, data['sampleNum'])
    _start_detect(data, str.replace(file, ".csv", ".png"))
    