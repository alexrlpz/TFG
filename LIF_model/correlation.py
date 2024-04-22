#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool

import scipy

def pairwise_correlation_worker(args):
    x, i, j = args
    c, p_value = scipy.stats.pearsonr(x[i], x[j])
    return c if not math.isnan(c) else None

def pairwise_correlation(x, bin=200, hist_range=[-1.0, 1.0], num_processes=4):
    cc = []
    args_list = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            args_list.append((x, i, j))

    with Pool(num_processes) as pool:
        results = pool.map(pairwise_correlation_worker, args_list)

    cc = [c for c in results if c is not None]
    hist, bin_edges = np.histogram(cc, bins=bin, range=hist_range)

    return np.mean(cc), hist, bin_edges

data_path = '/home/alejandro/Escritorio/TFG/TFG/LIF_model/LIF_simulations/'
folder = 'ae15b9e197447950b3671c54e580acb0'
path = data_path+folder

TRANSIENT = pickle.load(open(path+'/TRANSIENT','rb'))
tstop = pickle.load(open(path+'/tstop','rb'))

gids_E = pickle.load(open(path+'/gids_'+'E','rb')) 
times_E = pickle.load(open(path+'/times_'+'E','rb'))

gids_E = gids_E[times_E >= TRANSIENT]
times_E = times_E[times_E >= TRANSIENT]

NUM_NEURONS = 500

random_neurons = [random.randint(1, 8192) for _ in range(NUM_NEURONS)]
BIN_TIME = 80
NUM_BINS = int((tstop-TRANSIENT)/BIN_TIME)

total_firing_rates = np.zeros((NUM_NEURONS, NUM_BINS))
bin_edges = np.arange(TRANSIENT, tstop+BIN_TIME, BIN_TIME)

# Calculate random neurons firing rates

for i, neuron_id in enumerate(random_neurons):
    neuron_spikes_times = times_E[gids_E == neuron_id]                              # Times of the spikes produced by this neuron
    total_firing_rates[i], _ = np.histogram(neuron_spikes_times, bins=bin_edges)

total_firing_rates = total_firing_rates / BIN_TIME
    
if __name__ == "__main__":
    mean_corr, hist, bin_edges = pairwise_correlation(total_firing_rates, bin=200, hist_range=[-1.0, 1.0], num_processes=12)
    
plt.figure(figsize=(8, 6))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', edgecolor='black')
plt.xlabel('Correlation')
plt.ylabel('Frequency')
plt.title('Firing rates correlation histogram')
plt.axvline(mean_corr, color='r', linestyle='dashed', linewidth=1, label=f'Media: {mean_corr:.2f}')
plt.legend()
plt.grid(True)
plt.show()
