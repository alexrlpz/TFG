#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

import scipy

# Mean value and histogram of the correlation coefficient between pairs of neurons
   
def pairwiseCorrelation(x,bin = 100,hist_range = [-1.0,1.0]):
    cc = []
    for i in range(len(x)):
        for j in np.arange(i+1,len(x)):
                c,p_value = scipy.stats.pearsonr(x[i], x[j])
                if math.isnan(c) == False:
                    cc.append(c)

    hist, bin_edges = np.histogram(cc,bins=bin,range=hist_range)
    return np.mean(cc),hist,bin_edges

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
BIN_TIME = 100
NUM_BINS = int((tstop-TRANSIENT)/BIN_TIME)

total_firing_rates = np.zeros((NUM_NEURONS, NUM_BINS))
bin_edges = np.arange(TRANSIENT, tstop+BIN_TIME, BIN_TIME)

for i, neuron_id in enumerate(random_neurons):
    neuron_spikes_times = times_E[gids_E == neuron_id]                              # Times of the spikes produced by this neuron
    total_firing_rates[i], _ = np.histogram(neuron_spikes_times, bins=bin_edges)

total_firing_rates = total_firing_rates / BIN_TIME
    
mean_corr, hist, bin_edges = pairwiseCorrelation(total_firing_rates)

plt.figure(figsize=(8, 6))
plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', edgecolor='black')
plt.xlabel('Correlación')
plt.ylabel('Frecuencia')
plt.title('Histograma de Correlaciones entre Tasas de Disparo')
plt.axvline(mean_corr, color='r', linestyle='dashed', linewidth=1, label=f'Media: {mean_corr:.2f}')
plt.legend()
plt.grid(True)
plt.show()
