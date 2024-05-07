#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
import os
import sys
import math
import pickle
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import LIF_network
import random
import scipy
from multiprocessing import Pool

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

def zscore(data, ch, time):
    '''
    Compute the z score using the the maximum value of all channels instead of
    the standard deviation of the sample.
    '''
    tr_data = np.array(data)[:,time]
    tr_data -= np.mean(tr_data,axis = 1).reshape(-1,1)
    return np.max(np.abs(tr_data)),tr_data[ch] /np.max(np.abs(tr_data))


if len(sys.argv) < 2:
    print("ERROR. Usage: python3 plots.py <simulation_folder_hash>")
    sys.exit(1)
    
hash = sys.argv[1]  

# Create a LIF_network object
LIF_net = LIF_network.LIF_network()

# Load data
# path to simulation data
data_path = '/home/alejandro/Escritorio/TFG/TFG/LIF_model/LIF_simulations/'
folder = hash
path = data_path+folder

LIF_net.LIF_params = pickle.load(open(path+'/LIF_params','rb'))
print("\nSIMULATION PARAMS: \n")
print(LIF_net.LIF_params)
LIF_net.TRANSIENT = pickle.load(open(path+'/TRANSIENT','rb'))
LIF_net.dt = pickle.load(open(path+'/dt','rb'))
LIF_net.tstop = pickle.load(open(path+'/tstop','rb'))
LIF_net.tau = pickle.load(open(path+'/tau','rb'))

LIF_net.H_YX = pickle.load(open(data_path+'/H_YX','rb'))
LFP_data = pickle.load(open(path+'/LFP_data','rb'))
CDM_data = pickle.load(open(path+'/CDM_data','rb'))
lif_mean_nu_X = pickle.load(open(path+'/lif_mean_nu_X','rb'))  # mean spike rates
[bins, lif_nu_X] = pickle.load(open(path+'/lif_nu_X','rb'))

# Path to save plots generated
graphics_folder = (
    f'EE_{LIF_net.LIF_params["J_YX"][0][0]:.2f}_'
    f'IE_{LIF_net.LIF_params["J_YX"][0][1]:.2f}_'
    f'EI_{LIF_net.LIF_params["J_YX"][1][0]:.2f}_'
    f'II_{LIF_net.LIF_params["J_YX"][1][1]:.2f}'
)
# Crear la ruta completa al directorio de gráficos
graphics_path = f'simulations_graphics/{graphics_folder}'

if os.path.isdir(graphics_path):
    print("\nGraphics already created\n")
    print(graphics_folder)
    sys.exit(1)
else:    
    os.makedirs(graphics_path, exist_ok=True)


# Plot spikes and firing rates
fig = plt.figure(figsize=[6,5], dpi=150)
T = [4000, 4100]

# Spikes
ax1 = fig.add_axes([0.15,0.45,0.75,0.5])
ax1.set_title('Network spike raster')
for i, Y in enumerate(LIF_net.LIF_params['X']):       # X = ['E', 'I'] -> hace dos iteraciones, 0 E , 1 I
    times = pickle.load(open(path+'/times_'+Y,'rb'))  # Pilla times_E y times_I
    gids = pickle.load(open(path+'/gids_'+Y,'rb'))    # gids_E y gids_I
    gids = gids[times >= LIF_net.TRANSIENT]           # TRANSIENT = 2000, filtra los elementos de gids que están asociados con los tiempos que son mayores o iguales a TRANSIENT.
    times = times[times >= LIF_net.TRANSIENT]

    ii = (times >= T[0]) & (times <= T[1])             # crear un array booleano ii que indica si cada elemento de la variable times está dentro del rango definido por T.
    ax1.plot(times[ii], gids[ii], '.',
            mfc='C{}'.format(i),                                            # formato colores etc
            mec='w',                                                         # color w white
            label=r'$\langle \nu_\mathrm{%s} \rangle =%.2f$ s$^{-1}$' % (
                Y, lif_mean_nu_X[Y] / LIF_net.LIF_params['N_X'][i])         # lif_mean_nu, mean spike rates ||| N_X=[8192, 1024] 
           )                                                                # divide la tasa media de disparo de la población entre el nº de neuronas de la población 
ax1.legend(loc=1)                                                           # añadir leyenda, en Matplotlib, loc=1 suele referirse a la esquina superior derecha del gráfico.
ax1.axis('tight')                                                           # formato
ax1.set_xticklabels([])                                                     # eliminar etiquetas del eje x
ax1.set_ylabel('gid', labelpad=0)                                           # añade label al eje y, "gid", que debe ser group id

# Rates
ax2 = fig.add_axes([0.15,0.08,0.75,0.3])
ax2.set_title('Per-population spike-count histograms')
Delta_t = LIF_net.dt
binsr = np.linspace(T[0], T[1], int(np.diff(T) / Delta_t + 1))

for i, Y in enumerate(LIF_net.LIF_params['X']):
    times = pickle.load(open(path+'/times_'+Y,'rb'))
    ii = (times >= T[0]) & (times <= T[1])
    ax2.hist(times[ii], bins=binsr, histtype='step', label = r'$V_{\mathrm{%s}}$' % Y)

ax2.axis('tight')
ax2.legend(loc=1, handlelength=1)
ax2.set_xlabel('t (ms)', labelpad=0)
ax2.set_ylabel(r'$\nu_X$ (spikes/$\Delta t$)', labelpad=0)

# Save figure
plt.savefig(graphics_path+'/spikes_raster.png')

# Plot kernels and LFP/CDM data                         # kernel -> Spike-signal impulse response function 
# Create figure and panels
fig1 = plt.figure(figsize=[7,6], dpi=150)
fig1.suptitle("Kernels for extracellular potentials and CDM")
fig2 = plt.figure(figsize=[7,6], dpi=150)
fig2.suptitle("Extracellular potentials and component of the CDM (z-axis)")
ax1 = []
ax2 = []

# First row
for k in range(4):
    ax1.append(fig1.add_axes([0.1 + k*0.22,0.4,0.18,0.5]))
    ax2.append(fig2.add_axes([0.1 + k*0.22,0.4,0.18,0.5]))
# Second row
for k in range(4):
    ax1.append(fig1.add_axes([0.1 + k*0.22,0.1,0.18,0.2]))
    ax2.append(fig2.add_axes([0.1 + k*0.22,0.1,0.18,0.2]))

# Time arrays
LIF_net.dt*= 10 # take into account the decimate ratio
bins = bins[::10] # take into account the decimate ratio
time = np.arange(-LIF_net.tau,LIF_net.tau+LIF_net.dt,LIF_net.dt)
T = [4000,4100]
ii = (bins >= T[0]) & (bins <= T[1])
iii = np.where(bins >= T[0] + np.diff(T)[0]/2)[0][0]

# LFP probe
probe = 'GaussCylinderPotential'
k = 0
for X in LIF_net.LIF_params['X']:
    for Y in LIF_net.LIF_params['X']:
        n_ch = LIF_net.H_YX[f'{X}:{Y}'][probe].shape[0]
        for ch in range(n_ch):
            # decimate first
            dec_kernel = ss.decimate(LIF_net.H_YX[f'{X}:{Y}'][probe],
                                     q=10,zero_phase=True)
            # z-scored kernel from 0 to 1/2 of tau
            maxk, norm_ker = zscore(dec_kernel,ch,
                                    np.arange(int(time.shape[0]/2),
                                              int(3*time.shape[0]/4)))
            # z-scored LFP signal from T[0] to T[1]
            maxs,norm_sig = zscore(LFP_data[f'{X}{Y}'],ch,ii[:-1])
            # plot data stacked in the Z-axis
            ax1[k].plot(time[int(time.shape[0]/2):int(3*time.shape[0]/4)],
                        norm_ker - ch)
            ax2[k].plot(bins[ii],norm_sig - ch)
        ax1[k].set_title(f'H_{X}:{Y}')
        ax2[k].set_title(f'H_{X}:{Y}')
        if k == 0:
            ax1[k].set_yticks(np.arange(0,-n_ch,-1))
            ax1[k].set_yticklabels(['ch. '+str(ch) for ch in np.arange(1,n_ch+1)])
            ax2[k].set_yticks(np.arange(0,-n_ch,-1))
            ax2[k].set_yticklabels(['ch. '+str(ch) for ch in np.arange(1,n_ch+1)])
        else:
            ax1[k].set_yticks([])
            ax1[k].set_yticklabels([])
            ax2[k].set_yticks([])
            ax2[k].set_yticklabels([])
        ax1[k].set_xlabel(r'$tau_{ms}$')
        ax2[k].set_xlabel('t (ms)')

        # scales
        ax1[k].plot([time[int(0.59*time.shape[0])],time[int(0.59*time.shape[0])]],
                     [0,-1],linewidth = 2., color = 'k')
        sexp = np.round(np.log2(maxk))
        ax1[k].text(time[int(0.6*time.shape[0])],-0.5,r'$2^{%s}mV$' % sexp)
        ax2[k].plot([bins[iii],bins[iii]],
                     [0,-1],linewidth = 2., color = 'k')
        sexp = np.round(np.log2(maxs))
        ax2[k].text(bins[iii],-0.5,r'$2^{%s}mV$' % sexp)
        k+=1

# Current dipole moment
probe = 'KernelApproxCurrentDipoleMoment'
k = 0
for X in LIF_net.LIF_params['X']:
    for Y in LIF_net.LIF_params['X']:
        # Pick only the z-component of the CDM kernel.
        # (* 1E-4 : nAum --> nAcm unit conversion)
        dec_kernel = ss.decimate([1E-4*LIF_net.H_YX[f'{X}:{Y}'][probe][2]],
                                 q=10,zero_phase=True)
        maxk, norm_ker = zscore(dec_kernel,0,
                                 np.arange(int(time.shape[0]/2),
                                           int(3*time.shape[0]/4)))
        # z-scored CDM signal
        maxs,norm_sig = zscore([1E-4*CDM_data[f'{X}{Y}']],0,ii[:-1])
        # plot data
        ax1[k+4].plot(time[int(time.shape[0]/2):int(3*time.shape[0]/4)],norm_ker)
        ax2[k+4].plot(bins[ii],norm_sig)

        if k == 0:
            ax1[k+4].set_yticks([0])
            ax1[k+4].set_yticklabels([r'$P_z$'])
            ax2[k+4].set_yticks([0])
            ax2[k+4].set_yticklabels([r'$P_z$'])
        else:
            ax1[k+4].set_yticks([])
            ax1[k+4].set_yticklabels([])
            ax2[k+4].set_yticks([])
            ax2[k+4].set_yticklabels([])
        ax1[k+4].set_xlabel(r'$tau_{ms}$')
        ax2[k+4].set_xlabel('t (ms)')

        # scales
        ax1[k+4].plot([time[int(0.59*time.shape[0])],time[int(0.59*time.shape[0])]],
                     [0,-1],linewidth = 2., color = 'k')
        sexp = np.round(np.log2(maxk))
        ax1[k+4].text(time[int(0.6*time.shape[0])],-0.5,r'$2^{%s}nAcm$' % sexp)
        ax2[k+4].plot([bins[iii+5],bins[iii+5]],
                     [0,-1],linewidth = 2., color = 'k')
        sexp = np.round(np.log2(maxs))
        ax2[k+4].text(bins[iii],-0.5,r'$2^{%s}nAcm$' % sexp)
        k+=1

# Save figures
fig1.savefig(graphics_path+'/kernels.png')
fig2.savefig(graphics_path+'/extra_pt_CDM.png')

# Power spectrum
fig = plt.figure(figsize=[6,5], dpi=150)
T = [4000, 4100]

CDM_data_total = CDM_data['EE'] + CDM_data['IE'] + CDM_data['EI'] + CDM_data['II']
maxs,norm_sig = zscore([1E-4*CDM_data_total],0,ii[:-1])

ax1 = fig.add_axes([0.15,0.15,0.65,0.65])
ax1.set_title('CDM power spectrum')


frequencies, power_spectrum = ss.welch(norm_sig, fs=16.0)
plt.semilogy(frequencies, power_spectrum)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.text(1.05, 0.85, 'Parámetros: \n J_EE: ' + str("{:.2f}".format(LIF_net.LIF_params['J_YX'][0][0])) + 
                           '\n J_IE:  ' + str("{:.2f}".format(LIF_net.LIF_params['J_YX'][0][1])) + 
                           '\n J_EI: ' + str("{:.2f}".format(LIF_net.LIF_params['J_YX'][1][0])) + 
                           '\n J_II:  ' + str("{:.2f}".format(LIF_net.LIF_params['J_YX'][1][1]))                   
                            ,transform=ax1.transAxes, verticalalignment='bottom', fontsize=8) 

# Save figure
plt.savefig(graphics_path+'/power_spectrum.png')

# Firing rates correlation between pairs of neurons

gids_E = pickle.load(open(path+'/gids_'+'E','rb')) 
times_E = pickle.load(open(path+'/times_'+'E','rb'))

gids_E = gids_E[times_E >= LIF_net.TRANSIENT]
times_E = times_E[times_E >= LIF_net.TRANSIENT]
NUM_NEURONS = 500
BIN_TIME = 80
NUM_BINS = int((LIF_net.tstop-LIF_net.TRANSIENT)/BIN_TIME)

random_neurons = [random.randint(1, 8192) for _ in range(NUM_NEURONS)]

total_firing_rates = np.zeros((NUM_NEURONS, NUM_BINS))
bin_edges = np.arange(LIF_net.TRANSIENT, LIF_net.tstop+BIN_TIME, BIN_TIME)

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

# Save figure
plt.savefig(graphics_path+'/neurons_correlation.png')

