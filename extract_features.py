#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to extract and save features of the CDM data produced by simulations

import os
import pickle
import numpy as np
import pycatch22
import multiprocessing

# path to simulation data
data_path = '/home/alejandro/Escritorio/TFG/TFG/LIF_model/LIF_simulations'

# parameters of the LIF_network object
theta_data = []
# Current Dipole Moment (CDM) data
CDM_data = []

# Load simulation data
ldir = os.listdir(data_path)
for i,folder in enumerate(ldir):
    print(f'Loading file {i} out of {len(ldir)}')
    # load CDMs
    try:
        cdm = pickle.load(open(os.path.join(data_path,folder,"CDM_data"),'rb'))
        current_data = cdm[f'{"E"}{"E"}']+cdm[f'{"E"}{"I"}']+cdm[f'{"I"}{"E"}']+cdm[f'{"I"}{"I"}']
        CDM_data.append(current_data)
    except (FileNotFoundError, IOError):
        print(f'File CDM_data not found in {folder}')

    # load synapse parameters of recurrent connections and external input
    try:
        LIF_params = pickle.load(open(os.path.join(data_path,folder,"LIF_params"),'rb'))
        theta_data.append([LIF_params['J_YX'][0][0],
                                   LIF_params['J_YX'][0][1],
                                   LIF_params['J_YX'][1][0],
                                   LIF_params['J_YX'][1][1]
                                  ])
    except (FileNotFoundError, IOError):
        print(f'File LIF_params not found in {folder}')
        
# transform to np arrays
theta_data = np.array(theta_data)
CDM_data = np.array(CDM_data)

# remove the first 500 samples (transient response from convolution)
CDM_data = CDM_data[:,500:]                                                 # se queda un numpy array de shape (1002, 15500)

# Features
CDM_data_reshaped = []

for i in range(0,CDM_data.shape[0]):                                        # de 0 a 1002
    CDM_data_reshaped.append(CDM_data[i,:])                                 # añade cada numpy array con los datos de cada simulación (15500 valores)


def extract_features(data):
    return pycatch22.catch22_all(data)['values']

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    features = np.array(pool.map(extract_features, CDM_data_reshaped))
    pool.close()
    pool.join()
    
save_data_path = '/home/alejandro/Escritorio/TFG/TFG/simulations_data/'

np.save(save_data_path + 'features.npy', features)
np.save(save_data_path + 'theta_data.npy', theta_data)

print("\nFeatures and parameter values successfully extracted and saved\n")