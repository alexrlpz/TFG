#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
To do.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pycatch22

# Torch and SBI libraries
import torch
from sbi import analysis as analysis

from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from sklearn.model_selection import KFold

# path to simulation data
data_path = '/home/alejandro/Escritorio/TFG/TFG/LIF_model/LIF_simulations'

# parameters of the LIF_network object
theta_data = {'parameters':['J_EE',
                            'J_IE',
                            'J_EI',
                            'J_II'],
              'data': []}
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
        theta_data['data'].append([LIF_params['J_YX'][0][0],
                                   LIF_params['J_YX'][0][1],
                                   LIF_params['J_YX'][1][0],
                                   LIF_params['J_YX'][1][1]
                                  ])
    except (FileNotFoundError, IOError):
        print(f'File LIF_params not found in {folder}')

# transform to np arrays
theta_data['data'] = np.array(theta_data['data'])
CDM_data = np.array(CDM_data)

# remove the first 500 samples (transient response from convolution)
CDM_data = CDM_data[:,500:]                                                 # se queda un numpy array de shape (1002, 15500)
CDM_data_reshaped = []

for i in range(0,CDM_data.shape[0]):                                        # de 0 a 1002
    CDM_data_reshaped.append(CDM_data[i,:])                                 # añade cada numpy array con los datos de cada simulación (15500 valores)

features = np.array([pycatch22.catch22_all(CDM_data_reshaped[i])['values'] for i in range(len(CDM_data_reshaped))])  # features.shape = (1002, 22)


# pre-configured embedding network
embedding_net = FCEmbedding(
    input_dim=features.shape[1],
    num_hiddens=100,
    num_layers=2,
    output_dim=20,
)

# instantiate the SBI object
density_estimator_build_fun = posterior_nn(
    model="maf", hidden_features=100, num_transforms=2,
    embedding_net = embedding_net
)
inference = SNPE(density_estimator=density_estimator_build_fun)

# create splits of the 10-fold CV
kfold = KFold(n_splits=10, shuffle=True)
for kf, (train_index, test_index) in enumerate(kfold.split(features)):
        # TRAINING DATA
        train_theta = theta_data['data'][train_index,:]
        #train_CDM = CDM_data[train_index,:]
        train_features = features[train_index,:]
        
        # TEST DATA
        test_theta = theta_data['data'][test_index,:]
        #test_CDM = CDM_data[test_index,:]
        test_features = features[test_index,:]


# pass the simulated data to the inference object.
inference.append_simulations(
        torch.from_numpy(np.array(train_theta,dtype=np.float32)),
        torch.from_numpy(np.array(train_features,dtype=np.float32)))

# train the neural density estimator
density_estimator = inference.train()
print("\n")

# build the posterior
posterior = inference.build_posterior(density_estimator)

# Test the trained posterior
print('\n')
for i in range(0,10):
    # Randomly pick a sample
    sample = int(np.random.uniform(low = 0, high = len(test_theta)))
    print("Test sample: ", test_theta[sample])
    theta_o = torch.from_numpy(
                np.array(test_theta[sample],dtype=np.float32))  # Parámetros de esa simulación 
    x_o = torch.from_numpy(
                np.array(test_features[sample],dtype=np.float32))  # Valores del CDM para esa simulación 
    # Sample the posterior    
    posterior_samples = posterior.sample((5000,), x=x_o)

    # Plot posterior samples
    _ = analysis.pairplot(
        samples = posterior_samples,
        points=[theta_o],                       # Parámetros de la simulación 
        limits=None,
        figsize=(8, 5),
        upper="hist",
        points_colors=["red"],
        labels = theta_data['parameters'],
        points_offdiag=dict(marker="+", markersize=20)
    )
    plt.show()
