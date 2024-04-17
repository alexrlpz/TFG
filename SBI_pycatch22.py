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
import multiprocessing

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

#features = np.array([pycatch22.catch22_all(CDM_data_reshaped[i])['values'] for i in range(len(CDM_data_reshaped))])  # features.shape = (1002, 22)

# Z-Score normalization
media = np.mean(features, axis=0)
std_dev = np.std(features, axis=0)
features_norm = (features - media) / std_dev

# pre-configured embedding network
embedding_net = FCEmbedding(
    input_dim=features_norm.shape[1],
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
for kf, (train_index, test_index) in enumerate(kfold.split(features_norm)):
        # TRAINING DATA
        train_theta = theta_data['data'][train_index,:]
        #train_CDM = CDM_data[train_index,:]
        train_features = features_norm[train_index,:]
        
        # TEST DATA
        test_theta = theta_data['data'][test_index,:]
        #test_CDM = CDM_data[test_index,:]
        test_features = features_norm[test_index,:]


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
NUM_SAMPLES = 5000
total_PRE = []
total_covariances = []
total_PPC = []
print('\n')
for i in range(0,100):
    # Randomly pick a sample
    sample = int(np.random.uniform(low = 0, high = len(test_theta)))
    print("Test sample: ", test_theta[sample])
    theta_o = torch.from_numpy(
                np.array(test_theta[sample],dtype=np.float32))  # Parámetros reales de esa simulación 
    x_o = torch.from_numpy(
                np.array(test_features[sample],dtype=np.float32))  # Valores del CDM para esa simulación 
    # Sample the posterior    
    posterior_samples = posterior.sample((NUM_SAMPLES,), x=x_o)
    
    # Parameter recovery error (PRE)
    PRE = []
    for i in range(len(theta_o)):
        error = 0
        for j in range(len(posterior_samples)):
            error += np.abs(posterior_samples[j][i] - theta_o[i])
        
        PRE.append(error / NUM_SAMPLES)
        
    total_PRE.append(PRE)
    
    # Covariance between 2D marginals
    covariance_matrix = np.cov(posterior_samples, rowvar=False)
    covariances = []
    
    cov_EE_IE = covariance_matrix[0][1]
    cov_EE_EI = covariance_matrix[0][2]
    cov_EE_II = covariance_matrix[0][3]
    cov_IE_EI = covariance_matrix[1][2]
    cov_IE_II = covariance_matrix[1][3]
    cov_EI_II = covariance_matrix[2][3]
    
    covariances.append(cov_EE_IE)
    covariances.append(cov_EE_EI)
    covariances.append(cov_EE_II)
    covariances.append(cov_IE_EI)
    covariances.append(cov_IE_II)
    covariances.append(cov_EI_II)
    
    total_covariances.append(covariances)

    # Posterior Predictive Checks (PPC)
    PPC = 0
    posterior_samples = np.array(posterior_samples)
    x_o = np.array(x_o)
    all_x_pp = []
    
    for i in range(len(posterior_samples)):
        differences = np.abs(theta_data['data'] - posterior_samples[i])
        total_differences = np.sum(differences, axis=1)
        differences_ordered = np.argsort(total_differences)
        closest_params_position = differences_ordered[0]
        x_pp = features_norm[closest_params_position]
        all_x_pp.append(x_pp)
        features_errors = np.abs(x_pp-x_o)
        total_posterior_error = np.sum(features_errors) / len(features_errors)
        PPC += total_posterior_error
        
        
    PPC = PPC / NUM_SAMPLES
    total_PPC.append(PPC)

    # Plot posterior samples
    pairplot = analysis.pairplot(
        samples = posterior_samples,
        points=[theta_o],                       # Parámetros de la simulación 
        limits=None,
        figsize=(9.6, 6),
        upper="hist",
        points_colors=["red"],
        labels = theta_data['parameters'],
        points_offdiag=dict(marker="+", markersize=20)
    )
    pairplot[0].figure.text(0.03, 0.35, 'theta_o: \n\n J_EE: ' + str("{:.1f}".format(theta_o[0].item())) + 
                           '\n J_IE:  ' + str("{:.1f}".format(theta_o[1].item())) + 
                           '\n J_EI: ' + str("{:.1f}".format(theta_o[2].item())) + 
                           '\n J_II:  ' + str("{:.1f}".format(theta_o[3].item()))                   
                            ,verticalalignment='bottom', fontsize=10)
    
    pairplot[0].figure.text(0.25, 0.025, 'PRE error: \n\n PRE(J_EE): ' + str("{:.2f}".format(PRE[0].item())) + 
                        '\n PRE(J_IE):  ' + str("{:.2f}".format(PRE[1].item())) + 
                        '\n PRE(J_EI): ' + str("{:.2f}".format(PRE[2].item())) + 
                        '\n PRE(J_II):  ' + str("{:.2f}".format(PRE[3].item()))                   
                        ,verticalalignment='bottom', fontsize=10)
    
    pairplot[0].figure.text(0.025, 0.025, 'Covariance: \n\n Cov(EE,IE):  ' + str("{:.2f}".format(cov_EE_IE.item())) + 
                        '\n Cov(EE,EI):  ' + str("{:.2f}".format(cov_EE_EI.item())) + 
                        '\n Cov(EE,II):  ' + str("{:.2f}".format(cov_EE_II.item())) + 
                        '\n Cov(IE,EI):  ' + str("{:.2f}".format(cov_IE_EI.item())) +
                        '\n Cov(IE,II):  ' + str("{:.2f}".format(cov_IE_II.item())) +
                        '\n Cov(EI,II):  ' + str("{:.2f}".format(cov_EI_II.item()))                 
                        ,verticalalignment='bottom', fontsize=10)
    
    
    all_x_pp = np.array(all_x_pp)
    pairplot2 = analysis.pairplot(
        samples=all_x_pp,
        points=x_o,
        points_colors="red",
        figsize=(12, 12),
        offdiag="scatter",
        scatter_offdiag=dict(marker=".", s=5),
        points_offdiag=dict(marker="+", markersize=20),
        labels = [r"$x_{{{}}}$".format(d) for d in range(22)]
    )
    pairplot2[0].figure.text(0.18, 0.065, 'PPC error: \n\n' + str("{:.3f}".format(PPC.item()))               
                        ,verticalalignment='bottom', fontsize=15)
    plt.show()
    
    
print('\nGlobal results:\n')
print('\nAverage Parameter Recovery Error (PRE):\n')

total_PRE = np.array(total_PRE)
parameters = ['J_EE', 'J_IE', 'J_EI', 'J_II']
PRE_averages = [np.mean(total_PRE[:, i]) for i in range(4)]

for name, average in zip(parameters, PRE_averages):
    print(f"\nPRE({name}): {average}")

print('\nAverage Covariances:\n')

total_covariances = np.array(total_covariances)
covariances_types = ['EE_IE', 'EE_EI', 'EE_II', 'IE_EI', 'IE_II', 'EI_II']
covarianze_averages = [np.mean(total_covariances[:, i]) for i in range(6)]

for name, average in zip(covariances_types, covarianze_averages):
    print(f"\nCov({name}): {average}")
    
print('\nAverage Posterior Predictive Checks (PPC):\n')

average_PPC = np.mean(total_PPC)

print("\nPPC: ", average_PPC)
