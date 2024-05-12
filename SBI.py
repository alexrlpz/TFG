#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to do Simulation Based Inference

import numpy as np
import matplotlib.pyplot as plt

# Torch and SBI libraries
import torch
from sbi import analysis as analysis

from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# path to simulation data features and parameter values
save_data_path = '/home/alejandro/Escritorio/TFG/TFG/simulations_data/'


# parameters of the LIF_network object
theta_data = {'parameters':['J_EE',
                            'J_IE',
                            'J_EI',
                            'J_II'],
              'data': []}

features = np.load(save_data_path + 'features.npy')
theta_data['data'] = np.load(save_data_path + 'theta_data.npy')

# Z-Score normalization
# media = np.mean(features, axis=0)
# std_dev = np.std(features, axis=0)
# features_norm = (features - media) / std_dev

scaler = StandardScaler()
features_norm = scaler.fit_transform(features)

# Min-Max normalization
# min_vals = np.min(features, axis=0)
# max_vals = np.max(features, axis=0)

# features_norm = (features - min_vals) / (max_vals - min_vals)

# pre-configured embedding network
embedding_net = FCEmbedding(
    input_dim=features_norm.shape[1],
    num_hiddens=200,
    num_layers=2,
    output_dim=20,
)

# instantiate the SBI object
density_estimator_build_fun = posterior_nn(
    model="maf", z_score_x='structured', hidden_features=200, num_transforms=2
)
inference = SNPE(density_estimator=density_estimator_build_fun)

# create splits of the 10-fold CV
kfold = KFold(n_splits=10, shuffle=True)
for kf, (train_index, test_index) in enumerate(kfold.split(features_norm)):
        # TRAINING DATA
        train_theta = theta_data['data'][train_index,:]
        train_features = features_norm[train_index,:]
        
        # TEST DATA
        test_theta = theta_data['data'][test_index,:]
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
    pairplot[0].figure.text(0.03, 0.35, 'Theta_o: \n\n J_EE: ' + str("{:.1f}".format(theta_o[0].item())) + 
                           '\n J_IE:  ' + str("{:.1f}".format(theta_o[1].item())) + 
                           '\n J_EI: ' + str("{:.1f}".format(theta_o[2].item())) + 
                           '\n J_II:  ' + str("{:.1f}".format(theta_o[3].item()))                   
                            ,verticalalignment='bottom', fontsize=10)
    
    pairplot[0].figure.text(0.2, 0.025, 'PRE error: \n\n PRE(J_EE): ' + str("{:.2f}".format(PRE[0].item())) + 
                        '\n PRE(J_IE):  ' + str("{:.2f}".format(PRE[1].item())) + 
                        '\n PRE(J_EI): ' + str("{:.2f}".format(PRE[2].item())) + 
                        '\n PRE(J_II):  ' + str("{:.2f}".format(PRE[3].item()))                   
                        ,verticalalignment='bottom', fontsize=10)
    
    pairplot[0].figure.text(0.025, 0.025, 'Covariances: \n\n Cov(EE,IE):  ' + str("{:.2f}".format(cov_EE_IE.item())) + 
                        '\n Cov(EE,EI):  ' + str("{:.2f}".format(cov_EE_EI.item())) + 
                        '\n Cov(EE,II):  ' + str("{:.2f}".format(cov_EE_II.item())) + 
                        '\n Cov(IE,EI):  ' + str("{:.2f}".format(cov_IE_EI.item())) +
                        '\n Cov(IE,II):  ' + str("{:.2f}".format(cov_IE_II.item())) +
                        '\n Cov(EI,II):  ' + str("{:.2f}".format(cov_EI_II.item()))                 
                        ,verticalalignment='bottom', fontsize=10)
    
    pairplot[0].figure.text(0.37, 0.025, 'PPC error: \n\n' + str("{:.3f}".format(PPC.item()))           
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
