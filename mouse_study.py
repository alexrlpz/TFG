#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script to apply Simulation Based Inference to a real data database and study the relation 
# between mouses age and excitation/inhibition parameters

import pickle
from sklearn.preprocessing import StandardScaler
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import torch

# LOAD SIMULATIONS TO TRAIN SBI

save_data_path = '/home/alejandro/Escritorio/TFG/TFG/simulations_data/'
theta_data = {'parameters':['J_EE',
                            'J_IE',
                            'J_EI',
                            'J_II'],
              'data': []}

features = np.load(save_data_path + 'features.npy')
theta_data['data'] = np.load(save_data_path + 'theta_data.npy')
# Features normalization
scaler = StandardScaler()
features_norm = scaler.fit_transform(features)

# LOAD REAL DATABASE

data_path = 'mouses_data/Chini_LFP_baseline/catch22/emp_data'

# Import features calculated from mouses real data
mouses_data = pickle.load(open(data_path,'rb'))           # { 'ages': [4,6,8,10,12], 'data': [ [..], [..], ... ] }
mouses_features = mouses_data['data']

mouses_features_norm = []

# Normalize (Z-Score) features from data for each age (4,6,8,10,12)
for i in range(0,len(mouses_features)):
    normaliced = scaler.fit_transform(mouses_features[i])
    mouses_features_norm.append(normaliced)
    
# APPLY SBI

# create splits of the 10-fold CV
kfold = KFold(n_splits=10, shuffle=True)
for kf, (train_index, test_index) in enumerate(kfold.split(features_norm)):
        train_theta = theta_data['data'][train_index,:]                       # we only need to train
        train_features = features_norm[train_index,:]
        
# instantiate the SBI object
density_estimator_build_fun = posterior_nn(
    model="maf", z_score_x='structured', hidden_features=200, num_transforms=2
)
inference = SNPE(density_estimator=density_estimator_build_fun)
    
# pass the simulated data to the inference object.
inference.append_simulations(
        torch.from_numpy(np.array(train_theta,dtype=np.float32)),
        torch.from_numpy(np.array(train_features,dtype=np.float32)))

# train the neural density estimator
density_estimator = inference.train()
print("\n")

# build the posterior
posterior = inference.build_posterior(density_estimator)

final_results = np.zeros((len(mouses_features), 4))
for i in range(0,len(mouses_features)):                                          # for each age
    age_average_parameters = np.zeros((len(mouses_features[i]), 4))
    
    for j in range(0,len(mouses_features[i])):                                   # for each sample of that age
        posterior_samples = np.array(posterior.sample((4200,), x=mouses_features[i][j]))
        average_EE = np.mean(posterior_samples[:, 0])
        average_IE = np.mean(posterior_samples[:, 1])
        average_EI = np.mean(posterior_samples[:, 2])
        average_II = np.mean(posterior_samples[:, 3])
    
        age_average_parameters[j][0] = average_EE
        age_average_parameters[j][1] = average_IE
        age_average_parameters[j][2] = average_EI
        age_average_parameters[j][3] = average_II    
    
    final_results[i][0] = np.mean(age_average_parameters[:, 0])
    final_results[i][1] = np.mean(age_average_parameters[:, 1])
    final_results[i][2] = np.mean(age_average_parameters[:, 2])
    final_results[i][3] = np.mean(age_average_parameters[:, 3])
    
for i in range(0,len(mouses_features)):  
    print("\nResults for age: " + str(mouses_data['ages'][i]))
    print("\n Average J_EE: " + str(final_results[i][0]))
    print("\n Average J_IE: " + str(final_results[i][1]))
    print("\n Average J_EI: " + str(final_results[i][2]))
    print("\n Average J_II: " + str(final_results[i][3]))

excitation = []
inhibition = []
for i in range(0,len(mouses_features)):  
    print("\n")
    print("\nE/I_exc = " + str(np.abs(final_results[i][0]/final_results[i][2])))
    excitation.append(np.abs(final_results[i][0]/final_results[i][2]))
    print("\nE/I_inh = " + str(np.abs(final_results[i][1]/final_results[i][3])))
    inhibition.append(np.abs(final_results[i][1]/final_results[i][3]))
    

fig = plt.figure(figsize=[6,5], dpi=150)
ax1 = fig.add_axes([0.15,0.15,0.65,0.65])
plt.ylim(0, 0.15)
plt.plot(mouses_data['ages'], excitation, label='E/I_exc')
plt.plot(mouses_data['ages'], inhibition, label='E/I_inh')
plt.title('Excitation and inhibition across ages')
plt.xlabel('Age')
plt.ylabel('Value')
plt.legend()
#plt.text(0, 1.1,'Num transforms: ' + str(2),transform=ax1.transAxes, verticalalignment='bottom', fontsize=8) 
plt.show()