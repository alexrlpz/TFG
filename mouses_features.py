#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to prepare LFP data over mouse development for empirical validation with SBI tools.
"""

import sys
import os
import numpy as np
import pickle
import scipy.io
import scipy.signal as ss
import pycatch22

# path to empirical data
root_path = './'
data_path = os.path.join(root_path,
    'development_EI_decorrelation/baseline/LFP')

age_range = np.array([4,6,8,10,12])
emp_data = {'ages': age_range,
            'data':[[] for age in age_range]
            }

# choose between using features (use_feat = True) or raw data (False)
use_feat = True
# feature extraction method ('catch22')
feat_method = 'catch22'

# Size of the chunks into which the data trials will be divided
chunk_size = 500 # 5 s

# adapt sampling freq. of empirical data to sampling freq. of simulation data
adapt_sim = False

# sampling freq. of simulation data
fs_sim = 1000. / 0.625 # samples/s

ldir = os.listdir(data_path)
for i,file in enumerate(ldir):
    print(f'Loading file {i} out of {len(ldir)-1}')
    # get content from the mat structure
    structure = scipy.io.loadmat(data_path+'/'+file)
    age = structure['LFP']['age'][0][0][0][0]
    fs_emp = structure['LFP']['fs'][0][0][0][0]
    LFP = structure['LFP']['LFP'][0][0]
    sum_LFP = np.sum(LFP,axis = 0) # sum LFP across channels
    pos = np.where(emp_data['ages'] == age)[0]

    # check age
    if len(pos) > 0:
        # match sampling frequencies
        if fs_sim < fs_emp and adapt_sim:
            # downsample empirical data
            sum_LFP = ss.decimate(sum_LFP,
                                  q=int(fs_emp/fs_sim),
                                  zero_phase=True)

        # break empirical data into smaller chunks
        for ii in np.arange(0,len(sum_LFP)-chunk_size,chunk_size):
            sig = sum_LFP[ii:ii+chunk_size]
            # same preprocessing as simulation data
            sample = (sig - np.mean(sig))/np.std(sig)
            if use_feat:
                # catch22
                if feat_method == 'catch22':
                    feat = pycatch22.catch22_all(sample)
                    emp_data['data'][pos[0]].append(feat['values'])    
            else:
                emp_data['data'][pos[0]].append(sample)

# folder to save the new data
folder = str(fs_sim)+'_'+str(chunk_size) if adapt_sim else feat_method
# create folder
if not os.path.isdir(os.path.join(root_path,'mouses_data','Chini_LFP_baseline')):
    os.mkdir(os.path.join(root_path,'mouses_data','Chini_LFP_baseline'))
if not os.path.isdir(os.path.join(root_path,'mouses_data','Chini_LFP_baseline',folder)):
    os.mkdir(os.path.join(root_path,'mouses_data','Chini_LFP_baseline',folder))

# save data to file
pickle.dump(emp_data,
        open(os.path.join(root_path,'mouses_data','Chini_LFP_baseline',folder,
        'emp_data'),'wb'))
