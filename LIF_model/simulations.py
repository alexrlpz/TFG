#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
import nest
import os
import scipy.signal as ss
import pickle
import numpy as np
import LIF_network
import json
import hashlib

'''
To do.
'''
def get_size(start_path = '.'):
    '''
    Walk all sub-directories, summing file sizes
    '''
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

# Create root folder
if not os.path.isdir('LIF_simulations'):
    os.mkdir('LIF_simulations')

# ground truth parameters
# for J_EE in [1.589]:
#     for J_IE in [2.020]:
#         for J_EI in [-23.84]:
#             for J_II in [-8.441]:

# Exploration of LIF network parameters
n_simulations = 1
currentsim = 1
for J_EE in np.linspace(0.5,4.,10):
    for J_IE in np.linspace(0.5,4.,10):
        for J_EI in np.linspace(-40.,-1.,10):
            for J_II in np.linspace(-40.,-1.,10):
                print("\n-------------------\n",end=' ', flush=True)
                print("\nJ_EE = %s, J_IE = %s, J_EI = %s, J_II = %s,\n" % (
                       J_EE,J_IE,J_EI,J_II),
                       end=' ', flush=True)
                print("Computing simulation %s out of %s\n" % (currentsim,
                                                             n_simulations),
                                                             end=' ', flush=True)
                currentsim+=1
                # constraints on the parameter space search
                if (-J_EI > 2.*J_EE) and (-J_EI > 2.*J_IE) and\
                   (-J_II > 2.*J_EE) and (-J_II > 2.*J_IE):
                    enable = True
                else:
                    enable = False

                if enable:
                    # Create hash
                    js_0 = json.dumps([J_EE,J_IE,J_EI,J_II],
                                       sort_keys=True).encode()
                    folder = hashlib.md5(js_0).hexdigest()
                    # Create folder
                    if not os.path.isdir('LIF_simulations/'+folder):
                        os.mkdir('LIF_simulations/'+folder)

                    # create the LIF_network object
                    LIF_net = LIF_network.LIF_network()

                    # load/create kernel
                    try:
                        LIF_net.H_YX = pickle.load(open('LIF_simulations/H_YX','rb'))
                    except (FileNotFoundError, IOError):
                        print("\nKernel not found. Computing kernel...\n",
                               end=' ', flush=True)
                        LIF_net.create_kernel()
                        # Save kernel to file
                        pickle.dump(LIF_net.H_YX,open('LIF_simulations/H_YX','wb'))

                    # Modify parameters
                    LIF_net.LIF_params['J_YX'] = [[J_EE, J_IE], [J_EI, J_II]]
                    #LIF_net.LIF_params['tau_syn_YX'] = [[tau_syn_E, tau_syn_I],
                                                        #[tau_syn_E, tau_syn_I]]
                    #LIF_net.LIF_params['J_ext'] = J_ext

                    # create the LIF network
                    LIF_net.create_LIF_network()

                    # Simulation
                    print('Simulating...\n',end=' ', flush=True)
                    tac = time()
                    LIF_net.simulate(tstop=LIF_net.tstop)
                    toc = time()
                    print(f'The simulation took {toc - tac} seconds.\n',
                            end=' ', flush=True)

                    # mean spike/firing rates
                    lif_mean_nu_X = dict()  # mean spike rates
                    lif_nu_X = dict()  # binned firing rate

                    for i, X in enumerate(LIF_net.LIF_params['X']):
                        times = nest.GetStatus(LIF_net.spike_recorders[X])[0]['events']['times']
                        times = times[times >= LIF_net.TRANSIENT]

                        lif_mean_nu_X[X] = LIF_net.get_mean_spike_rate(times)
                        bins, lif_nu_X[X] = LIF_net.get_spike_rate(times)

                    # # compute LFP signals
                    # probe = 'GaussCylinderPotential'
                    # LFP_data = dict(EE = [],EI = [],IE = [],II = [])
                    #
                    # for X in LIF_net.LIF_params['X']:
                    #     for Y in LIF_net.LIF_params['X']:
                    #         n_ch = LIF_net.H_YX[f'{X}:{Y}'][probe].shape[0]
                    #         for ch in range(n_ch):
                    #             # LFP kernel at electrode 'ch'
                    #             kernel = LIF_net.H_YX[f'{X}:{Y}'][probe][ch,:]
                    #             # LFP signal
                    #             sig = np.convolve(lif_nu_X[X], kernel, 'same')
                    #             # Decimate signal (x10)
                    #             LFP_data[f'{X}{Y}'].append(ss.decimate(
                    #                                         sig,
                    #                                         q=10,
                    #                                         zero_phase=True))

                    # compute CDM
                    probe = 'KernelApproxCurrentDipoleMoment'
                    CDM_data = dict(EE = [],EI = [],IE = [],II = [])

                    for X in LIF_net.LIF_params['X']:
                        for Y in LIF_net.LIF_params['X']:
                            # Pick only the z-component of the CDM kernel
                            kernel = LIF_net.H_YX[f'{X}:{Y}'][probe][2,:]
                            # CDM
                            sig = np.convolve(lif_nu_X[X], kernel, 'same')
                            CDM_data[f'{X}{Y}'] = ss.decimate(sig,
                                                              q=10,
                                                              zero_phase=True)

                    # Save simulation data to file
                    print('Saving data...\n',end=' ', flush=True)
                    pickle.dump(LIF_net.LIF_params,open('LIF_simulations/'+folder+'/LIF_params','wb'))
                    pickle.dump(LIF_net.TRANSIENT,open('LIF_simulations/'+folder+'/TRANSIENT','wb'))
                    pickle.dump(LIF_net.dt,open('LIF_simulations/'+folder+'/dt','wb'))
                    pickle.dump(LIF_net.tstop,open('LIF_simulations/'+folder+'/tstop','wb'))
                    pickle.dump(LIF_net.tau,open('LIF_simulations/'+folder+'/tau','wb'))

                    # pickle.dump(LFP_data,open('LIF_simulations/'+folder+'/LFP_data','wb'))
                    pickle.dump(CDM_data['EE']+\
                                CDM_data['EI']+\
                                CDM_data['IE']+\
                                CDM_data['II'],open('LIF_simulations/'+folder+'/CDM_data','wb'))
                    # pickle.dump(lif_mean_nu_X,open('LIF_simulations/'+folder+'/lif_mean_nu_X','wb'))
                    # pickle.dump([bins, lif_nu_X],open('LIF_simulations/'+folder+'/lif_nu_X','wb'))
                    #
                    # for i, Y in enumerate(LIF_net.LIF_params['X']):
                    #     pickle.dump(nest.GetStatus(LIF_net.spike_recorders[Y])[0]['events']['times'],
                    #                 open('LIF_simulations/'+folder+'/times_'+Y,'wb'))
                    #     pickle.dump(nest.GetStatus(LIF_net.spike_recorders[Y])[0]['events']['senders'],
                    #                 open('LIF_simulations/'+folder+'/gids_'+Y,'wb'))

                    print('Done!\n',end=' ', flush=True)

                    # Check size and remove simulations with large files
                    th = 10 # MB
                    os.chdir('LIF_simulations')
                    print("\nFolder size: %s MB\n" % str(get_size(
                                                    folder)/(2**20)))
                    if get_size(folder)/(2**20) > th:
                        os.system('rm -r '+folder)
                    os.chdir('..')
