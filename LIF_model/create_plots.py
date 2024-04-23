#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import subprocess
import os
    
# path to simulation data
data_path = '/home/alejandro/Escritorio/TFG/TFG/LIF_model/LIF_simulations'

# Load simulation data
ldir = os.listdir(data_path)
for i,folder in enumerate(ldir):
    print(f'Generating plots of file {i} out of {len(ldir)}')
    subprocess.run(["python3", "plots.py", folder])