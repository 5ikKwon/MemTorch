import torch
from MemTorch import memtorch
from MemTorch.memtorch import bh
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
import math

from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map
from memtorch.bh.crossbar.Program import naive_program
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities
import copy
import matplotlib.pyplot as plt


reference_memristor = memtorch.bh.memristor.CBRAM
reference_memristor_params = {'time_series_resolution': 1e-6,
                              'L':50e-9,
                              'v_h':0.35,
                              'v_r':700,
                              'E_a':0.5,
                              'rho_on':4,
                              'rho_off':1.33e4,
                              'A':1330,
                              'R_th':10e5,
                              'beta':0.3,
                              'alpha':2.4,
                              'C':0.08,
                              'I_comp':1e-6
                            }


memristor = reference_memristor(**reference_memristor_params)
palette = ["#DA4453", "#8CC152", "#4A89DC", "#F6BB42", "#B600B0", "#535353"]
f = plt.figure(figsize=(16/3, 4))
plt.title('Hysteresis Loop')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
j = 0
for i in range(10):
    j = j + 1
    if j == 6:
        j = 0

    memristor = reference_memristor(**reference_memristor_params)
    voltage_signal, current_signal = memristor.plot_hysteresis_loop(duration=2, voltage_signal_amplitude=0.2, voltage_signal_frequency = 1, return_result=True)
    plt.plot(voltage_signal, current_signal, color=palette[j])

plt.grid()
plt.show()