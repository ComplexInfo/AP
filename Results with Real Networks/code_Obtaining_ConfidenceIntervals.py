######################################################################################################################################################################################################
## Importing the required packages

import numpy as np
from statistics import NormalDist
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import pandas as pd
# from scipy.stats import bernoulli
# from scipy.stats import pareto
import networkx as nx
# import matplotlib.pyplot as plt
# import random
# import scipy.io
# import collections
import pickle
# import copy

######################################################################################################################################################################################################
## Importing the required functions from from the Methods file

from Methods import *

######################################################################################################################################################################################################## 

####################### Case #######################
x = 0
y = 0

# Set initial conditions
Initial_H1_prob_blue = 0.8
Initial_H1_prob_red = 0.8

# Set parameters
R = 0.5
Alpha = 0.8
Beta = 0.2

Rho = 0.5
Inertia = 0.0

Alpha_b = Alpha
Beta_b = Beta
Rho_b = Rho
Inertia_b = Inertia

Alpha_r = Alpha
Beta_r = Beta
Rho_r = Rho
Inertia_r = Inertia


############ Model Prediction ############ 
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2))

FS = 11
row_label_scale = 0.36

# Plot the time series
plt.sca(axs[y])
SBM_Time_Series(Initial_H1_prob_blue, Initial_H1_prob_red, Alpha_b, Beta_b, Rho_b, Inertia_b, Alpha_r, Beta_r, Rho_r, Inertia_r, R, n = 5000000, T = 10, delta = 0.001)

# Plot the phaseplot
plt.sca(axs[y+1])
create_phase_plot(Initial_H1_prob_blue,  Initial_H1_prob_red, Alpha, Beta, R, Rho)

for y in [0]:
    axs[y].set_xlabel(r'Time $t$', fontsize=FS)    
    axs[y].set_ylabel(r'$\theta(t)$', fontsize=FS,labelpad=-1)
    axs[y].set_ylim(-0.01,1.01)
    axs[y].set_xticks(np.arange(0,11,2))  
    axs[y].axhline(y=0.5, color='darkgrey', linestyle='-.')
    axs[y].set_xticks([0, 2, 4, 6, 8, 10])   
    axs[y].set_yticks([0.00, 0.50, 1.00])        


for y in [1]:
    axs[y].set_xlabel(r'$\theta^{\mathcal{B}}$',fontsize = FS,labelpad=1.5)
    axs[y].set_ylabel(r'$\theta^{\mathcal{R}}$',fontsize = FS, labelpad=-2)
    axs[y].set_ylim(-0.0,1.0)
    axs[y].set_title(axs[y].get_title(), pad=10)
    axs[y].title.set_fontsize(FS)
    axs[y].set_xticks([0.00, 0.50, 1.00])
    axs[y].set_yticks([0.00, 0.50,1.00])

fig.suptitle('(i) 'r'$\alpha = $' + str(Alpha) + r', $\beta = $ ' + str(Beta) + r', $r = $' + str(R) + r', $\rho = $' + str(Rho), fontsize=12, y=1.035)
    
plt.subplots_adjust(wspace=0.3)  


############ Run the simulation multiple times to derive CIs ############ 
# num_runs = 10
# blue_series, red_series = run_simulation_multiple_times(num_runs, 'Brightkite_edges.npy', r = R, initial_H1_prob_blue = Initial_H1_prob_blue, initial_H1_prob_red  = Initial_H1_prob_red, alpha = Alpha, beta = Beta, c = 0.0, Intertia_delta = 0.0, Num_DataPoints = 0, Num_Steps = 500000)

num_runs = 50
graph = 'facebook_combined.npy'

blue_series, red_series = run_simulation_multiple_times(num_runs, graph, r = R, initial_H1_prob_blue = Initial_H1_prob_blue, initial_H1_prob_red  = Initial_H1_prob_red, alpha = Alpha, beta = Beta, c = 0.0, Intertia_delta = 0.0, Num_DataPoints = 0, Num_Steps = 500000)

name_blue = graph+'_blue_series_Alpha'+str(Alpha)+'_Beta_'+str(Beta)+'_R_'+str(R)+'_Rho_'+str(Rho)+'_InB_'+str(Initial_H1_prob_blue)+'_InR_'+str(Initial_H1_prob_red)+'.npy'
name_red = graph+'_red_series_Alpha'+str(Alpha)+'_Beta_'+str(Beta)+'_R_'+str(R)+'_Rho_'+str(Rho)+'_InB_'+str(Initial_H1_prob_blue)+'_InR_'+str(Initial_H1_prob_red)+'.npy'

np.save(name_blue, blue_series)
np.save(name_red, red_series)

######################################################################################################################################################################################################## 