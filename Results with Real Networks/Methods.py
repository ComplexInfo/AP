import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import bernoulli
from scipy.stats import pareto
import networkx as nx
import matplotlib.pyplot as plt
import random
import scipy.io
import collections
import pickle
import copy


############################################################################################################################################

"""
Simulating on a graph
"""

def AP_Sim_4(Graph_Name, r = 0.5, initial_H1_prob_blue = 0.5, initial_H1_prob_red = 0.5, alpha = 0, beta = 0, c = 0, Intertia_delta = 0.0, Num_DataPoints = 1, Num_Steps = 100000, show = 1):
    
    ###
    # Graph_Name: string name of the saved adjacency matrix
    # r (fraction of Red nodes)
    # initial_H1_prob_blue/initial_H1_prob_red: initial fraction of correct beliefs in the two parties
    # Num_Steps Number of iterations to run
    # alpha (trust in same party) between 0 and 1 
    # beta (mistrust in other party) between 0 and 1
    # Num_DataPoints: number of obsevations per person
    # (confirmation bias) between 0 and 1
    ###
    
    H1_Fraction_Blue = []
    H1_Fraction_Red = []
    
    TR=[]
    TB=[]
           
    #Obtaining the adjacency matrix of the graph G
    A = np.load(Graph_Name) 
    G = nx.from_numpy_array(A)

    Nodes = list(np.arange(A.shape[0]))
    Num_Nodes = A.shape[0]

    #degree vector of the graph G = (V,E)
    d_vec = [val for (node, val) in G.degree()]

    # Deciding the political affiliation of each node as iid Bernoulli random variables (a node is red with probability r)
    R = np.array([bernoulli.rvs(r, size=1) for v in G.nodes])
    R = R.flatten()

    #Initial beliefs (assigned to the two groups based on two initial probabilities)
    H = [bernoulli.rvs(initial_H1_prob_red, size=1) if R[v] ==1 else bernoulli.rvs(initial_H1_prob_blue, size=1) for v in G.nodes]
    H = np.array(H)
    H = H.flatten()
    initial_true_fraction_red = np.multiply(R==1,H)/np.sum(R==1)
    initial_true_fraction_blue = np.multiply(R==0,H)/np.sum(R==0)

    # Dynamical System 
    step = 0
    while step < Num_Steps:
        step+=1
        node = random.choice(Nodes)
        # print ('Random Node = ' + str(node))

        #Computing the prior
        neighbors = list(G.neighbors(node))
        
        #Degree of the node
        d = d_vec[node]

        if R[node] == 0:
            # print('chosen node is blue (0)')
            if d != 0:
                neighbors_with_H_0_and_R_0 = [neighbor for neighbor in neighbors if R[neighbor] == 0 and H[neighbor] == 0]
                FracNeigbors_SameParty_AgreesWith0 = len(neighbors_with_H_0_and_R_0)/d
                
                neighbors_with_H_1_and_R_0 = [neighbor for neighbor in neighbors if R[neighbor] == 0 and H[neighbor] == 1]
                FracNeigbors_SameParty_AgreesWith1 = len(neighbors_with_H_1_and_R_0)/d
            else:
                FracNeigbors_SameParty_AgreesWith0 = 0
                FracNeigbors_SameParty_AgreesWith1 = 0
            if d != 0:
                neighbors_with_H_0_and_R_1 = [neighbor for neighbor in neighbors if R[neighbor] == 1 and H[neighbor] == 0]
                FracNeigbors_OppositeParty_AgreesWith0 = len(neighbors_with_H_0_and_R_1)/d
                
                neighbors_with_H_1_and_R_1 = [neighbor for neighbor in neighbors if R[neighbor] == 1 and H[neighbor] == 1]
                FracNeigbors_OppositeParty_AgreesWith1 = len(neighbors_with_H_1_and_R_1)/d                
            else:
                FracNeigbors_OppositeParty_AgreesWith0 = 0
                FracNeigbors_OppositeParty_AgreesWith1 = 0
            
        if R[node] == 1:
            # print('chosen node is red (1)')
            if d != 0:
                neighbors_with_H_0_and_R_0 = [neighbor for neighbor in neighbors if R[neighbor] == 0 and H[neighbor] == 0]
                FracNeigbors_OppositeParty_AgreesWith0 = len(neighbors_with_H_0_and_R_0)/d
                
                neighbors_with_H_1_and_R_0 = [neighbor for neighbor in neighbors if R[neighbor] == 0 and H[neighbor] == 1]
                FracNeigbors_OppositeParty_AgreesWith1 = len(neighbors_with_H_1_and_R_0)/d
            else:
                FracNeigbors_OppositeParty_AgreesWith0 = 0
                FracNeigbors_OppositeParty_AgreesWith1 = 0
            if d != 0:
                neighbors_with_H_0_and_R_1 = [neighbor for neighbor in neighbors if R[neighbor] == 1 and H[neighbor] == 0]
                FracNeigbors_SameParty_AgreesWith0 = len(neighbors_with_H_0_and_R_1)/d
                
                neighbors_with_H_1_and_R_1 = [neighbor for neighbor in neighbors if R[neighbor] == 1 and H[neighbor] == 1]
                FracNeigbors_SameParty_AgreesWith1 = len(neighbors_with_H_1_and_R_1)/d
            else:
                FracNeigbors_SameParty_AgreesWith0 = 0
                FracNeigbors_SameParty_AgreesWith1 = 0
            
        ###### To visualize the subgraph spanned by chosen v and its neighbors, uncomment the part below and set the Num_Steps to 1
        # subgraph_nodes = [node] + list(G.neighbors(node))
        # subgraph = G.subgraph(subgraph_nodes)
        # node_colors = {v: 'blue' if R[v] == 0 else 'red' for v in subgraph.nodes()}
        # node_labels = {v: H[v] for v in subgraph.nodes()}
        # node_labels[node] = str(H[node])+'[v]'
        # pos = nx.spring_layout(subgraph)
        # nx.draw(subgraph, pos, node_color=[node_colors[n] for n in subgraph.nodes()], labels=node_labels, with_labels=True, node_size=1000, font_size=12)
            
        # print('FracNeigbors_SameParty_AgreesWith H0 = ' + str(FracNeigbors_SameParty_AgreesWith0))
        # print('FracNeigbors_OppositeParty_AgreesWith H0 = ' + str(FracNeigbors_OppositeParty_AgreesWith0))
        ######
        
        Control_Parameter = 0.3
        
        In_01_Diff = (FracNeigbors_SameParty_AgreesWith0-FracNeigbors_SameParty_AgreesWith1)
        # if In_01_Diff > Control_Parameter:
        #     In_01_Dif = Control_Parameter
        # if In_01_Diff < -Control_Parameter:
        #     In_01_Dif = -Control_Parameter
            
        
        
        Out_01_Diff = (FracNeigbors_OppositeParty_AgreesWith0-FracNeigbors_OppositeParty_AgreesWith1)
        # if Out_01_Diff > Control_Parameter:
        #     Out_01_Dif = Control_Parameter
        # if Out_01_Diff < -Control_Parameter:
        #     Out_01_Dif = -Control_Parameter
        
        
        prior = np.array([0.5, 0.5])
        Adj = np.array([0.5*alpha*In_01_Diff - 0.5*beta*Out_01_Diff, -0.5*alpha*In_01_Diff  + 0.5*beta*Out_01_Diff])
        prior = prior + Adj
        prior = np.array([prior])

###### To see the prior of the chosen node, uncomment the part below and set the Num_Steps to 1     
#         print('Adj = ' + str(Adj))
#         print('prior = ')
#         print(prior)
#         if prior[0,0]<prior[0,1]: #prior belief more on H1 (tail biased coin)
#             print('prior is more aligned with H1: coin is biased towards tails')
#         elif prior[0,0]>prior[0,1]: #prior belief more on H0 (heads biased coin)
#             print('prior is more aligned with H0: coin is biased towards heads')
#         else:
#             print('prior is unbiased')
        
#         print('')
        
        if Num_DataPoints > 0:
            Data_iteration = 0
            while Data_iteration < Num_DataPoints:
                #Observation
                Observation = random.choices([np.array([1, 0]), np.array([0, 1])], weights=[0.3,0.7], k = 1) 
                # heads = np.array([1, 0]) tails = np.array([0, 1])
                # H0: coin is biased towards heads [1, 0] with prob. 0.7 
                # H1:  coin is biased towards tails [0, 1] with prob. 0.7 (true)
                # P(0  | H0) = 0.7  #P(1  | H0) = 0.3    
                # P(0  | H1) = 0.3  #P(1  | H1) = 0.7   

                #Likelihood matrix element values

                # Col0:Obs 0(H) Col1: Obs 1(T)
                p_H0_0 = 0.7; p_H0_1 = 0.3 # row 0

                p_H1_0 = 0.3; p_H1_1 = 0.7 # row 1


                # if (Observation[0] == [1, 0]).all():
                #     print('Observation =' + str(Observation) + ': Heads')
                # else:
                #     print('Observation =' + str(Observation) + ': Tails')

                #Computing the likelihood
                if H[node] == 1: #current belief is on H1 (tail biased coin)
                    Likelihood = np.matrix([[p_H0_0**(1-c),p_H0_1],[p_H1_0**(1-c),p_H1_1]]) #discount the heads observation
                if H[node] == 0: #current belief is on H0 (heads biased coin)
                    Likelihood = np.matrix([[p_H0_0,p_H0_1**(1-c)],[p_H1_0,p_H1_1**(1-c)]]) #discount the tails observation

                # print('Likelihood Matrix with Confirmation Bias c = ' + str(c))
                # print(Likelihood)

                # Calculate the posterior probabilities based on observations
                posterior = np.multiply(np.matmul(Likelihood, *Observation), prior)
                # print('posterior = ')
                # print(posterior)

                prior = posterior

                Data_iteration+=1
        else:
            posterior = prior 

        if posterior[0,1] - posterior[0,0]>Intertia_delta:
            H[node] = 1
            # print('updated decision = ' + str(H[node]) + ' ( H1: Coin is biased towards Tails )')
        if posterior[0,1] - posterior[0,0]<Intertia_delta:
            H[node] = 0
            # print('updated decision = ' + str(H[node]) + '( H0: Coin is biased towards Heads )')
        # if posterior[0,1] == posterior[0,0]:
        #     H[node] = random.choices([0,1], weights=[0.5,0.5])[0]
            
        num_Blue = np.sum(R==0)    
        num_Red = np.sum(R==1)    
            
        blue_H1_frac = np.sum(np.multiply(H,R==0))/num_Blue
        red_H1_frac = np.sum(np.multiply(H,R==1))/num_Red
        # print('false_count  = ' + str(false_count))

    #     print('')
        H1_Fraction_Blue.append(blue_H1_frac)
        H1_Fraction_Red.append(red_H1_frac)
        
        TR.append(alpha*num_Red*(2*red_H1_frac-1) - beta*num_Blue*(2*blue_H1_frac-1))
        TB.append(alpha*num_Blue*(2*blue_H1_frac-1) - beta*num_Red*(2*red_H1_frac-1))

    # print('')
    # print('alpha = ' + str(alpha))
    # print('beta = ' + str(beta))
    # print('c = ' + str(c))
    # print('Num_DataPoints = ' + str(Num_DataPoints))
    # print('initial_false_count  = ' + str(initial_false_count))        
    # print('Final false_count  = ' + str(false_count))

    # print('')
    # return (H1_Fraction_Blue,H1_Fraction_Red)
    if show == 1:
        plt.figure()
        plt.plot(H1_Fraction_Blue,'b')
        plt.plot(H1_Fraction_Red,'r')
        plt.ylim(-0.01,1.01)
    
    return (H1_Fraction_Blue, H1_Fraction_Red, H, R, G)
    
#     plt.figure()
#     plt.plot(TB,'b')
#     plt.plot(TR,'r')
#     plt.axhline(y = 0.0, color = 'y', linestyle = '-.') 

############################################################################################################################################

# Define the system of differential equations
def AP_model_SBM_Graph(theta_t_b,theta_t_r, t, Alpha_b, Beta_b, Rho_b, Inertia_b, Alpha_r, Beta_r, Rho_r, Inertia_r, R, n):
    b_in_link_prob = (1-R)*Rho_b
    b_out_link_prob = R*(1-Rho_b)
    
    mu_b = np.array([b_in_link_prob*theta_t_b, b_in_link_prob*(1-theta_t_b), b_out_link_prob*theta_t_r, b_out_link_prob*(1-theta_t_r)]).reshape(4,1)
    Sigma_b = np.zeros((4,4))
    for i in [0,1,2,3]:
        Sigma_b[i,i] = mu_b[i,0]*(1-mu_b[i,0])/n

    r_in_link_prob = R*Rho_r
    r_out_link_prob = (1-R)*(1-Rho_r)                         
        
    mu_r = np.array([r_in_link_prob*theta_t_r, r_in_link_prob*(1-theta_t_r), r_out_link_prob*theta_t_b, r_out_link_prob*(1-theta_t_b)]).reshape(4,1)
    Sigma_r = np.zeros((4,4))
    for i in [0,1,2,3]:
        Sigma_r[i,i] = mu_r[i,0]*(1-mu_r[i,0])/n
                                        
                    
    coeffs_blue = np.array([Alpha_b, -Alpha_b, -Beta_b, Beta_b]).reshape(4,1)
    coeffs_red = np.array([Alpha_r, -Alpha_r, -Beta_r, Beta_r]).reshape(4,1)   
    
                    
#     p_b_01 = 1-NormalDist(mu=np.matmul(coeffs_blue.transpose(),mu_b)-Inertia_b, sigma=np.sqrt(np.matmul(np.matmul(coeffs_blue.transpose(),Sigma_b),coeffs_blue))).cdf(0)
#     p_b_10 = NormalDist(mu=np.matmul(coeffs_blue.transpose(),mu_b)+Inertia_b, sigma=np.sqrt(np.matmul(np.matmul(coeffs_blue.transpose(),Sigma_b),coeffs_blue))).cdf(0)                          
        
#     p_r_01 = 1-NormalDist(mu=np.matmul(coeffs_red.transpose(),mu_r)-Inertia_r, sigma=np.sqrt(np.matmul(np.matmul(coeffs_red.transpose(),Sigma_r),coeffs_red))).cdf(0)
#     p_r_10 = NormalDist(mu=np.matmul(coeffs_red.transpose(),mu_r)+Inertia_r, sigma=np.sqrt(np.matmul(np.matmul(coeffs_red.transpose(),Sigma_r),coeffs_red))).cdf(0)      
    
    if np.matmul(coeffs_blue.transpose(),mu_b)-Inertia_b > 0:
        p_b_01 = 1
    else:
        p_b_01 = 0
    if np.matmul(coeffs_blue.transpose(),mu_b)+Inertia_b > 0:
        p_b_10 = 1
    else:
        p_b_10 = 0
        
    if np.matmul(coeffs_red.transpose(),mu_r)-Inertia_r > 0:
        p_r_01 = 1
    else:
        p_r_01 = 0
    if np.matmul(coeffs_red.transpose(),mu_r)+Inertia_r > 0:
        p_r_10 = 1
    else:
        p_r_10 = 0        
                        
    theta_t_blue_dot = (1-theta_t_b)*p_b_01 - theta_t_b*p_b_10
    theta_t_red_dot = (1-theta_t_r)*p_r_01 - theta_t_r*p_r_10                        
                                                
    return [theta_t_blue_dot, theta_t_red_dot]                   


############################################################################################################################################

def Simulate_AP_model_SBM_Graph(Initial_H1_prob_blue, Initial_H1_prob_red, Alpha_b, Beta_b, Rho_b, Intertia_b, Alpha_r, Beta_r, Rho_r, Intertia_r, R, n = 5000000, T = 10, delta = 0.001):
    
    theta_t_blue = [Initial_H1_prob_blue]
    theta_t_red = [Initial_H1_prob_red]

    T = 10
    time = np.arange(0,T,delta)
    for t in time[:-1]:
        (theta_t_blue_dot, theta_t_red_dot) = AP_model_SBM_Graph(theta_t_blue[-1],theta_t_red[-1], t, Alpha_b, Beta_b, Rho_b, Intertia_b, Alpha_r, Beta_r, Rho_r, Intertia_r, R, n)
        theta_t_blue.append(theta_t_blue[-1] + delta*theta_t_blue_dot)
        theta_t_red.append(theta_t_red[-1] + delta*theta_t_red_dot)
    
    return time, theta_t_blue, theta_t_red

############################################################################################################################################
def dynamical_system_gradient(x, y, Alpha,Beta,R, Rho):
    Alpha_b = Alpha
    Beta_b = Beta
    Rho_b = Rho

    Alpha_r = Alpha
    Beta_r = Beta
    Rho_r = Rho
    
    # Initialize arrays to store gradient values
    dx_dt_grid = np.zeros_like(x)
    dy_dt_grid = np.zeros_like(y)
    
    # Iterate over each point in the grid
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            # Compute the gradient at each point on the grid
            dx_dt, dy_dt = AP_model_SBM_Graph(x[i, j], y[i, j], 1, Alpha_b, Beta_b, Rho_b, 0, Alpha_r, Beta_r, Rho_r, 0, R, 5E10)
            # Store the gradient values
            dx_dt_grid[i, j] = dx_dt
            dy_dt_grid[i, j] = dy_dt
    
    return dx_dt_grid, dy_dt_grid

# Create a function to generate the phase plot
def create_phase_plot(Initial_H1_prob_blue,  Initial_H1_prob_red, Alpha,Beta,R, Rho=0.5, theoretical = 0):
    
    def y1_function(x, alpha, beta, r, rho):
        return (rho*alpha*(1-r)/((1-rho)*beta*r)) * x - rho*alpha*(1-r)/(2*(1-rho)*beta*r) + 0.5

    def y2_function(x, alpha, beta, r, rho):
        return ((1-rho)*beta*(1-r)/(rho*alpha*r)) * x - (1-rho)*beta*(1-r)/(2*rho*alpha*r) + 0.5
    
    Alpha_b = Alpha
    Beta_b = Beta
    Rho_b = Rho

    Alpha_r = Alpha
    Beta_r = Beta
    Rho_r = Rho
    
    # Define the range of x and y values
    
    n_grid = 20
    scale = 15
    
    x_values = np.linspace(0, 1, n_grid)
    y_values = np.linspace(0, 1, n_grid)
    
    # Create a grid of x and y values
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    
    # Compute the gradient at each point on the grid
    dx_dt_grid, dy_dt_grid = dynamical_system_gradient(x_grid, y_grid,Alpha,Beta,R,Rho)
    
    # Normalize the gradients for better visualization
    magnitude = np.sqrt(dx_dt_grid ** 2 + dy_dt_grid ** 2)
    dx_dt_grid /= magnitude
    dy_dt_grid /= magnitude
    
    # Plot the phase plot
    # plt.figure(figsize=(8, 6))
    plt.quiver(x_grid, y_grid, dx_dt_grid, dy_dt_grid, scale=scale)

    # Define x values
    x_values = np.linspace(0, 1, 100)

    # Calculate y values
    y1_values = y1_function(x_values, Alpha, Beta, R, Rho)
    y2_values = y2_function(x_values, Alpha, Beta, R, Rho)

    y_0 = np.ones(100)
    y_1 = np.ones(100)

    # Plot
    plt.fill_between(x_values, y1_values, 0, color='blue', alpha=0.3)
    plt.fill_between(x_values, y2_values, 1, color='red', alpha=0.3)

    plt.plot(x_values, y1_values, label='y1', color='tab:blue')
    plt.plot(x_values, y2_values, label='y2', color='tab:red')
    
    # plt.plot(x_values, x_values, color='green', linestyle='--', label='y=x')    
    plt.plot(x_values, x_values, color='dimgrey', linestyle='--', label='y=x')        
    
    
    #Plotting the trajectory
    theta_t_blue = [Initial_H1_prob_blue]
    theta_t_red = [Initial_H1_prob_red]
    T = 10
    delta = 0.001
    time = np.arange(0,T,delta)
    for t in time[:-1]:
        (theta_t_blue_dot, theta_t_red_dot) = AP_model_SBM_Graph(theta_t_blue[-1],theta_t_red[-1], t, Alpha_b, Beta_b, Rho_b, 0, Alpha_r, Beta_r, Rho_r, 0, R, 5E10)
        theta_t_blue.append(theta_t_blue[-1] + delta*theta_t_blue_dot)
        theta_t_red.append(theta_t_red[-1] + delta*theta_t_red_dot)
    
    # arrow_interval = 150
    # for i in range(0, len(theta_t_blue)-1, arrow_interval):
    #     plt.arrow(theta_t_blue[i], theta_t_red[i], theta_t_blue[i+1]-theta_t_blue[i], theta_t_red[i+1]-theta_t_red[i], 
    #               shape='full', lw=1,length_includes_head=True, head_width=0.03, color='gold', fill=True)     
    # plt.plot(theta_t_blue, theta_t_red, color='yellow', linestyle = '--', linewidth = 1)   
    
    if theoretical == 1:
        if Beta/Alpha > (R/(1-R)) and Alpha/Beta < (R/(1-R)):
            if Initial_H1_prob_blue > 0.5:                                              
                theta_t_blue = [x for x in theta_t_blue if x > 0.5]
                theta_t_red = [x for x in theta_t_red if x > 0.5]
            if Initial_H1_prob_blue < 0.5:                                              
                theta_t_blue = [x for x in theta_t_blue if x < 0.5]
                theta_t_red = [x for x in theta_t_red if x < 0.5]  
    
    arrow_interval = 150
    for i in range(0, len(theta_t_blue)-1, arrow_interval):
        plt.arrow(theta_t_blue[i], theta_t_red[i], theta_t_blue[i+1]-theta_t_blue[i], theta_t_red[i+1]-theta_t_red[i], 
                  shape='full', lw=1,length_includes_head=True, head_width=0.05, color='purple', fill=True)     
    plt.plot(theta_t_blue, theta_t_red, color='purple', linestyle = '--', linewidth = 1)      
    
    plt.xticks([0.00, 0.25, 0.50, 0.75, 1.00])
    plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])    
    
    plt.xlabel(r'$\theta_{t}^{(\mathcal{B})}$', fontsize=15)
    plt.ylabel(r'$\theta_{t}^{\mathcal{R}}$', fontsize=15)    
    # plt.title('Phase Plot of the Dynamical System')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.show()
############################################################################################################################################
def SBM_Time_Series(Initial_H1_prob_blue, Initial_H1_prob_red, Alpha_b, Beta_b, Rho_b, Intertia_b, Alpha_r, Beta_r, Rho_r, Intertia_r, R, n = 5000000, T = 10, delta = 0.001):
    time, theta_blue, theta_red = Simulate_AP_model_SBM_Graph(Initial_H1_prob_blue, Initial_H1_prob_red, Alpha_b, Beta_b, Rho_b, Intertia_b, Alpha_r, Beta_r, Rho_r, Intertia_r, R, n = 5000000, T = 10, delta = delta)
    plt.plot(time, theta_blue, label=r'$\theta^{\mathcal{B}}(t)$',c='blue',linestyle = '-.',linewidth=2)
    plt.plot(time, theta_red, label=r'$\theta^{\mathcal{R}}(t)$', c='red', linestyle = ':',linewidth=2.25)
    plt.xlabel(r'Time $t$', fontsize=15)    
    plt.ylabel(r'$\theta_t$', fontsize=15)    
    plt.legend()
############################################################################################################################################
def parameter_location(Alpha,Beta, R, Rho):

    # Define the functions
    def y1(x):
        return x

    def y2(x):
        return 1/x

    def max_y1y2(x):
        return [max(i,1/i) for i in x]

    def min_y1y2(x):
        return [min(i,1/i) for i in x]

    y_min = 0.0001
    y_max = 4

    # Generate x values in the range 0 <= x < 5
    x_values = np.linspace(y_min, y_max, 400)  # Start from 0.01 to avoid division by zero in y2

    # Generate y values for y1 and y2
    y1_values = y1(x_values)
    y2_values = y2(x_values)

    # Plot the curves y1 = x and y2 = 1/x
    plt.plot(x_values, y1_values, label=r'$y_1 = x$', linestyle = ':', linewidth=2, color='k')
    plt.plot(x_values, y2_values, label=r'$y_2 = 1/x$', linestyle = '--', linewidth=2, color='k')

    # Fill the quadrants with different colors
    plt.fill_between(x_values, 0, min_y1y2(x_values), interpolate=True, color='blue', alpha=0.3, label=r'$y < y_1, y_2$')
    plt.fill_between(x_values, y1_values, y2_values, where=((y1_values < y2_values)), interpolate=True, color='green', alpha=0.3, label=r'$y_1<y < y_2$')
    plt.fill_between(x_values, y1_values, y2_values, where=((y1_values > y2_values)), interpolate=True, color='orange', alpha=0.3, label=r'$y_2<y < y_1$')
    plt.fill_between(x_values, y_max, max_y1y2(x_values), interpolate=True, color='red', alpha=0.3, label=r'$y > y_1, y_2$')


    # Add labels and legend
    plt.xlabel(r'${\alpha}/{\beta}$',fontsize = 12)
    plt.ylabel(r'${r}/{(1-r)}$',fontsize = 12)

    plt.ylim(y_min,y_max)
    plt.xlim(y_min,y_max)

    plt.xticks([0, 1, 2, 3, 4])
    plt.yticks([0, 1, 2, 3, 4])    

    # plt.axhline(1, color='grey', linewidth=0.5)
    # plt.axvline(1, color='grey', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    # plt.legend()

    plt.plot(Alpha*Rho/(Beta*(1-Rho)), R/(1-R), 'r*', markersize=10)
    
    # # Show the plot
    # plt.tight_layout(pad=1)          
    # plt.legend(loc='center right', fontsize = 11, edgecolor = 'inherit', ncol = 1)        
    # plt.savefig('FC_FourRegions' + '.pdf', bbox_inches='tight') 
############################################################################################################################################
def run_simulation_multiple_times(num_times, Graph_Name, r = 0.5, initial_H1_prob_blue = 0.5, initial_H1_prob_red = 0.5, alpha = 0, beta = 0, c = 0, Intertia_delta = 0.0, Num_DataPoints = 1, Num_Steps = 100000):
    all_blue_series = []
    all_red_series = []
    
    for _ in range(num_times):
        (Blue_Time_Series, Red_Time_Series, _, _, _) = AP_Sim_4(Graph_Name, r, initial_H1_prob_blue, initial_H1_prob_red, alpha, beta, c, Intertia_delta, Num_DataPoints, Num_Steps, show = 0)
        all_blue_series.append(Blue_Time_Series)
        all_red_series.append(Red_Time_Series)
    
    return np.array(all_blue_series), np.array(all_red_series)

def compute_confidence_band(series, confidence_level=0.95):
    mean_series = np.mean(series, axis=0)
    std_dev_series = np.std(series, axis=0)
    num_samples = series.shape[0]
    margin_of_error = std_dev_series * 1.96 / np.sqrt(num_samples)  # 1.96 for 95% confidence interval
    lower_bound = mean_series - margin_of_error
    upper_bound = mean_series + margin_of_error
    return lower_bound, upper_bound
############################################################################################################################################

############################################################################################################################################

############################################################################################################################################
