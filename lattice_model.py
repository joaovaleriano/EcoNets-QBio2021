#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:32:54 2021

@author: joao-valeriano
"""

# Import libraries
import numpy as np # Dealing with arrays
import matplotlib.pyplot as plt # Plotting
from matplotlib import colors # Dealing with color scales for plots
from numba import njit # Pre-compiling function for improved performance
from tqdm import tqdm # Print progressbars
import os # Operational systems commands

# 0: cooperator
# 1: defector

# Check if folder for saving figures exists, and if not, create it
if "lattice_model_movie" not in os.listdir():
    
    os.mkdir("lattice_model_movie") # Create folder

# np.random.seed(5) # Set seed for random number generation

# Generate initial network with random conditions
def gen_strat_mat(n=10, coop=0.5):
    # n: lattice size -> number of nodes = n^2
    # coop: cooperator proportion
        
    defec = 1-coop # defector proportion
    
    # Generate random n by n array with 0s (cooperators) and 1s (defectors), with given proportions
    strat_mat = np.random.choice([0,1], (n,n), p=[coop, defec])
    
    return strat_mat
    

# Generate list of neighbors for a given node
@njit # Decorator for pre-compiling function with numba
def neighbors(i, j, n):
    # i: row coordinate of node
    # j: column coordinate of node
    # n: lattice size
    
    # Write neighbors coordinates, and then get the rest of the division by the network size, to account
    # for the periodicity of the lattice
    return np.array([[i-1,j-1],
                     [i-1,j],
                     [i-1,j+1],
                     [i,j-1],
                     [i,j+1],
                     [i+1,j-1],
                     [i+1,j],
                     [i+1,j+1],])%n


# Generate fitness matrix
@njit
def gen_fit_mat(strat_mat, payoff_mat):
    # strat_mat: strategy matrix: 0s for cooperators and 1s for defectors
    # payoff_mat: payoff matrix
    
    n = strat_mat.shape[0] # Lattice size
    
    fit_mat = np.zeros((n,n)) # Create empty fitness matrix
    
    # Loop over all lattice nodes
    for i in range(n):
        for j in range(n):
            nb = neighbors(i,j,n) # Get neighbors' coordinates
            
            # Loop over neighbors
            for a, b in nb:
                # Sum contribution of payoff matrix for each different neighbor
                fit_mat[i,j] += payoff_mat[strat_mat[i,j],strat_mat[a,b]]
    
    return fit_mat

# One step time evolution of the strategy matrix
@njit
def evolve_strat_mat(strat_mat, fit_mat):
    # strat_mat: strategy matrix
    # fit_mat: fitness matrix
    
    n = strat_mat.shape[0] # Lattice size

    strat_mat_new = np.copy(strat_mat) # New strategy matrix, copying the past one
    
    # Loop over all lattice nodes
    for i in range(n):
        for j in range(n):
            nb = neighbors(i, j, n) # Get neighbors' coordinates
            idx_max_fit = [i,j] # Variable to store the coordinates of the neighbor with highest fitness
            
            # Loop over neighbors
            for nb_coords in nb:
                a, b = nb_coords # Separate neighbor coordinates
                
                # Check if actual neighbor has higher fitness then the highest registrated yet
                if fit_mat[a,b] > fit_mat[idx_max_fit[0], idx_max_fit[1]]:
                    idx_max_fit = [a,b]
            
            # Set new node strategy as the same of its neighbor with the highest fitness
            strat_mat_new[i,j] = strat_mat[idx_max_fit[0], idx_max_fit[1]]
    
    return strat_mat_new


# Plot strategy matrix (notice it doesn't include plt.show, so you can chose to show or save the figure)
def plot_strat_mat(strat_mat):
    # strat_mat: strategy matrix
    
    cmap = colors.ListedColormap(["blue", "red"]) # Define colormap for colorbar
    norm = colors.BoundaryNorm([0,0.5,1], cmap.N) # Define divisions for the colorbar
    
    # Plot matrix by pixels
    plt.imshow(strat_mat, cmap=cmap, norm=norm, interpolation="none")
    
    # Generate colorbar
    clb = plt.colorbar(ticks=[0.25, 0.75]) # Create colorbar and chose ticks' positions
    clb.ax.set_yticklabels(["Cooperators", "Defectors"], rotation=-30) # Set ticks' labels
    
    # Turn off x and y axis
    plt.axis("off")
    
# Plot strategy matrix and print the fitness over each pixel
def plot_strat_mat_w_fit(strat_mat, fit_mat):
    # strat_mat: strategy matrix
    # fit_mat: fitness matrix
    
    cmap = colors.ListedColormap(["blue", "red"]) # Define colormap for plotting
    norm = colors.BoundaryNorm([0,0.5,1], cmap.N) # Define divisions for colorbar
    
    # Plot matrix
    plt.imshow(strat_mat, cmap=cmap, norm=norm, interpolation="none")
    
    # Generate colorbar
    clb = plt.colorbar(ticks=[0.25, 0.75]) # Create colorbar and choose ticks' positions
    clb.ax.set_yticklabels(["Cooperators", "Defectors"], rotation=-30) # Set ticks' labels
    
    # Turn off x and y axis
    plt.axis("off")
    
    n = strat_mat.shape[0] # Lattice size
    
    # Loop over nodes to write fitness on the image
    for i in range(n):
        for j in range(n):
            plt.text(j, i, int(fit_mat[i,j]), va="center", ha="center") # Write the fitness of the node

# Generate new matrix taking into account the comparison of the actual and the past one,
# so one can know with nodes changed behavior or not, for more informative coloring
@njit
def gen_comp_mat(strat_mat, strat_mat_new):
    # strat_mat: past strategy matrix
    # start_mat_new: latest strategy matrix
    
    n = strat_mat.shape[0] # Lattice size
    
    # New matrix to which we'll add new values after comparing the matrices from consecutive timesteps
    comp_strat_mat = np.copy(strat_mat_new).astype(np.float64)
    # Notice we're copying the latest strategy matrix, so that we don't have to change anything for 
    # nodes that haven't change from the last timestep
    
    # Loop over all nodes
    for i in range(n):
        for j in range(n):
            
            # Look for nodes that used to be cooperators and became defectors
            if strat_mat[i,j] == 0 and strat_mat_new[i,j] == 1:
                comp_strat_mat[i,j] = 0.625
                # The value 0.625 is used considering where I want the color related to this be in the colorbar
            
            # Look for nodes that used to be defectors and became cooperators
            elif strat_mat[i,j] == 1 and strat_mat_new[i,j] == 0:
                comp_strat_mat[i,j] = 0.375
                # The value 0.375 was also chosen thinking of the colorbar for the nodes
    
    return comp_strat_mat

# Plot strategy matrix with more colors, specifying nodes that changed strategy or not
def plot_strat_mat_w_past(strat_mat, strat_mat_new):
    # strat_mat: past strategy matrix
    # strat_mat_new: latest strategy matrix
    
    cmap = colors.ListedColormap(["blue", "green", "yellow", "red"]) # Define colormap for plotting
    norm = colors.BoundaryNorm([0, 0.25, 0.5, 0.75, 1.0], cmap.N) #  # Define divisions for colorbar
    
    # Generate matrix differentiating nodes that recently changed strategy
    strat_mat_plot = gen_comp_mat(strat_mat, strat_mat_new)
    
    # Plot matrix
    plt.imshow(strat_mat_plot, cmap=cmap, norm=norm, interpolation="none", aspect="equal")
    
    # Generate colorbars
    clb = plt.colorbar(ticks=[0.125, 0.375, 0.625, 0.875]) # Create colorbar and choose ticks' positions
    clb.ax.set_yticklabels(["Cont. Coop.", "New Coop.", 
                            "New Defec.", "Cont. Defec."], rotation=-30) # Set ticks' labels
    
    # Turn off x and y axis
    plt.axis("off")

##############################################################################

nt = 201 # Number of timesteps
n = 100 # Lattice size

# Payoff matrix:
#        C    D
#    C   1    0
#    D   b    eps

eps = 0.

b = np.linspace(1.5, 2, 6) # Different b values
timesteps = [i for i in range(nt)] # Timesteps to save the cooperator frequency

# Array to store cooperator frequency for different b values and different timesteps
coop_freq = np.zeros((len(b), len(timesteps)))

# Loop over b values
for j in tqdm(range(len(b))):
    
    payoff_mat = np.array([[1., 0],[b[j], eps]]) # Define the payoff matrix
    
    ####################### Possible initial conditions ######################
    
    # -------------- Cooperator cluster initial condition --------------------
    strat_mat = np.ones((n,n), dtype=np.int) # # Generate initial matrix with all defectors
    strat_mat[n//2-10:n//2+10,n//2-10:n//2+10] = 0 # Make a cluster of cooperators in the middle
    # ------------------------------------------------------------------------
    
    # ------------------- Random initial condition ---------------------------
    # np.random.seed(13) # Set random number generator seed to fix initial condition
    # strat_mat = gen_strat_mat(n, coop=0.9) # Generate initial matrix: random matrix
    # ------------------------------------------------------------------------
    
    ##########################################################################
    
    # Save cooperator frequency for the initial condition
    coop_freq[:,0] = 1 - np.sum(strat_mat)/n**2
    
    # Time evolution = Loop over timesteps
    for i in range(1, nt):
        fit_mat = gen_fit_mat(strat_mat, payoff_mat) # Generate fitness matrix
        strat_mat = evolve_strat_mat(strat_mat, fit_mat) # Evolve strategy matrix
        
        # Save the cooperator frequency for the desired timestes
        if i in timesteps:
            coop_freq[j,timesteps.index(i)] = 1 - np.sum(strat_mat)/n**2

# Plot cooperator frequency time evolution for different b values
for i in range(len(b)):
    plt.plot(timesteps, coop_freq[i], label=f"$b = {b[i]}$") # Plot cooperator frequency over time
    plt.legend(loc=(1.01, 0.5)) # Add legend
    plt.xlabel("$b$")
    plt.ylabel("Cooperator Frequency")
    # plt.yscale("log") # Set scale of y axis