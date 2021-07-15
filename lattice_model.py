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
    
    # Write neighbors coordinates, and then get the rest of the division by the network size, 
    # to account for the periodicity of the lattice
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
    
    # Initialize lists to save fitnesses of cooperator and defector neighbors
    for i in range(n):
        for j in range(n):
            
            coop_nb_fit = [-1]
            defec_nb_fit = [-1]
            
            # Check if focal node is cooperator or defector and add its fitness to
            # the corresponding list
            if strat_mat[i,j] == 0:
                coop_nb_fit.append(fit_mat[i,j])
            else:
                defec_nb_fit.append(fit_mat[i,j])
            
            # Loop over neighbors, adding their fitnesses to the appropriate lists
            for nb in neighbors(i, j, n):
                a, b = nb
                if strat_mat[a,b] == 0:
                    coop_nb_fit.append(fit_mat[a,b])
                else:
                    defec_nb_fit.append(fit_mat[a,b])
                    
            # Check if cooperators or defectors neighbors have higher fitness and
            # update the focal node's strategy
            if max(coop_nb_fit) > max(defec_nb_fit):
                strat_mat_new[i,j] = 0
            
            elif max(coop_nb_fit) < max(defec_nb_fit):
                strat_mat_new[i,j] = 1
            
            # In case of a fitness tie between cooperators and defectors, sort the
            # new strategy for the focal node
            else:
                sort_strat = np.random.randint(0,2)
                strat_mat_new[i,j] = sort_strat
    
    return strat_mat_new


# Plot strategy matrix (notice it doesn't include plt.show, so you can chose to show or save the figure)
def plot_strat_mat(strat_mat):
    # strat_mat: strategy matrix
    
    cmap = colors.ListedColormap(["blue", "yellow"]) # Define colormap for colorbar
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
    
    cmap = colors.ListedColormap(["blue", "yellow"]) # Define colormap for plotting
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

# Generate time evolution cooperator frequency for different b values and
# multiple random number generator seeds
def gen_coop_freq_evol(n, nt, b, eps=0., init_coop_freq=0.5, init_cond="random", 
                       seeds=[None]):
    # n: lattice side -> number of sites = n^2
    # nt: number of timesteps to evolve
    # b: array of values for b parameter value for the payoff matrix
    # eps: eps parameter value for the payoff matrix
    # init_coop_freq: frequency of cooperators on initial condition
    # init_cond: initial condition of the lattice
    # seeds: random number seeds to generate different initial conditions for each b value


    timesteps = [i for i in range(nt)] # Timesteps to save the cooperator frequency
    
    # Array to store cooperator frequency for different b values and different timesteps
    coop_freq = np.zeros((len(b), len(seeds), len(timesteps)))
    
    # Loop over b values
    for j in range(len(b)):
        
        payoff_mat = np.array([[1., 0],[b[j], eps]]) # Define the payoff matrix
        
        for k in range(len(seeds)):
        
            #################### Possible initial conditions #################

            # Random initial condition
            if init_cond == "random":
                np.random.seed(seeds[k]) # Set random number generator seed to fix initial condition
                strat_mat = gen_strat_mat(n, coop=init_coop_freq) # Generate initial matrix: random matrix
            
            # Cooperator cluster initial condition
            if init_cond == "cluster":
                strat_mat = np.ones((n,n), dtype=np.int) # Generate initial matrix with all defectors
                cs = int(np.sqrt(init_coop_freq)*n/2) # Half cooperator cluster side
                strat_mat[n//2-cs:n//2+cs,n//2-cs:n//2+cs] = 0 # Make a cluster of cooperators in the middle
                
            ##################################################################
            
            # Save cooperator frequency for the initial condition
            coop_freq[j,k,0] = 1 - np.sum(strat_mat)/n**2
            
            # Time evolution = Loop over timesteps
            for i in range(1, nt):
                fit_mat = gen_fit_mat(strat_mat, payoff_mat) # Generate fitness matrix
                strat_mat = evolve_strat_mat(strat_mat, fit_mat) # Evolve strategy matrix
                
                # Save the cooperator frequency for the desired timestesps
                coop_freq[j,k,i] = 1 - np.sum(strat_mat)/n**2
            
            print(f"\rb: {j+1}/{len(b)}; seed: {k+1}/{len(seeds)}", end="")
                
    return coop_freq
    

# Plot the time evolution of cooperator frequency for different b values and
# error margins indicating multiple random number generator seeds
def plot_coop_freq_evol(coop_freq, b, save_files=False, same_graph=False):
    # coop_freq: array with cooperator frequencies for different values of b, for different seeds, through time
    #        |-> shape: (len(b), # of seeds, # of timesteps)
    # b: array with different b values considered
    # save_files: wheter to save plots to files or not
    # same_graph: wether to put all plots together in a single graph
    
    
    timesteps = [i for i in range(coop_freq.shape[2])] # Timesteps for plotting
    
    avg_coop_freq = np.mean(coop_freq, axis=1)
    min_coop_freq = np.min(coop_freq, axis=1)
    max_coop_freq = np.max(coop_freq, axis=1)
    
    # If one wishes to save all curves to the same graph
    if same_graph:
        # Plot cooperator frequency time evolution for different b values
        plt.figure(figsize=(10,7))
        for j in range(coop_freq.shape[0]):        
            plt.plot(timesteps, avg_coop_freq[j], lw=3, label=f"$b={b[j]:.2f}$")
            plt.fill_between(timesteps, min_coop_freq[j], max_coop_freq[j], alpha=0.3)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlim(0, timesteps[-1])
        plt.ylim(0, 1)
        plt.xlabel("Time", fontsize=24)
        plt.ylabel("Cooperator Frequency", fontsize=24)
        plt.legend(loc=(1.01, 0.5), fontsize=16)
        
        # Save plot to file or show it
        if save_files:
            plt.savefig(f"lattice_100x100_coop_freq_vs_time_variation_b={b[0]:.2f}-{b[-1]:.2f}.pdf", 
                        bbox_inches="tight")
            plt.close()
        
        else:
            plt.show()
        
    # If one wishes to save separate graphs
    else:
        # Plot cooperator frequency time evolution for different b values
        for j in range(coop_freq.shape[0]):        
            plt.figure(figsize=(10,7))
            plt.plot(timesteps, avg_coop_freq[j], lw=3)
            plt.fill_between(timesteps, min_coop_freq[j], max_coop_freq[j], alpha=0.3)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlim(0, timesteps[-1])
            plt.ylim(0, 1)
            plt.xlabel("Time", fontsize=24)
            plt.ylabel("Cooperator Frequency", fontsize=24)
            plt.title(f"$b = {b[j]:.2f}$", fontsize=26)
                
            # Save plot to file or show it
            if save_files:
                plt.savefig(f"lattice_100x100_coop_freq_vs_time_variation_b={b[j]:.2f}.pdf", 
                            bbox_inches="tight")
                plt.close()
            
            else:
                plt.show()


##############################################################################

# Generate final cooperator frequency for different b values
def gen_final_coop_freq(n, nt, nt_save, b, eps=0., init_coop_freq=0.5, 
                         init_cond="random", save_files=False, seed=None):
    # n: lattice side -> number of sites = n^2
    # nt: number of timesteps to evolve before annotating results
    # nt: number of timesteps to annotate results for calculating statistics
    # b: array of values for b parameter value for the payoff matrix
    # eps: eps parameter value for the payoff matrix
    # init_coop_freq: frequency of cooperators on initial condition
    # init_cond: initial condition of the lattice
    # save_files: wether to save plots to files or not
    # seed: random number generator seed
    
    
    # Array to store cooperator frequency for different b values and different timesteps
    coop_freq = np.zeros((len(b), nt_save))
    
    # Loop over b values
    for j in range(len(b)):
        
        payoff_mat = np.array([[1., 0],[b[j], eps]]) # Define the payoff matrix
        
        ####################### Possible initial conditions ##################

        # Random initial condition
        if init_cond == "random":
            np.random.seed(seed) # Set random number generator seed to fix initial condition
            strat_mat = gen_strat_mat(n, coop=init_coop_freq) # Generate initial matrix: random matrix
        
        # Cooperator cluster initial condition
        if init_cond == "cluster":
            strat_mat = np.ones((n,n), dtype=np.int) # Generate initial matrix with all defectors
            cs = int(np.sqrt(init_coop_freq)*n/2) # Half cooperator cluster side
            strat_mat[n//2-cs:n//2+cs,n//2-cs:n//2+cs] = 0 # Make a cluster of cooperators in the middle
            
        ######################################################################
        
        # Time evolution = Loop over timesteps
        for i in range(1, nt):
            fit_mat = gen_fit_mat(strat_mat, payoff_mat) # Generate fitness matrix
            strat_mat = evolve_strat_mat(strat_mat, fit_mat) # Evolve strategy matrix
            
            print(f"\rb: {j+1}/{len(b)}; time: {i+1}/{nt}", end="")
        
        for i in range(nt_save):
            fit_mat = gen_fit_mat(strat_mat, payoff_mat) # Generate fitness matrix
            strat_mat = evolve_strat_mat(strat_mat, fit_mat) # Evolve strategy matrix
            
            # Save the cooperator frequency for the desired timestesps
            coop_freq[j,i] = 1 - np.sum(strat_mat)/n**2
            
            print(f"\rb: {j+1}/{len(b)}; time: {i+1}/{nt_save}", end="")
    
    return coop_freq

# Plot statistics of final cooperator frequency for different b values
def plot_final_coop_freq(coop_freq, b, save_files=False):
    # coop_freq: array containing some timesteps of the cooperator frequency for different values of b
    #        |-> shape: (len(b), # of timesteps)
    # b: array of b values considered for generating "coop_freq"
    # save_files: wether or not to save plot to file
    
    final_coop_freq_avg = np.mean(coop_freq, axis=1) # Average final cooperator frequencies
    final_coop_freq_min = np.min(coop_freq, axis=1) # Minimum final cooperator frequencies
    final_coop_freq_max = np.max(coop_freq, axis=1) # Maximum final cooperator frequencies
    
    # Generate errorbars from minimum to maximum cooperator frequencies
    errorbars = np.zeros((2, len(b)))
    for i in range(len(b)):
        errorbars[:,i] = [final_coop_freq_avg[i]-final_coop_freq_min[i],
                        final_coop_freq_max[i]-final_coop_freq_avg[i]]
    
    # Set colors for plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(b)))
    
    # Plot final cooperator frequency for different b values
    plt.figure(figsize=(10,7))
    for i in range(len(b)):
        # Plot markers with errorbars
        plt.errorbar(b[i], final_coop_freq_avg[i], errorbars[:,i:i+1], 
                     color=colors[i], marker="o", markersize=10, capsize=5,
                     label=f"$b = {b[i]:0.2f}$")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("$b$", fontsize=24)
        plt.ylabel("Final Cooperator Frequency", fontsize=24)
    
    # Save plot to file or show it
    if save_files:
        plt.savefig("lattice_100x100_final_coop_freq_vs_b.pdf", bbox_inches="tight")
        plt.close()
        
    else:
        plt.show()


##############################################################################

# Plot the lattice's time evolution
def plot_lat_evol(n, nt, b, eps=0., init_coop_freq=0.5, init_cond="random", save_files=False, seed=None):
    # n: lattice side -> number of sites = n^2
    # nt: number of timesteps to consider evolution
    # b: b parameter value for the payoff matrix
    # eps: eps parameter value for the payoff matrix
    # init_coop_freq: frequency of cooperators on initial condition
    # init_cond: initial condition of the lattice
    # save_files: wether to save plots to files or not
    # seed: random number generator seed
    
    
    # Check if folder for saving figures exists, and if not, create it
    if "lattice_model_movie" not in os.listdir():
        
        os.mkdir("lattice_model_movie") # Create folder
    
    
    # Define the payoff matrix
    payoff_mat = np.array([[1., 0],[b, eps]])
    
    ####################### Possible initial conditions ######################

    # Random initial condition
    if init_cond == "random":
        np.random.seed(seed) # Set random number generator seed to fix initial condition
        strat_mat = gen_strat_mat(n, coop=init_coop_freq) # Generate initial matrix: random matrix
    
    # Cooperator cluster initial condition
    if init_cond == "cluster":
        strat_mat = np.ones((n,n), dtype=np.int) # Generate initial matrix with all defectors
        cs = int(np.sqrt(init_coop_freq)*n/2) # Half cooperator cluster side
        strat_mat[n//2-cs:n//2+cs,n//2-cs:n//2+cs] = 0 # Make a cluster of cooperators in the middle
        
    ##########################################################################    
    
    
    # Plot initial condition
    plot_strat_mat(strat_mat)
    plt.title(f"$b = {b:.2f}$")
    
    # Save or show plot
    if save_files:
        plt.savefig(f"lattice_model_movie/lattice_model_{0:04d}", dpi=300, bbox_inches="tight")
        plt.close() 
        
    else:
        plt.show()
    
    
    # Time evolution = Loop over timesteps
    for i in tqdm(range(1, nt)):
        fit_mat = gen_fit_mat(strat_mat, payoff_mat) # Calculate fitness matrix
        strat_mat = evolve_strat_mat(strat_mat, fit_mat) # Time evolution by one timestep
        plot_strat_mat(strat_mat) # Plot lattice
        plt.title(f"$b = {b:.2f}$") # Add title 
        
        # Save or show plot
        if save_files:
            plt.savefig(f"lattice_model_movie/lattice_model_{i:04d}", dpi=300, bbox_inches="tight")
            plt.close() 
        
        else:
            plt.show()