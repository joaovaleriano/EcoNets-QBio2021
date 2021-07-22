#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:49:00 2021

@author: joao-valeriano
"""


# Import packages
import numpy as np # Dealing with arrays
import matplotlib.pyplot as plt # Plotting
import networkx as nx # Working with networks
from tqdm import tqdm


# Generate random circular network of cooperators (0) and defectors (1)
def gen_net(number_of_nodes, number_of_edges, coop_freq, seed):
    # number_of_nodes: number of nodes in the network
    # number_of_edges: minimum degree
    # coop_freq: cooperator frequency
    
    # Create circular network
    network = nx.barabasi_albert_graph(number_of_nodes, number_of_edges, seed)
    
    # Array to store colormap indicating cooperators (blue) and defectors (red)
    colormap = []
    
    # Generate array with strategies for each node, randomly sorted
    np.random.seed(0)
    strat = np.random.choice([0,1], number_of_nodes, p=[coop_freq, 1-coop_freq])
    
    # Loop over nodes
    for i in range(network.number_of_nodes()):
        
        if strat[i] == 0: # Set node as a cooperator
            network.nodes[i]["strat"] = 0
            colormap.append("blue")
        
        else: # Set node as a defector
            network.nodes[i]["strat"] = 1
            colormap.append("yellow")
            
        # Initialize the fitness of each node
        network.nodes[i]["fit"] = 0
        
        # Set node positions
        network.nodes[i]["pos"] = (np.cos(i*2*np.pi/number_of_nodes), 
                                      np.sin(i*2*np.pi/number_of_nodes))
            
    return network, colormap


def gen_net_control_defec(number_of_nodes, number_of_edges, frac="bottom", 
                          perc=0.1, seed=None):
    # number_of_nodes: number of nodes in the network
    # number_of_edges: minimum degree
    # coop_freq: cooperator frequency
    # frac: which fraction to set as defectors with highest ("top") or lowest ("bottom") degree
    # perc: percentage of the given fraction to set as defector
    
    # Create circular network
    network = nx.barabasi_albert_graph(number_of_nodes, number_of_edges, seed)
    
    node_pos, node_deg = np.array(network.degree).T
    
    if frac=="bottom":
        node_pos_defec = node_pos[node_deg <= np.percentile(node_deg, 100*perc)]
        
    else:
        node_pos_defec = node_pos[node_deg >= np.percentile(node_deg, 100*(1-perc))]
    
    print(node_pos_defec)
    print(node_deg[node_deg >= np.percentile(node_deg, 100*(1-perc))])
    # Array to store colormap indicating cooperators (blue) and defectors (red)
    colormap = []

    # Loop over nodes
    for i in range(network.number_of_nodes()):
        
        if i in node_pos_defec: # Set node as a cooperator
            network.nodes[i]["strat"] = 1
            colormap.append("yellow")
        
        else: # Set node as a defector
            network.nodes[i]["strat"] = 0
            colormap.append("blue")
            
        # Initialize the fitness of each node
        network.nodes[i]["fit"] = 0
        
        # Set node positions
        network.nodes[i]["pos"] = (np.cos(i*2*np.pi/number_of_nodes), 
                                      np.sin(i*2*np.pi/number_of_nodes))
            
    return network, colormap

    
# Calculate fitness matrix over the network
def calc_fit_mat(network, payoff_mat):
    # network: the network object from the NetworkX package
    # payoff_mat: payoff matrix for the game
    
    # Loop over nodes
    for i in range(network.number_of_nodes()):
        
        network.nodes[i]["fit"] = 0 # Set fitness to zero initially
        
        # Sum the contribution of each neighbor for the fitness of the focal node
        for nb in network.neighbors(i):
            network.nodes[i]["fit"] += payoff_mat[network.nodes[i]["strat"],
                                                    network.nodes[nb]["strat"]]
            
    # No need to return anything, we're just editing the "fit" attribute of the network


# Evolution of the network by a time step
def evolve_strats(network, colormap, payoff_mat):
    # network: the network object from the NetworkX package
    # colormap: colors of the nodes indicating cooperators and defectors
    # payoff_mat: payoff matrix for the game
    
    past_network = nx.Graph.copy(network)
    past_colormap = [i for i in colormap]
    
    # Colors of nodes for cooperators (blue) and defectors (red)
    colors = ["blue", "yellow"]
    
    # Loop over nodes
    for i in range(past_network.number_of_nodes()):
        
        # Initialize lists to save fitnesses of cooperator and defector neighbors        
        coop_nb_fit = [-1]
        defec_nb_fit = [-1]
        
        # Check if focal node is cooperator or defector and add its fitness to
        # the corresponding list
        if past_network.nodes[i]["strat"] == 0:
            coop_nb_fit.append(past_network.nodes[i]["fit"])
        else:
            defec_nb_fit.append(past_network.nodes[i]["fit"])
        
        # Loop over neighbors, adding their fitnesses to the appropriate lists
        for nb in past_network.neighbors(i):
            if past_network.nodes[nb]["strat"] == 0:
                coop_nb_fit.append(past_network.nodes[nb]["fit"])
            else:
                defec_nb_fit.append(past_network.nodes[nb]["fit"])
                
        # Check if cooperators or defectors neighbors have higher fitness and
        # update the focal node's strategy
        if max(coop_nb_fit) > max(defec_nb_fit):
            network.nodes[i]["strat"] = 0
            colormap[i] = colors[0]
        
        elif max(coop_nb_fit) < max(defec_nb_fit):
            network.nodes[i]["strat"] = 1
            colormap[i] = colors[1]
        
        # In case of a fitness tie between cooperators and defectors, sort the
        # new strategy for the focal node
        else:
            sort_strat = np.random.choice([0,1])
            network.nodes[i]["strat"] = sort_strat
            colormap[i] = colors[sort_strat]
            
        # Change the strategy of the focal node to the same of its fittest neighbor

    # Calculate the new fitness matrix
    calc_fit_mat(network, payoff_mat)


##############################################################################

# Show the time evolution of a network
def show_time_evol(n_nodes, init_coop_freq, number_of_edges, nt, b, eps, 
                   frac, perc, seed=None):
    # n: number of nodes in the network
    # init_coop_freq: cooperator frequency in the initial condition
    # n_edges: minimum degree
    # nt: number of timesteps to run time evolution
    # b: b parameter of payoff matrix
    # eps: eps parameter of payoff matrix
    # frac: which fraction to set as defectors with highest ("top") or lowest ("bottom") degree
    # perc: percentage of the given fraction to set as defector
    # seed: seed for random number generation

    # Payoff matrix
    payoff_mat = np.array([[1, 0],[b, eps]])
    
    # Initialize network and calculate the fitness of its nodes
    # network, colormap = gen_net(n_nodes, number_of_edges, init_coop_freq, seed)
    network, colormap = gen_net_control_defec(n_nodes, number_of_edges, 
                                frac, perc, seed)
    calc_fit_mat(network, payoff_mat)
    
    # Get node positions
    node_pos = nx.get_node_attributes(network, "pos")
    
    # Draw the initial network
    nx.draw(network, node_size=50, node_color=colormap)#, with_labels=True)
    # plt.savefig(f"barabasi_albert_movie/barabasi_albert{0:04d}.png", dpi=300)
    plt.show()
    
    # Time evolution of the network
    for i in range(1, nt):
        evolve_strats(network, colormap, payoff_mat) # Evolve the network by a timestep
        
        # Plot the network
        nx.draw(network, node_size=50, node_color=colormap)#, with_labels=True)
        plt.title(f"{i}")
        # plt.savefig(f"barabasi_albert_movie/barabasi_albert{i:04d}.png", dpi=300)
        plt.show()

show_time_evol(n_nodes=100, init_coop_freq=0.9, number_of_edges=1, 
                nt=100, b=1.5, eps=0, frac="bottom", perc=0.1, seed=None)

##############################################################################

# Generate Cooperator Frequency curves for different b values
def gen_coop_freq_evol(n_nodes, nt, b, eps, seeds, init_coop_freq, number_of_edges):
    # n: number of nodes in the network
    # nt: number of timesteps to run time evolution
    # b: b parameter of payoff matrix
    # eps: eps parameter of payoff matrix
    # seed: seed for random number generation    
    # init_coop_freq: cooperator frequency in the initial condition
    # number_of_edges: minimum degree
    

    # Array to store cooperator frequencies for all timesteps and b values
    coop_freqs = np.zeros((nt, len(b), len(seeds)))
    
    # Loop over b values
    for j in tqdm(range(len(b))):
        
        # Loop over different seeds
        for k in range(len(seeds)):
            
            payoff_mat = np.array([[1., 0],[b[j], eps]]) # Define the payoff matrix
            
            # Set random number generator seed
            np.random.seed(seeds[k])
            
            # Initialize network and calculate its fitness matrix
            network, colormap = gen_net(n_nodes, number_of_edges, init_coop_freq, seeds[k])
            calc_fit_mat(network, payoff_mat)
            
            coop_freqs[0,j] = 1 - sum(nx.get_node_attributes(network, "strat").values()) / n_nodes
            
            # Time evolution of the network
            for i in range(1, nt):
                
                evolve_strats(network, colormap, payoff_mat) # Evolve the network by a timestep
                
                coop_freqs[i,j,k] = 1 - sum(nx.get_node_attributes(network, "strat").values()) / n_nodes
    
    return coop_freqs


def plot_coop_freq_evol(coop_freqs, b, title=None, save_files=False):
    
    # Array with timesteps
    timesteps = np.linspace(1, coop_freqs.shape[0], coop_freqs.shape[0])
    
    avg_coop_freqs = np.mean(coop_freqs, axis=2) # Average cooperator frequencies
    std_coop_freqs = np.std(coop_freqs, axis=2) # Standard deviation of cooperator frequencies
    
    # Set colors for plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(b)))
    
    # Plot cooperator frequency time evolution for different b values
    plt.figure(figsize=(10,7))
    for i in range(len(b)):
        plt.plot(timesteps, avg_coop_freqs[:,i], color=colors[i], lw=3,
                 label=f"$b/c = {1/(b[i]-1):0.2f}$", alpha=1.) # Plot cooperator frequency over time
        plt.fill_between(timesteps, avg_coop_freqs[:,i]-std_coop_freqs[:,i], 
                         avg_coop_freqs[:,i]+std_coop_freqs[:,i], color=colors[i], alpha=0.3)
        plt.legend(loc=(1.01, 0.1), fontsize=16) # Add legend
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlim(0, coop_freqs.shape[0])
        plt.ylim(0, 1)
        plt.xlabel("Time", fontsize=24)
        plt.ylabel("Cooperator Frequency", fontsize=24)
        plt.ylim(0, 1)
        plt.title(title, fontsize=28)
        
    if save_files:
        plt.savefig(f"barabasi_albert_coop_freq_evol_error.pdf",
                    bbox_inches="tight")
    
    else:    
        plt.show()
    
    return coop_freqs
        

# Plot evolution of cooperator frequencies for different b values
# b = [1.1, 1.2, 1.3, 1.4, 1.5]
# seeds = [i for i in range(10)]
# coop_freqs = gen_coop_freq_evol(n_nodes=100, nt=100, b=b, eps=0., seeds=seeds,
#                                 init_coop_freq=0.9, number_of_edges=12)
# plot_coop_freq_evol(coop_freqs, b, title="Minimum $k$ = 12", save_files=False)

##############################################################################

# Generate final cooperator frequency for different b values
def gen_final_coop_freq(n_nodes, number_of_edges, nt, nt_save, b, eps=0., init_coop_freq=0.5, seed=None):
    # n: lattice side -> number of sites = n^2
    # number_of_edges: minimum degree
    # nt: number of timesteps to evolve before annotating results
    # nt_save: number of timesteps to annotate results for calculating statistics
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
        

        network, colormap = gen_net(n_nodes, number_of_edges, init_coop_freq, seed)
        calc_fit_mat(network, payoff_mat)
        
        # Time evolution = Loop over timesteps
        for i in range(1, nt):
            evolve_strats(network, colormap, payoff_mat) # Evolve the network by a timestep
            
            print(f"\rb: {j+1}/{len(b)}; time: {i+1}/{nt}", end="")
        
        for i in range(nt_save):
            evolve_strats(network, colormap, payoff_mat) # Evolve the network by a timestep
            
            # Save the cooperator frequency for the desired timestesps
            coop_freq[j,i] = 1 - sum(nx.get_node_attributes(network, "strat").values()) / n_nodes
            
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
        plt.savefig("barabasi_albert_final_coop_freq_vs_b.pdf", bbox_inches="tight")
        plt.close()
        
    else:
        plt.show()

# b = np.linspace(1, 2, 11)
# coop_freq = gen_final_coop_freq(n_nodes=100, number_of_edges=100, nt=80, nt_save=20, 
#                                 b=b, eps=0., init_coop_freq=0.9, seed=0)

# plot_final_coop_freq(coop_freq, b, save_files=False)


##############################################################################

# Generate final cooperator frequency for different b values
def gen_final_coop_freq(n_nodes, number_of_edges, nt, nt_save, b, eps=0., init_coop_freq=0.5, seeds=[None]):
    # n: lattice side -> number of sites = n^2
    # number_of_edges: minimum degree
    # nt: number of timesteps to evolve before annotating results
    # nt_save: number of timesteps to annotate results for calculating statistics
    # b: array of values for b parameter value for the payoff matrix
    # eps: eps parameter value for the payoff matrix
    # init_coop_freq: frequency of cooperators on initial condition
    # init_cond: initial condition of the lattice
    # save_files: wether to save plots to files or not
    # seed: random number generator seed
    
    
    # Array to store cooperator frequency for different b values and different timesteps
    coop_freq = np.zeros((len(b), len(number_of_edges), len(seeds), nt_save))
    
    # Loop over b values
    for j in range(len(b)):
        for k in range(len(number_of_edges)):
            for s in range(len(seeds)):
                payoff_mat = np.array([[1., 0],[b[j], eps]]) # Define the payoff matrix
                
                network, colormap = gen_net(n_nodes, number_of_edges[k], init_coop_freq, seeds[s])
                calc_fit_mat(network, payoff_mat)
                
                # Time evolution = Loop over timesteps
                for i in range(1, nt):
                    evolve_strats(network, colormap, payoff_mat) # Evolve the network by a timestep
                    
                    print(f"\rb: {j+1}/{len(b)}; time: {i+1}/{nt}", end="")
                
                for i in range(nt_save):
                    evolve_strats(network, colormap, payoff_mat) # Evolve the network by a timestep
                    
                    # Save the cooperator frequency for the desired timestesps
                    coop_freq[j,k,s,i] = 1 - sum(nx.get_node_attributes(network, "strat").values()) / n_nodes
                    
                    print(f"\rb: {j+1}/{len(b)}; time: {i+1}/{nt_save}", end="")
    
    return coop_freq

# Plot statistics of final cooperator frequency for different b values
def plot_final_coop_freq(coop_freq, b, number_of_edges, save_files=False):
    # coop_freq: array containing some timesteps of the cooperator frequency for different values of b
    #        |-> shape: (len(b), # of timesteps)
    # b: array of b values considered for generating "coop_freq"
    # save_files: wether or not to save plot to file
    
    avg_coop_freq = np.mean(coop_freq, axis=-1) # Average final cooperator frequencies
    
    final_coop_freq_avg = np.mean(avg_coop_freq, axis=-1) # Average final cooperator frequencies
    final_coop_freq_min = np.min(avg_coop_freq, axis=-1) # Minimum final cooperator frequencies
    final_coop_freq_max = np.max(avg_coop_freq, axis=-1) # Maximum final cooperator frequencies
    
    # Generate errorbars from minimum to maximum cooperator frequencies
    errorbars = np.zeros((2, len(b), len(number_of_edges)))
    for j in range(len(number_of_edges)):
        for i in range(len(b)):
            errorbars[:,i,j] = [final_coop_freq_avg[i,j]-final_coop_freq_min[i,j],
                            final_coop_freq_max[i,j]-final_coop_freq_avg[i,j]]
    
    # Set colors for plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(b)))
    
    # Plot final cooperator frequency for different b values
    plt.figure(figsize=(10,7))
    for j in range(len(b)):
        # Plot markers with errorbars
        plt.errorbar(number_of_edges, final_coop_freq_avg[j,:], errorbars[:,j,:], 
                      color=colors[j], marker="o", markersize=10, capsize=5,
                      label=f"$b/c = {1/(b[j]-1):0.2f}$")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("Minimum $k$", fontsize=24)
        plt.ylabel("Final Cooperator Frequency", fontsize=24)
    plt.legend(loc=(1.01, 0.5), fontsize=16)
    
    # Save plot to file or show it
    if save_files:
        plt.savefig("barabasi_albert_final_coop_freq_vs_b.pdf", bbox_inches="tight")
        plt.close()
        
    else:
        plt.show()

# b = np.linspace(1.1, 2, 6)
# number_of_edges = np.arange(1, 10, 1)
# coop_freq = gen_final_coop_freq(n_nodes=100, number_of_edges=number_of_edges, nt=80, nt_save=20, 
#                                 b=b, eps=0., init_coop_freq=0.9, seeds=[i for i in range(100)])

# plot_final_coop_freq(coop_freq, b, number_of_edges, save_files=False)
