#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 09:55:51 2021

@author: joao-valeriano
"""

import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# mat = np.array([[0,1,0,1],
#                 [1,0,1,0],
#                 [0,1,0,1],
#                 [1,0,1,0]])

# g = nx.from_numpy_matrix(mat)


def gen_full_net(n, coop_freq0, coop_freq1, d=0, seed=None):
    # number_of_nodes: number of nodes in the network
    # knn: number of nearest neighbors to connect
    # rewire: probability of rewiring a connection
    # coop_freq: cooperator frequency
    
    # Create network
    network = nx.complete_multipartite_graph(n, n)
    
    # Array to store colormap indicating cooperators (blue) and defectors (red)
    colormap = []
    
    # Generate array with strategies for each node, randomly sorted
    np.random.seed(seed)
    strat0 = np.random.choice([0,1], n, p=[coop_freq0, 1-coop_freq0])
    strat1 = np.random.choice([0,1], n, p=[coop_freq1, 1-coop_freq1])
    
    # Loop over nodes
    for i in range(n):
        
        if strat0[i] == 0: # Set node as a cooperator
            network.nodes[i]["strat"] = 0
            colormap.append("blue")
        
        else: # Set node as a defector
            network.nodes[i]["strat"] = 1
            colormap.append("yellow")
            
        # Initialize the fitness of each node
        network.nodes[i]["fit"] = 0
        
        # Set node positions
        network.nodes[i]["pos"] = (0,-i-d)
        
    for i in range(n, 2*n):
        
        if strat1[i-n] == 0: # Set node as a cooperator
            network.nodes[i]["strat"] = 0
            colormap.append("blue")
        
        else: # Set node as a defector
            network.nodes[i]["strat"] = 1
            colormap.append("yellow")
            
        # Initialize the fitness of each node
        network.nodes[i]["fit"] = 0
        
        # Set node positions
        network.nodes[i]["pos"] = (1,n-i-d)
            
    return network, colormap


def gen_random_net(n, coop_freq0, coop_freq1, n_keep, d=0, seed=None):
    # number_of_nodes: number of nodes in the network
    # knn: number of nearest neighbors to connect
    # rewire: probability of rewiring a connection
    # coop_freq: cooperator frequency
    
    # Create network
    network = nx.complete_multipartite_graph(n, n)
    
    # Array to store colormap indicating cooperators (blue) and defectors (red)
    colormap = []
    
    # Generate array with strategies for each node, randomly sorted
    np.random.seed(seed)
    strat0 = np.random.choice([0,1], n, p=[coop_freq0, 1-coop_freq0])
    strat1 = np.random.choice([0,1], n, p=[coop_freq1, 1-coop_freq1])
    
    # Loop over nodes
    for i in range(n):
        
        if strat0[i] == 0: # Set node as a cooperator
            network.nodes[i]["strat"] = 0
            colormap.append("blue")
        
        else: # Set node as a defector
            network.nodes[i]["strat"] = 1
            colormap.append("yellow")
            
        # Initialize the fitness of each node
        network.nodes[i]["fit"] = 0
        
        # Set node positions
        network.nodes[i]["pos"] = (0,-i-d)
        
    for i in range(n, 2*n):
        
        if strat1[i-n] == 0: # Set node as a cooperator
            network.nodes[i]["strat"] = 0
            colormap.append("blue")
        
        else: # Set node as a defector
            network.nodes[i]["strat"] = 1
            colormap.append("yellow")
            
        # Initialize the fitness of each node
        network.nodes[i]["fit"] = 0
        
        # Set node positions
        network.nodes[i]["pos"] = (1,n-i-d)
    
    save_edges = list(network.edges)
    np.random.shuffle(save_edges)
    
    network.remove_edges_from(list(network.edges))
    
    B_nodes = [i for i in range(n, 2*n)]
    
    for i in range(n):
        node = np.random.choice(B_nodes)
        network.add_edge(i, node)
        B_nodes.remove(node)
        save_edges.remove((i,node))
    
    network.add_edges_from(save_edges[:n_keep-n])
    
    # for i in range(n_remove):
    #     node = np.random.choice(len(network.edges))
    #     network.remove_edge(*list(network.edges)[node])
    
    return network, colormap


def gen_modular_net(cluster_size, n_cluster, coop_freq0, coop_freq1, n_keep, poiss_l, seed=None):
    
    cluster_list = []
    
    colormap = []
    
    for i in range(n_cluster):
        
        d=cluster_size*i
        
        net, cmap = gen_random_net(cluster_size, coop_freq0, coop_freq1, n_keep,
                                 d+i, seed=None)
        
        map = {}
        for j in range(cluster_size):
                map[j] = j+cluster_size*i
        for j in range(cluster_size, 2*cluster_size):
                map[j] = j+cluster_size*i+cluster_size*(n_cluster-1)
                
        net = nx.relabel.relabel_nodes(net, map)
        
        cluster_list.append(net)
        colormap += cmap

    network = nx.compose_all(cluster_list)
    
    for i in range(n_cluster-1):
        
        n_edges = max(np.random.poisson(poiss_l),1)
        edge_count = 0
        
        while edge_count < n_edges:
            edge = (np.random.randint(cluster_size*i, 
                                           cluster_size*(i+1)),
                    np.random.randint(cluster_size*(n_cluster+i+1), 
                                           cluster_size*(n_cluster+i+2)))
            
            if edge not in list(network.edges):
                network.add_edge(*edge)
                edge_count += 1

    return network, colormap

net, cmap = gen_modular_net(20, 5, 0.5, 0.5, 160, 1)
nx.draw(net, nx.get_node_attributes(net, "pos"), node_color=cmap, node_size=100)


def plot_partition(network, colormap, n, p):
    if p == 0:
        G = bipartite.projected_graph(network, [i for i in range(n)])
        nx.draw(G, node_color=colormap[:n], with_labels=True)
        plt.title("A")
        plt.show()
        
    if p == 1:
        G = bipartite.projected_graph(network, [i for i in range(n, 2*n)])
        nx.draw(G, node_color=colormap[n:n+n], with_labels=True)
        plt.title("B")
        plt.show()

# Calculate fitness matrix over the network
def calc_fit_mat(network, n, payoff_mat0, payoff_mat1):
    # network: the network object from the NetworkX package
    # payoff_mat: payoff matrix for the game
    
    # Loop over nodes
    for i in range(n):
        
        network.nodes[i]["fit"] = 0 # Set fitness to zero initially
        
        # Sum the contribution of each neighbor for the fitness of the focal node
        for nb in network.neighbors(i):
            # print(network.nodes[nb]["strat"])
            network.nodes[i]["fit"] += payoff_mat0[network.nodes[i]["strat"],
                                                    network.nodes[nb]["strat"]]
        # print("\n")
    # Loop over nodes
    for i in range(n, 2*n):
        
        network.nodes[i]["fit"] = 0 # Set fitness to zero initially
        
        # Sum the contribution of each neighbor for the fitness of the focal node
        for nb in network.neighbors(i):
            network.nodes[i]["fit"] += payoff_mat1[network.nodes[i]["strat"],
                                                    network.nodes[nb]["strat"]]
            
    # No need to return anything, we're just editing the "fit" attribute of the network


# Evolution of the network by a time step
def evolve_strats(network, colormap, n, payoff_mat0, payoff_mat1):
    # network: the network object from the NetworkX package
    # colormap: colors of the nodes indicating cooperators and defectors
    # payoff_mat: payoff matrix for the game
    
    past_network = nx.Graph.copy(network)
    past_colormap = [i for i in colormap]
    
    # Colors of nodes for cooperators (blue) and defectors (red)
    colors = ["blue", "yellow"]
    
    A = bipartite.projected_graph(network, [i for i in range(n)])
    B = bipartite.projected_graph(network, [i for i in range(n, 2*n)])
    
    coopA = 0
    coopB = 0
    
    # Loop over A nodes
    for i in range(n):
        
        # Initialize lists to save fitnesses of cooperator and defector neighbors        
        coop_nb_fit = [-1]
        defec_nb_fit = [-1]
        
        # Check if focal node is cooperator or defector and add its fitness to
        # the corresponding list
        if A.nodes[i]["strat"] == 0:
            coop_nb_fit.append(A.nodes[i]["fit"])
        else:
            defec_nb_fit.append(A.nodes[i]["fit"])
        
        # Loop over neighbors, adding their fitnesses to the appropriate lists
        for nb in A.neighbors(i):
            if A.nodes[nb]["strat"] == 0:
                coop_nb_fit.append(A.nodes[nb]["fit"])
            else:
                defec_nb_fit.append(A.nodes[nb]["fit"])
                
        # Check if cooperators or defectors neighbors have higher fitness and
        # update the focal node's strategy
        if max(coop_nb_fit) > max(defec_nb_fit):
            network.nodes[i]["strat"] = 0
            colormap[i] = colors[0]
            coopA += 1
        
        elif max(coop_nb_fit) < max(defec_nb_fit):
            network.nodes[i]["strat"] = 1
            colormap[i] = colors[1]
        
        # In case of a fitness tie between cooperators and defectors, sort the
        # new strategy for the focal node
        else:
            n_defec_tie = defec_nb_fit.count(max(defec_nb_fit))
            n_coop_tie = coop_nb_fit.count(max(coop_nb_fit))
            tie_strats = [0]*n_coop_tie + [1]*n_defec_tie
            sort_strat = np.random.choice(tie_strats)
            network.nodes[i]["strat"] = sort_strat
            colormap[i] = colors[sort_strat]
            
            if sort_strat == 0:
                coopA += 1
            
    # Loop over B nodes
    for i in range(n, 2*n):
        
        # Initialize lists to save fitnesses of cooperator and defector neighbors        
        coop_nb_fit = [-1]
        defec_nb_fit = [-1]
        
        # Check if focal node is cooperator or defector and add its fitness to
        # the corresponding list
        if B.nodes[i]["strat"] == 0:
            coop_nb_fit.append(B.nodes[i]["fit"])
        else:
            defec_nb_fit.append(B.nodes[i]["fit"])
        
        # Loop over neighbors, adding their fitnesses to the appropriate lists
        for nb in B.neighbors(i):
            if B.nodes[nb]["strat"] == 0:
                coop_nb_fit.append(B.nodes[nb]["fit"])
            else:
                defec_nb_fit.append(B.nodes[nb]["fit"])
                
        # Check if cooperators or defectors neighbors have higher fitness and
        # update the focal node's strategy
        if max(coop_nb_fit) > max(defec_nb_fit):
            network.nodes[i]["strat"] = 0
            colormap[i] = colors[0]
            coopB += 1
        
        elif max(coop_nb_fit) < max(defec_nb_fit):
            network.nodes[i]["strat"] = 1
            colormap[i] = colors[1]
        
        # In case of a fitness tie between cooperators and defectors, sort the
        # new strategy for the focal node
        else:
            n_defec_tie = defec_nb_fit.count(max(defec_nb_fit))
            n_coop_tie = coop_nb_fit.count(max(coop_nb_fit))
            tie_strats = [0]*n_coop_tie + [1]*n_defec_tie
            sort_strat = np.random.choice(tie_strats)
            network.nodes[i]["strat"] = sort_strat
            colormap[i] = colors[sort_strat]
            
            if sort_strat == 0:
                coopB += 1
    
    # Calculate the new fitness matrix
    calc_fit_mat(network, n, payoff_mat0, payoff_mat1)
    
    return coopA/n, coopB/n


# Show the time evolution of a network
def show_random_time_evol(cluster_size, n_cluster, init_coop_freq0, 
                          init_coop_freq1, n_keep, nt, b0, b1, poiss_l, seed=None):
    # n: number of nodes in the network
    # init_coop_freq: cooperator frequency in the initial condition
    # knn: number of nearest neighbors to connect
    # rewire: probability of rewiring a connection
    # nt: number of timesteps to run time evolution
    # b: b parameter of payoff matrix
    # eps: eps parameter of payoff matrix
    # seed: seed for random number generation

    # Payoff matrix
    payoff_mat0 = np.array([[1, 0],[b0, 0]])
    payoff_mat1 = np.array([[1, 0],[b1, 0]])
    
    # Initialize network and calculate the fitness of its nodes
    
    network, colormap = gen_modular_net(cluster_size, n_cluster, init_coop_freq0, 
                                        init_coop_freq1, n_keep, poiss_l, seed)
    
    n = cluster_size * n_cluster
    
    calc_fit_mat(network, n, payoff_mat0, payoff_mat1)
    
    # Get node positions
    node_pos = nx.get_node_attributes(network, "pos")
    
    # Draw the initial network
    nx.draw(network, node_pos, node_size=50, node_color=colormap)#, with_labels=True)
    # plt.savefig(f"small_world_movie/small_world{0:04d}.png", dpi=300)
    plt.show()
    
    # Time evolution of the network
    for i in range(1, nt):
        evolve_strats(network, colormap, n, payoff_mat0, payoff_mat1) # Evolve the network by a timestep
        
        # Plot the network
        nx.draw(network, node_pos, node_size=50, node_color=colormap)#, with_labels=True)
        plt.title(f"{i}")
        # plt.savefig(f"bipartite_movie/small_world{i:04d}.png", dpi=300)
        plt.show()

# show_random_time_evol(20, 5, 0.9, 0.9, 160, nt=300, b0=1.6, b1=1.6, poiss_l=10, seed=0)


##############################################################################

# Generate final cooperator frequency for different b values
def gen_final_coop_freq(cluster_size, n_cluster, n_keep, nt, nt_save, b, init_coop_freq0=0.5, 
                        init_coop_freq1=0.5, poiss_l=1, seeds=[None]):
    # n: lattice side -> number of sites = n^2
    # knn: number of nearest neighbors to connect
    # rewire: probability of rewiring a connection
    # nt: number of timesteps to evolve before annotating results
    # nt_save: number of timesteps to annotate results for calculating statistics
    # b: array of values for b parameter value for the payoff matrix
    # eps: eps parameter value for the payoff matrix
    # init_coop_freq: frequency of cooperators on initial condition
    # init_cond: initial condition of the lattice
    # save_files: wether to save plots to files or not
    # seed: random number generator seed
    
    n = cluster_size*n_cluster
    
    # Array to store cooperator frequency for different b values and different timesteps
    coop_freqs0 = np.zeros((len(b), len(poiss_l), len(seeds), nt_save))
    coop_freqs1 = np.zeros((len(b), len(poiss_l), len(seeds), nt_save))
    
    # Loop over b values
    for j in range(len(b)):
        for l in range(len(poiss_l)):
            for s in range(len(seeds)):
                
                payoff_mat = np.array([[1., 0],[b[j], 0]]) # Define the payoff matrix
                
                network, colormap = gen_modular_net(cluster_size, n_cluster, init_coop_freq0, 
                                                   init_coop_freq1, n_keep, poiss_l[l], seeds[s])
                calc_fit_mat(network, n, payoff_mat, payoff_mat)
                
                # Time evolution = Loop over timesteps
                for i in range(1, nt):
                    evolve_strats(network, colormap, n, payoff_mat, payoff_mat) # Evolve the network by a timestep
                    
                    print("\r"+"\t"*10, end="")
                    print(f"\rb: {j+1}/{len(b)}; lambda: {l+1}/{len(poiss_l)}; time: {i+1}/{nt}; seed: {s+1}/{len(seeds)}", end="")
                
                for i in range(nt_save):
                    coop_freqs0[j,l,s,i], coop_freqs1[j,l,s,i] = evolve_strats(network, colormap, n, 
                                                                         payoff_mat, payoff_mat) # Evolve the network by a timestep
                    
                    print("\r"+"\t"*10, end="")
                    print(f"\rb: {j+1}/{len(b)}; lambda: {l+1}/{len(poiss_l)}; time: {i+1}/{nt_save}; seed: {s+1}/{len(seeds)}", end="")
    
    return coop_freqs0, coop_freqs1

# Plot statistics of final cooperator frequency for different b values
def plot_final_coop_freq(coop_freqs0, coop_freqs1, b, poiss_l, save_files=False):
    # coop_freq: array containing some timesteps of the cooperator frequency for different values of b
    #        |-> shape: (len(b), # of timesteps)
    # b: array of b values considered for generating "coop_freq"
    # save_files: wether or not to save plot to file
    
    avg_coop_freqs0 = np.mean(coop_freqs0, axis=-1)
    avg_coop_freqs1 = np.mean(coop_freqs1, axis=-1)
    
    final_coop_freq_avg0 = np.mean(avg_coop_freqs0, axis=-1) # Average final cooperator frequencies
    final_coop_freq_min0 = np.min(avg_coop_freqs0, axis=-1) # Minimum final cooperator frequencies
    final_coop_freq_max0 = np.max(avg_coop_freqs0, axis=-1) # Maximum final cooperator frequencies
    
    final_coop_freq_avg1 = np.mean(avg_coop_freqs1, axis=-1) # Average final cooperator frequencies
    final_coop_freq_min1 = np.min(avg_coop_freqs1, axis=-1) # Minimum final cooperator frequencies
    final_coop_freq_max1 = np.max(avg_coop_freqs1, axis=-1) # Maximum final cooperator frequencies
    
    # Generate errorbars from minimum to maximum cooperator frequencies
    errorbars0 = np.zeros((2, len(b), len(poiss_l)))
    errorbars1 = np.zeros((2, len(b), len(poiss_l)))
    for i in range(len(b)):
        for j in range(len(poiss_l)):
            errorbars0[:,i,j] = [final_coop_freq_avg0[i,j]-final_coop_freq_min0[i,j],
                            final_coop_freq_max0[i,j]-final_coop_freq_avg0[i,j]]
            errorbars1[:,i,j] = [final_coop_freq_avg1[i,j]-final_coop_freq_min1[i,j],
                            final_coop_freq_max1[i,j]-final_coop_freq_avg1[i,j]]
    
    # Set colors for plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(b)))
    
    # Plot final cooperator frequency for different b values
    plt.subplots(1, 2, figsize=(15,7))
    plt.subplot(1, 2, 1)
    for j in range(len(poiss_l)):
        # Plot markers with errorbars
        plt.errorbar(b+j*0.01, final_coop_freq_avg0[:,j], errorbars0[:,:,j], 
                     color=colors[j], marker="o", markersize=10, capsize=5,
                     label=f"$\lambda = {poiss_l[j]:d}$")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("$b_{B}$", fontsize=24)
        plt.ylabel("Final Coop. Freq. A", fontsize=24)
    plt.subplot(1, 2, 2)
    for j in range(len(poiss_l)):
        # Plot markers with errorbars
        plt.errorbar(b+j*0.01, final_coop_freq_avg1[:, j], errorbars1[:,:,j], 
                     color=colors[j], marker="o", markersize=10, capsize=5,
                     label=f"$\lambda = {poiss_l[j]:d}$")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("$b_{B}$", fontsize=24)
        plt.ylabel("Final Coop. Freq. B", fontsize=24)
    plt.legend(loc=(1.01, 0.5), fontsize=16)
    
    # Save plot to file or show it
    if save_files:
        plt.savefig("bipartite_final_coop_freq_vs_b.pdf", bbox_inches="tight")
        plt.close()
        
    else:
        plt.show()

# b = np.arange(1.1, 1.5, 0.1)
# poiss_l = np.arange(1, 10, 3)
# seeds = [i for i in range(20)]
# coop_freqs0, coop_freqs1 = gen_final_coop_freq(20, 5, 160, nt=300, nt_save=20, b=b, init_coop_freq0=0.9, 
#                                              init_coop_freq1=0.9, poiss_l=poiss_l, seeds=seeds)

# plot_final_coop_freq(coop_freqs0, coop_freqs1, b, poiss_l, save_files=False)

##############################################################################

# Generate final cooperator frequency for different b values
def gen_final_coop_freq(cluster_size, n_cluster, n_keep, nt, nt_save, b0, b1, init_coop_freq0=0.5, 
                        init_coop_freq1=0.5, poiss_l=1, seeds=[None]):
    # n: lattice side -> number of sites = n^2
    # knn: number of nearest neighbors to connect
    # rewire: probability of rewiring a connection
    # nt: number of timesteps to evolve before annotating results
    # nt_save: number of timesteps to annotate results for calculating statistics
    # b: array of values for b parameter value for the payoff matrix
    # eps: eps parameter value for the payoff matrix
    # init_coop_freq: frequency of cooperators on initial condition
    # init_cond: initial condition of the lattice
    # save_files: wether to save plots to files or not
    # seed: random number generator seed
    
    n = cluster_size*n_cluster
    
    # Array to store cooperator frequency for different b values and different timesteps
    coop_freqs0 = np.zeros((len(b1), len(poiss_l), len(seeds), nt_save))
    coop_freqs1 = np.zeros((len(b1), len(poiss_l), len(seeds), nt_save))
    
    payoff_mat0 = np.array([[1., 0],[b0, 0]]) # Define the payoff matrix
    
    # Loop over b values
    for j in range(len(b1)):
        for l in range(len(poiss_l)):
            for s in range(len(seeds)):
                
                payoff_mat1 = np.array([[1., 0],[b1[j], 0]]) # Define the payoff matrix
                
                network, colormap = gen_modular_net(cluster_size, n_cluster, init_coop_freq0, 
                                                   init_coop_freq1, n_keep, poiss_l[l], seeds[s])
                calc_fit_mat(network, n, payoff_mat0, payoff_mat1)
                
                # Time evolution = Loop over timesteps
                for i in range(1, nt):
                    evolve_strats(network, colormap, n, payoff_mat0, payoff_mat1) # Evolve the network by a timestep
                    
                    print("\r"+"\t"*10, end="")
                    print(f"\rb: {j+1}/{len(b1)}; lambda: {l+1}/{len(poiss_l)}; time: {i+1}/{nt}; seed: {s+1}/{len(seeds)}", end="")
                
                for i in range(nt_save):
                    coop_freqs0[j,l,s,i], coop_freqs1[j,l,s,i] = evolve_strats(network, colormap, n, 
                                                                         payoff_mat0, payoff_mat1) # Evolve the network by a timestep
                    
                    print("\r"+"\t"*10, end="")
                    print(f"\rb: {j+1}/{len(b1)}; lambda: {l+1}/{len(poiss_l)}; time: {i+1}/{nt_save}; seed: {s+1}/{len(seeds)}", end="")
    
    return coop_freqs0, coop_freqs1

# Plot statistics of final cooperator frequency for different b values
def plot_final_coop_freq(coop_freqs0, coop_freqs1, b0, b1, poiss_l, save_files=False):
    # coop_freq: array containing some timesteps of the cooperator frequency for different values of b
    #        |-> shape: (len(b), # of timesteps)
    # b: array of b values considered for generating "coop_freq"
    # save_files: wether or not to save plot to file
    
    avg_coop_freqs0 = np.mean(coop_freqs0, axis=-1)
    avg_coop_freqs1 = np.mean(coop_freqs1, axis=-1)
    
    final_coop_freq_avg0 = np.mean(avg_coop_freqs0, axis=-1) # Average final cooperator frequencies
    final_coop_freq_min0 = np.min(avg_coop_freqs0, axis=-1) # Minimum final cooperator frequencies
    final_coop_freq_max0 = np.max(avg_coop_freqs0, axis=-1) # Maximum final cooperator frequencies
    
    final_coop_freq_avg1 = np.mean(avg_coop_freqs1, axis=-1) # Average final cooperator frequencies
    final_coop_freq_min1 = np.min(avg_coop_freqs1, axis=-1) # Minimum final cooperator frequencies
    final_coop_freq_max1 = np.max(avg_coop_freqs1, axis=-1) # Maximum final cooperator frequencies
    
    # Generate errorbars from minimum to maximum cooperator frequencies
    errorbars0 = np.zeros((2, len(b1), len(poiss_l)))
    errorbars1 = np.zeros((2, len(b1), len(poiss_l)))
    for i in range(len(b1)):
        for j in range(len(poiss_l)):
            errorbars0[:,i] = [final_coop_freq_avg0[i,j]-final_coop_freq_min0[i,j],
                            final_coop_freq_max0[i,j]-final_coop_freq_avg0[i,j]]
            errorbars1[:,i] = [final_coop_freq_avg1[i,j]-final_coop_freq_min1[i,j],
                            final_coop_freq_max1[i,j]-final_coop_freq_avg1[i,j]]
    
    # Set colors for plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(b1)))
    
    # Plot final cooperator frequency for different b values
    plt.figure(figsize=(10,7))
    for j in range(len(poiss_l)):
        # Plot markers with errorbars
        plt.errorbar(b1, final_coop_freq_avg0[:, j], errorbars0[:,j], 
                     color=colors[j], marker="o", markersize=10, capsize=5,
                     label=f"$b = {b1[j]:0.2f}$")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("$b_{B}$", fontsize=24)
        plt.ylabel("Final Cooperator Frequency", fontsize=24)
    # plt.legend(loc=(1.01, 0.5), fontsize=16)
    
    # Save plot to file or show it
    if save_files:
        plt.savefig("bipartite_final_coop_freq_vs_b.pdf", bbox_inches="tight")
        plt.close()
        
    else:
        plt.show()