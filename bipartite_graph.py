#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:29:49 2021

@author: joao-valeriano
"""

import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Generate random circular network of cooperators (0) and defectors (1)
def gen_full_net(n0, n1, coop_freq0, coop_freq1, seed=None):
    # number_of_nodes: number of nodes in the network
    # knn: number of nearest neighbors to connect
    # rewire: probability of rewiring a connection
    # coop_freq: cooperator frequency
    
    # Create network
    network = nx.complete_multipartite_graph(n0, n1)
    
    # Array to store colormap indicating cooperators (blue) and defectors (red)
    colormap = []
    
    # Generate array with strategies for each node, randomly sorted
    np.random.seed(seed)
    strat0 = np.random.choice([0,1], n0, p=[coop_freq0, 1-coop_freq0])
    strat1 = np.random.choice([0,1], n1, p=[coop_freq1, 1-coop_freq1])
    
    # Loop over nodes
    for i in range(n0):
        
        if strat0[i] == 0: # Set node as a cooperator
            network.nodes[i]["strat"] = 0
            colormap.append("blue")
        
        else: # Set node as a defector
            network.nodes[i]["strat"] = 1
            colormap.append("yellow")
            
        # Initialize the fitness of each node
        network.nodes[i]["fit"] = 0
        
        # Set node positions
        network.nodes[i]["pos"] = (0,-i)
        
    for i in range(n0, n0+n1):
        
        if strat1[i-n0] == 0: # Set node as a cooperator
            network.nodes[i]["strat"] = 0
            colormap.append("blue")
        
        else: # Set node as a defector
            network.nodes[i]["strat"] = 1
            colormap.append("yellow")
            
        # Initialize the fitness of each node
        network.nodes[i]["fit"] = 0
        
        # Set node positions
        network.nodes[i]["pos"] = (1,n0-i)
            
    return network, colormap

def gen_random_net(n0, n1, coop_freq0, coop_freq1, n_remove, seed=None):
    # number_of_nodes: number of nodes in the network
    # knn: number of nearest neighbors to connect
    # rewire: probability of rewiring a connection
    # coop_freq: cooperator frequency
    
    # Create network
    network = nx.complete_multipartite_graph(n0, n1)
    
    # Array to store colormap indicating cooperators (blue) and defectors (red)
    colormap = []
    
    # Generate array with strategies for each node, randomly sorted
    np.random.seed(seed)
    strat0 = np.random.choice([0,1], n0, p=[coop_freq0, 1-coop_freq0])
    strat1 = np.random.choice([0,1], n1, p=[coop_freq1, 1-coop_freq1])
    
    # Loop over nodes
    for i in range(n0):
        
        if strat0[i] == 0: # Set node as a cooperator
            network.nodes[i]["strat"] = 0
            colormap.append("blue")
        
        else: # Set node as a defector
            network.nodes[i]["strat"] = 1
            colormap.append("yellow")
            
        # Initialize the fitness of each node
        network.nodes[i]["fit"] = 0
        
        # Set node positions
        network.nodes[i]["pos"] = (0,-i)
        
    for i in range(n0, n0+n1):
        
        if strat1[i-n0] == 0: # Set node as a cooperator
            network.nodes[i]["strat"] = 0
            colormap.append("blue")
        
        else: # Set node as a defector
            network.nodes[i]["strat"] = 1
            colormap.append("yellow")
            
        # Initialize the fitness of each node
        network.nodes[i]["fit"] = 0
        
        # Set node positions
        network.nodes[i]["pos"] = (1,n0-i)
    
    for i in range(n_remove):
        node = np.random.choice(len(network.edges))
        network.remove_edge(*list(network.edges)[node])
    
    return network, colormap

# network, colormap = gen_random_net(10, 20, 0.5, 0.5, 90, None)
# nx.draw_networkx_nodes(network, nx.get_node_attributes(network, "pos"), 
#                        node_color=colormap, node_size=100)
# nx.draw_networkx_edges(network, nx.get_node_attributes(network, "pos"))
# plt.show()

def gen_ring_net(n0, n1, coop_freq0, coop_freq1, k, seed=None):
    # number_of_nodes: number of nodes in the network
    # knn: number of nearest neighbors to connect
    # rewire: probability of rewiring a connection
    # coop_freq: cooperator frequency
    
    # Create network
    network = nx.complete_multipartite_graph(n0, n1)
    
    # Array to store colormap indicating cooperators (blue) and defectors (red)
    colormap = []
    
    # Generate array with strategies for each node, randomly sorted
    np.random.seed(seed)
    strat0 = np.random.choice([0,1], n0, p=[coop_freq0, 1-coop_freq0])
    # strat1 = np.random.choice([0,1], n1, p=[coop_freq1, 1-coop_freq1])
       
    strat1 = strat0
    
    # for i in range(n0//10*9):
    #     network.nodes[i]["strat"] = 0
    #     colormap[i] = "blue"
    #     network.nodes[i+n0]["strat"] = 0
    #     colormap[i+n0] = "blue"
    
    # for i in range(n0//10*9, n0):
    #     network.nodes[i]["strat"] = 1
    #     colormap[i] = "yellow"
    #     network.nodes[i+n0]["strat"] = 1
    #     colormap[i+n0] = "yellow"

    # Loop over nodes
    for i in range(n0):
        
        if strat0[i] == 0: # Set node as a cooperator
            network.nodes[i]["strat"] = 0
            colormap.append("blue")
        
        else: # Set node as a defector
            network.nodes[i]["strat"] = 1
            colormap.append("yellow")
            
        # Initialize the fitness of each node
        network.nodes[i]["fit"] = 0
        
        # Set node positions
        network.nodes[i]["pos"] = (0,-i)
        
    for i in range(n0, n0+n1):
        
        if strat1[i-n0] == 0: # Set node as a cooperator
            network.nodes[i]["strat"] = 0
            colormap.append("blue")
        
        else: # Set node as a defector
            network.nodes[i]["strat"] = 1
            colormap.append("yellow")
            
        # Initialize the fitness of each node
        network.nodes[i]["fit"] = 0
        
        # Set node positions
        network.nodes[i]["pos"] = (1,n0-i)

    for i in range(n0):
        for j in range(n0, n0+n1):
            if j not in [i+n0+l for l in range(k)]:
                network.remove_edge(i, j)
        
        if i+k > n0:
            for l in range(i+k-n0):
                network.add_edge(i, n0+l)
    
    return network, colormap

# network, colormap = gen_ring_net(10, 10, 0.5, 0.5, 2)
# nx.draw_networkx_nodes(network, nx.get_node_attributes(network, "pos"), 
#                         node_color=colormap, node_size=100)
# nx.draw_networkx_edges(network, nx.get_node_attributes(network, "pos"))
# plt.text(0, 1, "A", ha='center', va='center')
# plt.text(1, 1, "B", ha='center', va='center')
# plt.axis("off")
# plt.show()

def plot_partition(network, colormap, n0, n1, p):
    if p == 0:
        G = bipartite.projected_graph(network, [i for i in range(n0)])
        nx.draw(G, node_color=colormap[:n0], with_labels=True)
        plt.title("A")
        plt.show()
        
    if p == 1:
        G = bipartite.projected_graph(network, [i for i in range(n0, n0+n1)])
        nx.draw(G, node_color=colormap[n0:n0+n1], with_labels=True)
        plt.title("B")
        plt.show()

# Calculate fitness matrix over the network
def calc_fit_mat(network, n0, n1, payoff_mat0, payoff_mat1):
    # network: the network object from the NetworkX package
    # payoff_mat: payoff matrix for the game
    
    # Loop over nodes
    for i in range(n0):
        
        network.nodes[i]["fit"] = 0 # Set fitness to zero initially
        
        # Sum the contribution of each neighbor for the fitness of the focal node
        for nb in network.neighbors(i):
            # print(network.nodes[nb]["strat"])
            network.nodes[i]["fit"] += payoff_mat0[network.nodes[i]["strat"],
                                                    network.nodes[nb]["strat"]]
        # print("\n")
    # Loop over nodes
    for i in range(n0, n0+n1):
        
        network.nodes[i]["fit"] = 0 # Set fitness to zero initially
        
        # Sum the contribution of each neighbor for the fitness of the focal node
        for nb in network.neighbors(i):
            network.nodes[i]["fit"] += payoff_mat1[network.nodes[i]["strat"],
                                                    network.nodes[nb]["strat"]]
            
    # No need to return anything, we're just editing the "fit" attribute of the network


# Evolution of the network by a time step
def evolve_strats(network, colormap, n0, n1, payoff_mat0, payoff_mat1):
    # network: the network object from the NetworkX package
    # colormap: colors of the nodes indicating cooperators and defectors
    # payoff_mat: payoff matrix for the game
    
    past_network = nx.Graph.copy(network)
    past_colormap = [i for i in colormap]
    
    # Colors of nodes for cooperators (blue) and defectors (red)
    colors = ["blue", "yellow"]
    
    A = bipartite.projected_graph(network, [i for i in range(n0)])
    B = bipartite.projected_graph(network, [i for i in range(n0, n0+n1)])
    
    coopA = 0
    coopB = 0
    
    # Loop over A nodes
    for i in range(n0):
        
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
    for i in range(n0, n0+n1):
        
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
    calc_fit_mat(network, n0, n1, payoff_mat0, payoff_mat1)
    
    return coopA/n0, coopB/n1
    
# network, colormap = gen_ring_net(10, 10, 0.5, 0.5, 5)

# payoff_mat0 = np.array([[1,0],
#                         [2,0]])
# payoff_mat1 = np.array([[1,0],
#                         [3,0]])

# calc_fit_mat(network, 10, 10, payoff_mat0, payoff_mat1)

# nx.draw(network, nx.get_node_attributes(network, "pos"), 
#                         node_color=colormap, node_size=200, with_labels=True)
# plt.text(0, 1, "A", ha='center', va='center')
# plt.text(1, 1, "B", ha='center', va='center')
# plt.axis("off")
# plt.show()

# print(nx.get_node_attributes(network, "fit"))
# plot_partition(network, colormap, 10, 10, 0)
# plot_partition(network, colormap, 10, 10, 1)

# evolve_strats(network, colormap, 5, 5, payoff_mat0, payoff_mat1)

# nx.draw(network, nx.get_node_attributes(network, "pos"), 
#                         node_color=colormap, node_size=200, with_labels=True)
# plt.text(0, 1, "A", ha='center', va='center')
# plt.text(1, 1, "B", ha='center', va='center')
# plt.axis("off")
# plt.show()

# plot_partition(network, colormap, 5, 5, 0)
# plot_partition(network, colormap, 5, 5, 1)
# print(nx.get_node_attributes(network, "fit"))


# Show the time evolution of a network
def show_ring_time_evol(n0, n1, init_coop_freq0, init_coop_freq1, k, nt, b0, b1, seed=None):
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
    # network, colormap = gen_ring_net(n0, n1, init_coop_freq0, init_coop_freq1, k, seed)
    
    network, colormap = gen_ring_net(n0, n1, init_coop_freq0, init_coop_freq1, k, seed)
    # for i in range(n0):
    #     network.nodes[i+n0]["strat"] = network.nodes[i]["strat"]
    #     colormap[i+n0] = colormap[i]
    
    calc_fit_mat(network, n0, n1, payoff_mat0, payoff_mat1)
    
    # Get node positions
    node_pos = nx.get_node_attributes(network, "pos")
    
    # Draw the initial network
    nx.draw(network, node_pos, node_size=50, node_color=colormap)#, with_labels=True)
    # plt.savefig(f"small_world_movie/small_world{0:04d}.png", dpi=300)
    plt.show()
    
    # Time evolution of the network
    for i in range(1, nt):
        evolve_strats(network, colormap, n0, n1, payoff_mat0, payoff_mat1) # Evolve the network by a timestep
        
        # Plot the network
        nx.draw(network, node_pos, node_size=50, node_color=colormap)#, with_labels=True)
        plt.title(f"{i}")
        # plt.savefig(f"small_world_movie/small_world{i:04d}.png", dpi=300)
        plt.show()

# show_ring_time_evol(100, 100, 0.5, 0.5, 5, nt=100, b0=1.1, b1=1.1, seed=None)

# Show the time evolution of a network
def show_ring_time_evol_wparts(n0, n1, init_coop_freq0, init_coop_freq1, k, nt, b0, b1, seed=None):
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
    # network, colormap = gen_ring_net(n0, n1, init_coop_freq0, init_coop_freq1, k, seed)
    
    network, colormap = gen_ring_net(n0, n0, init_coop_freq0, init_coop_freq1, k, seed)
    
    # for i in range(n0):
    #     network.nodes[i+n0]["strat"] = network.nodes[i]["strat"]
    #     colormap[i+n0] = colormap[i]
    
    # for i in range(n0//6):
    #     network.nodes[i]["strat"] = 0
    #     colormap[i] = "blue"
    # for i in range(n0//6, n0):
    #     network.nodes[i]["strat"] = 1
    #     colormap[i] = "yellow"
    # for i in range(n0//3):
    #     network.nodes[i+n0]["strat"] = 0
    #     colormap[i+n0] = "blue"
    # for i in range(n0//3, n0):
    #     network.nodes[i+n0]["strat"] = 1
    #     colormap[i+n0] = "yellow"
    
    A = bipartite.projected_graph(network, [i for i in range(n0)])
    B = bipartite.projected_graph(network, [i for i in range(n0, n0+n1)])
    
    for i in range(n0):
        A.nodes[i]["pos"] = (np.cos(i*2*np.pi/n0), 
                                      np.sin(i*2*np.pi/n0))
        
    for i in range(n0, n0+n1):
        B.nodes[i]["pos"] = (np.cos(i*2*np.pi/n1), 
                                      np.sin(i*2*np.pi/n1))
    
    calc_fit_mat(network, n0, n1, payoff_mat0, payoff_mat1)
    
    # Get node positions
    node_pos = nx.get_node_attributes(network, "pos")
    
    # Draw the initial network
    plt.subplots(1,3)
    plt.subplot(1,3,1)
    nx.draw(A, nx.get_node_attributes(A, "pos"), node_size=50, node_color=colormap[:n0])
    plt.subplot(1,3,2)
    nx.draw(network, node_pos, node_size=50, node_color=colormap)#, with_labels=True)
    plt.subplot(1,3,3)
    nx.draw(B, nx.get_node_attributes(B, "pos"), node_size=50, node_color=colormap[n0:])
    plt.suptitle(f"{0}")
    # plt.savefig(f"small_world_movie/small_world{i:04d}.png", dpi=300)
    plt.show()
    
    # Time evolution of the network
    for i in range(1, nt):
        evolve_strats(network, colormap, n0, n1, payoff_mat0, payoff_mat1) # Evolve the network by a timestep
        
        # Plot the network
        plt.subplots(1,3)
        plt.subplot(1,3,1)
        nx.draw(A, nx.get_node_attributes(A, "pos"), node_size=25, node_color=colormap[:n0])
        plt.subplot(1,3,2)
        nx.draw(network, node_pos, node_size=25, node_color=colormap)#, with_labels=True)
        plt.subplot(1,3,3)
        nx.draw(B, nx.get_node_attributes(B, "pos"), node_size=25, node_color=colormap[n0:])
        plt.suptitle(f"{i}")
        # plt.savefig(f"small_world_movie/small_world{i:04d}.png", dpi=300)
        plt.show()
        
# show_ring_time_evol(100, 100, 0.9, 0.9, 5, nt=100, b0=1.2, b1=1.2, seed=None)


##############################################################################

# Generate Cooperator Frequency curves for different b values
def gen_coop_freq_evol(n0, n1, nt, b0, b1, seeds, init_coop_freq0, init_coop_freq1, k):
    # n: number of nodes in the network
    # nt: number of timesteps to run time evolution
    # b0: b parameter of payoff matrix A
    # b1: array for values of b parameter of payoff matrix A
    # eps: eps parameter of payoff matrix
    # seed: seed for random number generation
    # init_coop_freq: cooperator frequency in the initial condition
    # knn: number of nearest neighbors to connect
    # rewire: probability of rewiring a connection
    
    # Array to store cooperator frequencies for all timesteps and b values
    coop_freqs0 = np.zeros((nt, len(b1), len(seeds)))
    coop_freqs1 = np.zeros((nt, len(b1), len(seeds)))
    
    payoff_mat0 = np.array([[1., 0],[b0, 0]]) # Define the payoff matrix
    
    # Loop over b values
    for j in tqdm(range(len(b1))):
        
        # Loop over different seeds
        for s in tqdm(range(len(seeds))):
            
            payoff_mat1 = np.array([[1., 0],[b1[j], 0]]) # Define the payoff matrix
            
            # Set random number generator seed
            np.random.seed(seeds[s])
            
            # Initialize network and calculate its fitness matrix
            network, colormap = gen_ring_net(n0, n1, init_coop_freq0, init_coop_freq1, 
                                             k, seed=seeds[s])
            A = bipartite.projected_graph(network, [i for i in range(n0)])
            B = bipartite.projected_graph(network, [i for i in range(n0, n0+n1)])
            
            calc_fit_mat(network, n0, n1, payoff_mat0, payoff_mat1)
            
            coop_freqs0[0,j,s] = 1 - sum(nx.get_node_attributes(A, "strat").values()) / n0
            coop_freqs1[0,j,s] = 1 - sum(nx.get_node_attributes(B, "strat").values()) / n1
            
            # Time evolution of the network
            for i in range(1, nt):
                
                coop_freqs0[i,j,s], coop_freqs1[i,j,k] = evolve_strats(network, colormap, n0, n1, payoff_mat0, payoff_mat1) # Evolve the network by a timestep
    
    return coop_freqs0, coop_freqs1


def plot_coop_freq_evol(coop_freqs0, coop_freqs1, b0, b1, title=None, save_files=False):
    
    # Array with timesteps
    timesteps = np.linspace(1, coop_freqs0.shape[0], coop_freqs0.shape[0])
    
    avg_coop_freqs0 = np.mean(coop_freqs0, axis=2) # Average cooperator frequencies
    std_coop_freqs0 = np.std(coop_freqs0, axis=2) # Standard deviation of cooperator frequencies
    
    avg_coop_freqs1 = np.mean(coop_freqs1, axis=2) # Average cooperator frequencies
    std_coop_freqs1 = np.std(coop_freqs1, axis=2) # Standard deviation of cooperator frequencies
    
    # Set colors for plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(b1)))
    
    # Plot cooperator frequency time evolution for different b values
    plt.subplots(1, 2, figsize=(15,7))
    for i in range(len(b1)):
        plt.subplot(1, 2, 1)
        plt.plot(timesteps, avg_coop_freqs0[:,i], color=colors[i], lw=3,
                 label=f"$b = {b1[i]:0.2f}$", alpha=1.) # Plot cooperator frequency over time
        plt.fill_between(timesteps, avg_coop_freqs0[:,i]-std_coop_freqs0[:,i], 
                         avg_coop_freqs0[:,i]+std_coop_freqs0[:,i], color=colors[i], alpha=0.15)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlim(0, coop_freqs0.shape[0])
        plt.ylim(0, 1)
        plt.xlabel("Time", fontsize=24)
        plt.ylabel("Cooperator Frequency of A", fontsize=24)
        plt.ylim(0, 1)
        plt.title(title, fontsize=28)
        
        plt.subplot(1, 2, 2)
        plt.plot(timesteps, avg_coop_freqs1[:,i], color=colors[i], lw=3,
                 label=f"$b = {b1[i]:0.2f}$", alpha=1.) # Plot cooperator frequency over time
        plt.fill_between(timesteps, avg_coop_freqs1[:,i]-std_coop_freqs1[:,i], 
                         avg_coop_freqs1[:,i]+std_coop_freqs1[:,i], color=colors[i], alpha=0.15)
        plt.legend(loc=(1.01, 0.1), fontsize=16) # Add legend
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlim(0, coop_freqs0.shape[0])
        plt.ylim(0, 1)
        plt.xlabel("Time", fontsize=24)
        plt.ylabel("Cooperator Frequency of B", fontsize=24)
        plt.ylim(0, 1)
        plt.title(title, fontsize=28)
        
    if save_files:
        plt.savefig(f"bipartite_coop_freq_evol_error.pdf",
                    bbox_inches="tight")
    
    else:    
        plt.show()


b0 = 1.4
b1 = np.arange(1.2, 1.5, 0.05)

n0 = 100
n1 = 100
nt = 100
seeds = [i for i in range(200)]
init_coop_freq0 = 0.9
init_coop_freq1 = 0.9
k = 5

coop_freqs0, coop_freqs1 = gen_coop_freq_evol(n0, n1, nt, b0, b1, seeds, 
                                              init_coop_freq0, init_coop_freq1, k)
plot_coop_freq_evol(coop_freqs0, coop_freqs1, b0, b1)


##############################################################################

# Generate final cooperator frequency for different b values
def gen_final_coop_freq(n0, n1, k, nt, nt_save, b0, b1, init_coop_freq0=0.5, init_coop_freq1=0.5, seeds=[None]):
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
    
    
    # Array to store cooperator frequency for different b values and different timesteps
    coop_freqs0 = np.zeros((len(b1), len(seeds), nt_save))
    coop_freqs1 = np.zeros((len(b1), len(seeds), nt_save))
    
    payoff_mat0 = np.array([[1., 0],[b0, 0]]) # Define the payoff matrix
    
    # Loop over b values
    for j in range(len(b1)):
        for s in range(len(seeds)):
            payoff_mat1 = np.array([[1., 0],[b1[j], 0]]) # Define the payoff matrix
            
            network, colormap = gen_ring_net(n0, n1, init_coop_freq0, init_coop_freq1, k, seeds[s])
            calc_fit_mat(network, n0, n1, payoff_mat0, payoff_mat1)
            
            # Time evolution = Loop over timesteps
            for i in range(1, nt):
                evolve_strats(network, colormap, n0, n1, payoff_mat0, payoff_mat1) # Evolve the network by a timestep
                
                print(f"\rb: {j+1}/{len(b1)}; time: {i+1}/{nt}", end="")
            
            for i in range(nt_save):
                coop_freqs0[j,s,i], coop_freqs1[j,s,i] = evolve_strats(network, colormap, n0, n1, 
                                                                     payoff_mat0, payoff_mat1) # Evolve the network by a timestep
                
                print(f"\rb: {j+1}/{len(b1)}; time: {i+1}/{nt_save}", end="")
    
    return coop_freqs0, coop_freqs1

# Plot statistics of final cooperator frequency for different b values
def plot_final_coop_freq(coop_freqs0, coop_freqs1, b0, b1, save_files=False):
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
    errorbars0 = np.zeros((2, len(b1)))
    errorbars1 = np.zeros((2, len(b1)))
    for i in range(len(b1)):
        errorbars0[:,i] = [final_coop_freq_avg0[i]-final_coop_freq_min0[i],
                        final_coop_freq_max0[i]-final_coop_freq_avg0[i]]
        errorbars1[:,i] = [final_coop_freq_avg1[i]-final_coop_freq_min1[i],
                        final_coop_freq_max1[i]-final_coop_freq_avg1[i]]
    
    # Set colors for plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(b1)))
    
    # Plot final cooperator frequency for different b values
    plt.figure(figsize=(10,7))
    for j in range(len(b1)):
        # Plot markers with errorbars
        plt.errorbar(b1[j:j+1], final_coop_freq_avg0[j:j+1], errorbars0[:,j:j+1], 
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

# b0 = 1.1
# b1 = np.linspace(1.1, 1.5, 5)
# seeds = [i for i in range(10)]
# coop_freqs0, coop_freqs1 = gen_final_coop_freq(n0=100, n1=100, k=5, nt=80, nt_save=20, b0=b0, b1=b1, 
#                     init_coop_freq0=0.5, init_coop_freq1=0.5, seeds=seeds)
# plot_final_coop_freq(coop_freqs0, coop_freqs1, b0, b1, save_files=False)
