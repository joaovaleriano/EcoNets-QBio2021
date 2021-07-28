#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:24:40 2021

@author: joao-valeriano
"""

import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


def gen_nested_net(n0, n1, coop_freq0, coop_freq1, seed=None):
    # number_of_nodes: number of nodes in the network
    # knn: number of nearest neighbors to connect
    # rewire: probability of rewiring a connection
    # coop_freq: cooperator frequency
    
    # Create network
    network = nx.complete_multipartite_graph(n0, n1)
    
    # Array to store colormap indicating cooperators (blue) and defectors (red)
    colormap = []
    
    network.remove_edges_from(list(network.edges))
    network.add_edge(0, n0)
    network.add_edge(0, n0+1)
    network.add_edge(1, n0)
    network.add_edge(1, n0+1)
    
    for i in range(2, n0):
        
        A = bipartite.projected_graph(network, [i for i in range(n0)])
        B = bipartite.projected_graph(network, [i for i in range(n0, n0+n1)])
        
        nodes = np.random.choice([i for i in range(n0, n0+n1)], 2, p=np.array(B.degree)[:,1]/np.sum(np.array(B.degree)[:,1]))
        network.add_edge(i, nodes[0])
        # network.add_edge(i, nodes[1])
        
        nodes = np.random.choice([i for i in range(n0)], 2, p=np.array(A.degree)[:,1]/np.sum(np.array(A.degree)[:,1]))
        network.add_edge(i+n0, nodes[0])
        # network.add_edge(i+n0, nodes[1])
    
    # Generate array with strategies for each node, randomly sorted
    np.random.seed(seed)
    # strat0 = np.random.choice([0,1], n0, p=[coop_freq0, 1-coop_freq0])
    # strat1 = np.random.choice([0,1], n1, p=[coop_freq1, 1-coop_freq1])
    
    strat0 = np.zeros(n0)
    strat1 = np.zeros(n0)
    
    deg0 = np.array(network.degree([i for i in range(n0)]))
    deg1 = np.array(network.degree([i for i in range(n0, n0+n1)]))
    
    # deg0 = deg0[:,0][np.argsort(deg0[:,-1])][-round(n0*(1-coop_freq0)):]
    # deg1 = deg1[:,0][np.argsort(deg1[:,-1])][-round(n1*(1-coop_freq1)):]
    
    deg0 = deg0[:,0][np.argsort(deg0[:,-1])][:round(n0*(1-coop_freq0))]
    deg1 = deg1[:,0][np.argsort(deg1[:,-1])][:round(n1*(1-coop_freq1))]
    
    strat0[deg0] = 1.
    strat1[deg1-n0] = 1.
    
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

net, cmap = gen_nested_net(100, 100, 0.5, 0.5, seed=None)
nx.draw(net, nx.get_node_attributes(net, "pos"), node_color=cmap)


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


# Show the time evolution of a network
def show_random_time_evol(n0, n1, init_coop_freq0, init_coop_freq1, n_remove, nt, b0, b1, seed=None):
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
    
    network, colormap = gen_nested_net(n0, n1, init_coop_freq0, init_coop_freq1, seed)
    
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
        # plt.savefig(f"bipartite_movie/small_world{i:04d}.png", dpi=300)
        plt.show()

# show_ring_time_evol(100, 100, 0.5, 0.5, 5, nt=100, b0=1.1, b1=1.1, seed=None)


##############################################################################

# Show the time evolution of a network
def show_random_time_evol_wparts(n0, n1, init_coop_freq0, init_coop_freq1, n_keep, nt, b0, b1, seed=None):
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
    network, colormap = gen_nested_net(n0, n1, init_coop_freq0, init_coop_freq1, seed)
    
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
        
# show_random_time_evol_wparts(100, 100, 0.05, 0.05, 200, nt=100, b0=1.9, b1=1.9, seed=1)
