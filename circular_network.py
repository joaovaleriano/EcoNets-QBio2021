#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 17:17:44 2021

@author: joao-valeriano
"""

# Import packages
import numpy as np # Dealing with arrays
import matplotlib.pyplot as plt # Plotting
import networkx as nx # Working with networks


# Generate random circular network of cooperators (0) and defectors (1)
def gen_net(number_of_nodes, number_of_neighbs, coop_freq):
    # number_of_nodes: number of nodes in the network
    # number_of_neighbs: number of nearest neighbors to consider from each side of the nodes
    # coop_freq: cooperator frequency
    
    # Create circular network
    circulant = nx.circulant_graph(number_of_nodes, 
                                   [i for i in range(1, number_of_neighbs+1)])
    
    # Array to store colormap indicating cooperators (blue) and defectors (red)
    colormap = []
    
    # Generate array with strategies for each node, randomly sorted
    strat = np.random.choice([0,1], number_of_nodes, p=[coop_freq, 1-coop_freq])
    
    # Loop over nodes
    for i in range(circulant.number_of_nodes()):
        
        if strat[i] == 0: # Set node as a cooperator
            circulant.nodes[i]["strat"] = 0
            colormap.append("blue")
        
        else: # Set node as a defector
            circulant.nodes[i]["strat"] = 1
            colormap.append("red")
            
        # Initialize the fitness of each node
        circulant.nodes[i]["fit"] = 0
        
        # Set node positions
        circulant.nodes[i]["pos"] = (np.cos(i*2*np.pi/number_of_nodes), 
                                     np.sin(i*2*np.pi/number_of_nodes))
            
    return circulant, colormap

    
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
    colors = ["blue", "red"]
    
    # Loop over nodes
    for i in range(past_network.number_of_nodes()):
        
        # Initialize lists to save fitnesses of cooperator and defector neighbors        
        coop_nb_fit = [0]
        defec_nb_fit = [0]
        
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

# Payoff matrix
eps = 0.
b = 1.5
payoff_mat = np.array([[1, 0],[b, eps]])

# Initial cooperator frequency
coop_freq = 0.5

# Set random number generator seed
np.random.seed(466)

# Initialize network and calculate the fitness of its nodes
network, colormap = gen_net(100, 4, 0.9)
calc_fit_mat(network, payoff_mat)

# Get node positions
node_pos = nx.get_node_attributes(network, "pos")

# Draw the initial network
nx.draw(network, node_pos, node_size=50, node_color=colormap, with_labels=True)
plt.show()

# Number of timesteps to run
nt = 100

# Time evolution of the network
for i in range(nt):
    
    evolve_strats(network, colormap, payoff_mat) # Evolve the network by a timestep
    
    # Plot the network
    nx.draw(network, node_pos, node_size=50, node_color=colormap, with_labels=True)
    plt.show()