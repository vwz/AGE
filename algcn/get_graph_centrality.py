from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np
import sys
import os
from utils import load_randomalpdata

dataset = sys.argv[1]

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_train, labels, graph = load_randomalpdata(dataset, 0, 4)

#graph central
def centralissimo(G):
    centralities = []
    #centralities.append(nx.degree_centrality(G))       #print 'degree centrality: check.'
    #centralities.append(nx.closeness_centrality(G))    #print 'closeness centrality: check.'
    #centralities.append(nx.betweenness_centrality(G))  #print 'betweenness centrality: check.'
    #centralities.append(nx.eigenvector_centrality(G))  #print 'eigenvector centrality: check.'
    centralities.append(nx.pagerank(G))                #print 'page rank: check.'
    #centralities.append(nx.harmonic_centrality(G))
    L = len(centralities[0])
    Nc = len(centralities)
    cenarray = np.zeros((Nc,L))
    for i in range(Nc):
    	cenarray[i][list(centralities[i].keys())]=list(centralities[i].values())
    normcen = (cenarray.astype(float)-np.min(cenarray,axis=1)[:,None])/(np.max(cenarray,axis=1)-np.min(cenarray,axis=1))[:,None]
    return normcen


directory = "res/"+dataset+"/graphcentrality/"
if not os.path.exists(directory):
    os.makedirs(directory)

G = nx.Graph(graph)
normcen = centralissimo(G)	#the larger the score is, the more representative the node is in the graph
np.savetxt(directory+'normcen', normcen, fmt='%.6f', delimiter=' ', newline='\n')

