import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import numpy as np

def er_graph(N,p):
    ''' Generate an ER graph. '''
    G = nx.Graph()
    G.add_nodes_from(list(range(N)))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if (node1<node2) and (bernoulli.rvs(p=p)):
                G.add_edge(node1, node2)
    return G

plt.figure()
nx.draw(er_graph(50,0.08), node_size=40, node_color='gray')
plt.savefig('er1.jpg')

def plot_degree_distribution(G):
    degree_sequence = [d for n, d in G.degree()]
    plt.hist(degree_sequence, histtype='step')
    plt.xlabel('Degree $k$')
    plt.ylabel('$P(k)$')
    plt.title('Degree distribution')

plt.figure()
G = er_graph(50,0.08)
plot_degree_distribution(G)
plt.savefig('hist1.jpg')

plt.figure()
G = er_graph(500,0.8)
plot_degree_distribution(G)
plt.savefig('hist2.jpg')

A1 = np.loadtxt('./files/adj_allVillageRelationships_vilno_1.csv', delimiter=',')
A2 = np.loadtxt('./files/adj_allVillageRelationships_vilno_2.csv', delimiter=',')

G1 = nx.to_networkx_graph(A1)
G2 = nx.to_networkx_graph(A2)

def basic_net_stats(G):
    print('Number of nodes: %d' % G.number_of_nodes())
    print('Number of edges: %d' % G.number_of_edges())
    degree_sequence = [d for n, d in G.degree()]
    print("Average degree: %.2f" % np.mean(degree_sequence))

basic_net_stats(G1)
basic_net_stats(G2)

plt.figure()
plot_degree_distribution(G1)
plot_degree_distribution(G2)
plt.savefig('village_hist.jpg')
