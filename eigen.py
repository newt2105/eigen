import networkx as nx
import numpy as np
from gen import create_random_physical_network, create_sfc_graph
from test2 import Get_PHY_matrix, Get_SFC_matrix, M, Embeding

# SubtrateNetwork = nx.Graph()
# SubtrateNetwork.add_node(1, weight=3)
# SubtrateNetwork.add_node(2, weight=8)
# SubtrateNetwork.add_node(3, weight=15)
# SubtrateNetwork.add_node(4, weight=5)
# SubtrateNetwork.add_node(5, weight=11)
# SubtrateNetwork.add_node(6, weight=2)
# SubtrateNetwork.add_edge(1,6, weight=7)
# SubtrateNetwork.add_edge(1,3, weight=7)
# SubtrateNetwork.add_edge(1,4, weight=9)
# SubtrateNetwork.add_edge(2,5, weight=9)
# SubtrateNetwork.add_edge(3,5, weight=4)
# SubtrateNetwork.add_edge(3,4, weight=4)
# SubtrateNetwork.add_edge(3,6, weight=3)
# SubtrateNetwork.add_edge(4,5, weight=3)
# SubtrateNetwork.add_edge(4,6, weight=2)

# ServiceChain = nx.DiGraph()
# ServiceChain.add_node("a", weight=3)
# ServiceChain.add_node("b", weight=2)
# ServiceChain.add_node("c", weight=5)
# ServiceChain.add_node("d", weight=3)
# ServiceChain.add_edge("a", "b", weight=2)
# ServiceChain.add_edge("b", "a", weight=2)
# ServiceChain.add_edge("b", "c", weight=2)
# ServiceChain.add_edge("c", "b", weight=5)

PHYgraph = create_random_physical_network(6, 9)
print(PHYgraph.nodes)
print(PHYgraph.edges)
SFCgraph = create_sfc_graph(3, 4)
print(SFCgraph.nodes)
print(SFCgraph.edges)


A_i = Get_PHY_matrix(PHYgraph)
# print(A_i)

A_nct = Get_SFC_matrix(PHYgraph,SFCgraph)
# print(A_nct)
M = M(A_i, A_nct)


P = Embeding(PHYgraph, SFCgraph, M)

print(P)



