import networkx as nx
import random
import numpy as np

random_nodePHY = random.randint(2, 8)

def create_random_physical_network(num_nodes, num_edges):
  G = nx.gnm_random_graph(num_nodes, num_edges, seed=None, directed=False)
    # Tạo một ánh xạ từ số thành chữ cái in thường
  node_mapping = {}
  for i, node in enumerate(G.nodes):
      node_mapping[node] = node+1
  G = nx.relabel_nodes(G, node_mapping)

  for node in G.nodes():
     G.nodes[node]['weight'] = random.randint(1,10)  
  for (u, v) in G.edges():
      G.edges[u,v]['weight'] = random.randint(1,10)                      
  return G


def create_sfc_graph(num_nodes, num_edges):
  G = nx.gnm_random_graph(num_nodes,  num_edges, seed=None, directed=True)

  # Tạo một ánh xạ từ số thành chữ cái in thường
  node_mapping = {}
  lowercase_letters = list("abcdefghijklmnopqrstuvwxyz")
  for i, node in enumerate(G.nodes):
      node_mapping[node] = lowercase_letters[i]

  # Gán tên mới cho các nút trong đồ thị
  G = nx.relabel_nodes(G, node_mapping)
  for node in G.nodes():
     G.nodes[node]['weight'] = random.randint(1,10)
  for (u, v) in G.edges():
      G.edges[u,v]['weight'] = random.randint(1,10)
  return G

