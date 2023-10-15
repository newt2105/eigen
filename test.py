import networkx as nx
import numpy as np

def GetBandwidth(graph, path): #Lay ra bandwidth nho nhat cua do thi
    graph_bandwidth_list = nx.get_edge_attributes(graph, name="weight")
    # print("graph_bandwidth_list:",graph_bandwidth_list)
    path_list = [(a,b) for a in path for b in path if not (a==b) and (a<=b)]
    # print("pathlist:",path_list)
    bandwidth_list = [graph_bandwidth_list.get(l, -1) for l in path_list if graph_bandwidth_list.get(l, -1) >= 0]
    # print("bndlist:", bandwidth_list)
    bandwidth_list.sort()
    return bandwidth_list[0]

def GetMaxIndirectBandwidth(graph, start, destination): #Lay ra maxbandwidth cua node a den node b
    available_path = list(nx.all_simple_paths(graph, start, destination))
    # print("available_path: ",available_path)
    bandwidth_list = [(GetBandwidth(graph, path), path) for path in available_path]
    bandwidth_list.sort(reverse=True, key=lambda x:x[0])
    # print("bandwidth_list: ", bandwidth_list)
    if (not len(bandwidth_list)):
        return None, None
    return bandwidth_list[0]

SubtrateNetwork = nx.DiGraph()
SubtrateNetwork.add_node(1, weight=3)
SubtrateNetwork.add_node(2, weight=8)
SubtrateNetwork.add_node(3, weight=15)
SubtrateNetwork.add_node(4, weight=5)
SubtrateNetwork.add_node(5, weight=11)
SubtrateNetwork.add_node(6, weight=2)
SubtrateNetwork.add_edge(1,2, weight=7)
SubtrateNetwork.add_edge(2,1, weight=7)
SubtrateNetwork.add_edge(2,3, weight=9)
SubtrateNetwork.add_edge(3,2, weight=9)
SubtrateNetwork.add_edge(3,4, weight=4)
SubtrateNetwork.add_edge(4,3, weight=4)
SubtrateNetwork.add_edge(1,4, weight=3)
SubtrateNetwork.add_edge(4,1, weight=3)
SubtrateNetwork.add_edge(2,4, weight=2)
SubtrateNetwork.add_edge(4,2, weight=2)
SubtrateNetwork.add_edge(1,5, weight=1)
SubtrateNetwork.add_edge(5,1, weight=1)
SubtrateNetwork.add_edge(4,5, weight=3)
SubtrateNetwork.add_edge(5,4, weight=3)
SubtrateNetwork.add_edge(3,5, weight=10)
SubtrateNetwork.add_edge(5,3, weight=10)
SubtrateNetwork.add_edge(5,6, weight=5)
SubtrateNetwork.add_edge(6,5, weight=5)

ServiceChain = nx.DiGraph()
ServiceChain.add_node("a", weight=3)
ServiceChain.add_node("b", weight=2)
ServiceChain.add_node("c", weight=5)
ServiceChain.add_node("d", weight=3)
ServiceChain.add_edge("a", "b", weight=2)
ServiceChain.add_edge("b", "c", weight=2)
ServiceChain.add_edge("c", "d", weight=2)
ServiceChain.add_edge("d", "b", weight=5)
ServiceChain.add_edge("b", "a", weight=5)

############################################


A_i = np.array(nx.adjacency_matrix(SubtrateNetwork).todense())

for i in range(len(A_i)):
    for j in range(len(A_i[i])):
        if A_i[i][j] > 0:
            continue
        if i==j:
            A_i[i][j] = nx.get_node_attributes(SubtrateNetwork, name="weight").get(i+1, 0)
            continue
        A_i[i][j], _ = GetMaxIndirectBandwidth(SubtrateNetwork, i+1,j+1)


##########################################


A_nct = np.pad(np.array(nx.adjacency_matrix(ServiceChain).todense()), [(0,2),(0,2)], mode="constant")

for i in range(len(ServiceChain.nodes)):
    A_nct[i][i] = nx.get_node_attributes(ServiceChain, name="weight").get(list(ServiceChain.nodes)[i],0)

for i in range(len(ServiceChain.nodes)):
    A_nct[i][i] = nx.get_node_attributes(ServiceChain, name="weight").get(list(ServiceChain.nodes)[i],0)
# Tìm tất cả các cặp cạnh lặp
repeated_edges = []
for i in range(len(A_nct)):
    for j in range(i + 1, len(A_nct)):
        if A_nct[i][j] >=0 and A_nct[j][i]>=0:
            repeated_edges.append((i, j))

# Gộp các cạnh lặp thành một cạnh duy nhất
for i, j in repeated_edges:
    weight = A_nct[i][j] + A_nct[j][i]
    A_nct[i][j] = weight
    A_nct[j][i] = weight


##########################################


def get_node_key_by_index(graph, index):
    node_names = list(graph.nodes)
    return node_names[index]
def get_node_index_by_key(graph, key):
    node_names = list(graph.nodes)
    return node_names.index(key)


############################################


M = A_nct * A_i.transpose()


############################################

def Embeding(subtrate, service, M):
    P = np.zeros(shape=(4,6), dtype=np.int8)
    for vnf in service.nodes:
        # print(vnf)
        i = list(service.nodes).index(vnf)
        # print(i)
        row = M[i]
        
        k_list = [(j, row[j]) for j in range(len(row))]
        k_list.sort(reverse=True, key=lambda x:x[1])
        k_list = [j[0] for j in k_list]
        
        
        for k in k_list:
            node_check = CheckNode(subtrate, service, P, k, i)
            link_check, path = CheckLink(subtrate, service, P, k, i)
            if not node_check:
                # print(i, k, "node fail")
                continue
            if not link_check:
                # print(i, k, "link fail")
                continue
            P[i][k] = 1
            print(get_node_key_by_index(service,i),get_node_key_by_index(subtrate,k))
            UpdateGraph(subtrate, service, P, k, i, path)
            break
    # map all
    if (P.sum() < 4):
        # cancel mapping if unable to map all
        P = np.zeros(shape=(4,6), dtype=np.int8)
    return P

def UpdateGraph(subtrate, service, P, k, i, path):
    node_caps = nx.get_node_attributes(subtrate, name="weight")
    link_caps = nx.get_edge_attributes(subtrate, name="weight")
    prev_i = i-1
    prev_i_name = get_node_key_by_index(service, prev_i)
    i_name = get_node_key_by_index(service, i)
    prev_k = P[prev_i].argmax()
    prev_k_name = get_node_key_by_index(subtrate, prev_k)
    k_name = get_node_key_by_index(subtrate, k)

    node_req = nx.get_node_attributes(service, name="weight").get(i_name)
    node_caps[k_name] -= node_req

    nx.set_node_attributes(subtrate, node_caps, "weight")

    if not path:
        return

    link_req = nx.get_edge_attributes(service, name="weight").get((prev_i_name, i_name)) 
    path_list = [(a,b) for a in path for b in path if not (a==b) and (a<=b)]
    for p in path_list:
        if (p not in link_caps.keys()):
            continue
        link_caps[p] -= link_req

    nx.set_edge_attributes(subtrate, link_caps, "weight")
    return

def CheckNode(subtrate, service, P, k, i):
    sg_k_cap = nx.get_node_attributes(subtrate, name="weight")[get_node_key_by_index(subtrate, k)]
    nct_i_req = nx.get_node_attributes(service, name="weight")[get_node_key_by_index(service, i)]
    # Node req
    if (nct_i_req > sg_k_cap):
        return False
    # Map once
    if (any(P.transpose()[k] == 1)):
        return False
    return True

def CheckLink(subtrate, service, P, k, i):
    i_name = get_node_key_by_index(service, i)
    k_name = get_node_key_by_index(subtrate, k)
    if (i == 0):
        return True, None
    prev_i = i-1
    prev_i_name = get_node_key_by_index(service, prev_i)
    prev_k = P[prev_i].argmax()
    prev_k_name = get_node_key_by_index(subtrate, prev_k)
    req = nx.get_edge_attributes(service, name="weight").get((prev_i_name, i_name))
    avail, path = GetMaxIndirectBandwidth(subtrate, prev_k_name, k_name)
    if not avail:
        return False, None
    if (req > avail):
        return False, None
    return True, path

P = Embeding(SubtrateNetwork, ServiceChain, M)
print(P)