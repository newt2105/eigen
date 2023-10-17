import networkx as nx
import numpy as np
from gen import create_random_physical_network, create_sfc_graph

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
        return 0,0
    return bandwidth_list[0]

def Get_PHY_matrix(graph1):
    A_i = np.array(nx.adjacency_matrix(graph1).todense())

    for i in range(len(A_i)):
        for j in range(len(A_i[i])):
            if A_i[i][j] > 0:
                continue
            if i==j:
                A_i[i][j] = nx.get_node_attributes(graph1, name="weight").get(i+1, 0)
                continue
            A_i[i][j], _ = GetMaxIndirectBandwidth(graph1, i+1,j+1)

    return A_i

def Get_SFC_matrix(graph1, graph2):
    A_i = Get_PHY_matrix(graph1)
    num_rows_A_i = len(A_i)
    num_cols_A_i = len(A_i[0])  
    A_nct = np.array(nx.adjacency_matrix(graph2).todense())

    num_rows_A_nct = len(A_nct)
    num_cols_A_nct = len(A_nct[0])
    

    a = num_rows_A_i - num_rows_A_nct
    b = num_cols_A_i - num_cols_A_nct
    A_nct = np.pad(np.array(nx.adjacency_matrix(graph2).todense()), [(0,a),(0,b)], mode="constant")

    for i in range(len(graph2.nodes)):
        A_nct[i][i] = nx.get_node_attributes(graph2, name="weight").get(list(graph2.nodes)[i],0)

    for i in range(len(graph2.nodes)):
        A_nct[i][i] = nx.get_node_attributes(graph2, name="weight").get(list(graph2.nodes)[i],0)
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

    return A_nct

def get_node_key_by_index(graph, index):
    node_names = list(graph.nodes)
    return node_names[index]
def get_node_index_by_key(graph, key):
    node_names = list(graph.nodes)
    return node_names.index(key)

def M(matrix1, matrix2):
    M = matrix2 * matrix1.transpose()
    return M

def Embeding(subtrate, service, M):
    A_i = Get_PHY_matrix(subtrate)
    num_rows_A_i = len(A_i)
    num_cols_A_i = len(A_i[0])
    P = np.zeros(shape=(num_rows_A_i,num_cols_A_i), dtype=np.int8)
    for vnf in service.nodes:
        # print(vnf)
        i = list(service.nodes).index(vnf)
        # print(i)
        row = M[i]
        
        k_list = [(j, row[j]) for j in range(len(row))]
        k_list.sort(reverse=True, key=lambda x:x[1])
        k_list = [j[0] for j in k_list]
        # print(k_list)
        # print(k_list)
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
            # print(get_node_key_by_index(service,i),get_node_key_by_index(subtrate,k))
            UpdateGraph(subtrate, service, P, k, i, path)
            break
    # map all
    if (P.sum() < num_rows_A_i):
        # cancel mapping if unable to map all
        P = np.zeros(shape=(num_rows_A_i,num_cols_A_i), dtype=np.int8)
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
    # print("i:", i)
    # print("i_name:", i_name)
    k_name = get_node_key_by_index(subtrate, k)
    # print("k_name:", k_name)
    if (i == 0):
        return True, None
    prev_i = i-1
    # print("prev_i_name:",prev_i)
    prev_i_name = get_node_key_by_index(service, prev_i)
    prev_k = P[prev_i].argmax()
    prev_k_name = get_node_key_by_index(subtrate, prev_k)
    req = nx.get_edge_attributes(service, name="weight").get((prev_i_name, i_name))
    avail, path = GetMaxIndirectBandwidth(subtrate, prev_k_name, k_name)
    # print(prev_i_name, i_name," " ,prev_k_name, k_name )
    # print(req , avail)
    if not avail:
        return False, None
    if (req > avail):
        return False, None
    return True, path