
import numpy as np
import networkx as nx
from random import random, randint
from math import floor, log
np.random.seed(44)

def nearest_neigbor(vec_pos,query_vec):
    nearest_neighbor_index = -1
    nearest_dist = float('inf')

    nodes = []
    edges = []
    for i in range(np.shape(vec_pos)[0]):
        nodes.append((i,{"pos": vec_pos[i,:]}))
        if i<np.shape(vec_pos)[0]-1:
            edges.append((i,i+1))
        else:
            edges.append((i,0))

        dist = np.linalg.norm(query_vec-vec_pos[i])
        if dist < nearest_dist:
            nearest_neighbor_index = i
            nearest_dist = dist
        
    G_lin = nx.Graph()
    G_lin.add_nodes_from(nodes)
    G_lin.add_edges_from(edges)

    nodes = []
    nodes.append(("*",{"pos": vec_pos[nearest_neighbor_index,:]}))
    G_best = nx.Graph()
    G_best.add_nodes_from(nodes)
    return G_lin, G_best



def layer_num(max_layers: int):
    # new element's topmost layer: notice the normalization by mL
    mL = 1.5
    layer_i = floor(-1 * log(random()) * mL)
    # ensure we don't exceed our allocated layers.
    layer_i = min(layer_i, max_layers-1)
    return layer_i
    #return randint(0,max_layers-1)




def construct_HNSW(vec_pos,m_nearest_neighbor):
    max_layers = 4

    vec_num = np.shape(vec_pos)[0]
    dist_mat = np.zeros((vec_num,vec_num))

    for i in range(vec_num):
        for j in range(i,vec_num):
            dist = np.linalg.norm(vec_pos[i,:]-vec_pos[j,:])
            dist_mat[i,j] = dist
            dist_mat[j,i] = dist

    node_layer = []
    for i in range(np.shape(vec_pos)[0]):
        node_layer.append(layer_num(max_layers))
        
    max_num_of_layers = max(node_layer) + 1 ## layer indices start from 0
    GraphArray = []
    for layer_i in range(max_num_of_layers):
        nodes = []
        edges = []
        edges_nn = []
        for i in range(np.shape(vec_pos)[0]): ## Number of Vectors
            if node_layer[i] >= layer_i:
                nodes.append((i,{"pos": vec_pos[i,:]}))

        G = nx.Graph()
        G.add_nodes_from(nodes)

        pos=nx.get_node_attributes(G,'pos')

        for i in range (len(G.nodes)):
            node_i = nodes[i][0]
            nearest_edges = -1
            nearest_distances = float('inf')
            candidate_edges = range(0,i)
            candidate_edges_indices = []
            
            #######################
            for j in candidate_edges:
                node_j = nodes[j][0]
                candidate_edges_indices.append(node_j)
            
            dist_from_node = dist_mat[node_i,candidate_edges_indices]
            num_nearest_neighbor = min(m_nearest_neighbor,i) ### Add note comment
            
            if num_nearest_neighbor > 0:
                indices = np.argsort(dist_from_node)
                for nn_i in range(num_nearest_neighbor):
                        edges_nn.append((node_i,candidate_edges_indices[indices[nn_i]]))
            
            for j in candidate_edges:
                node_j = nodes[j][0]            
                dist = np.linalg.norm(pos[node_i]-pos[node_j])
                if dist < nearest_distances:
                    nearest_edges = node_j
                    nearest_distances = dist
            
            if nearest_edges != -1:
                edges.append((node_i,nearest_edges))

        G.add_edges_from(edges_nn)

        GraphArray.append(G)
        
    return GraphArray


## Search the Graph
def search_HNSW(GraphArray,G_query):
    max_layers = len(GraphArray)
    G_top_layer = GraphArray[max_layers - 1]
    num_nodes = G_top_layer.number_of_nodes()
    entry_node_r = randint(0,num_nodes-1)
    nodes_list = list(G_top_layer.nodes)
    entry_node_index = nodes_list[entry_node_r]
    #entry_node_index = 26

    SearchPathGraphArray = []
    EntryGraphArray = []
    for l_i in range(max_layers):
        layer_i = max_layers - l_i - 1
        G_layer = GraphArray[layer_i]
        
        G_entry = nx.Graph()
        nodes = []
        p = G_layer.nodes[entry_node_index]['pos']
        nodes.append((entry_node_index,{"pos": p}))
        G_entry.add_nodes_from(nodes)
        
        nearest_node_layer = entry_node_index
        nearest_distance_layer = np.linalg.norm( G_layer.nodes[entry_node_index]['pos'] - G_query.nodes['Q']['pos'])
        current_node_index = entry_node_index

        G_path_layer = nx.Graph()
        nodes_path = []
        p = G_layer.nodes[entry_node_index]['pos']
        nodes_path.append((entry_node_index,{"pos": p}))

        cond = True
        while cond:
            nearest_node_current = -1
            nearest_distance_current = float('inf')
            for neihbor_i in G_layer.neighbors(current_node_index):
                vec1 = G_layer.nodes[neihbor_i]['pos']
                vec2 = G_query.nodes['Q']['pos']
                dist = np.linalg.norm( vec1 - vec2)
                if dist < nearest_distance_current:
                    nearest_node_current = neihbor_i
                    nearest_distance_current = dist
            
            if nearest_distance_current < nearest_distance_layer:
                nearest_node_layer = nearest_node_current
                nearest_distance_layer = nearest_distance_current
                nodes_path.append((nearest_node_current,{"pos": G_layer.nodes[nearest_node_current]['pos']}))
            else:
                cond = False
        
        entry_node_index = nearest_node_layer

        G_path_layer.add_nodes_from(nodes_path)
        SearchPathGraphArray.append(G_path_layer)
        EntryGraphArray.append(G_entry)

    SearchPathGraphArray.reverse()
    EntryGraphArray.reverse()
 
    return SearchPathGraphArray, EntryGraphArray

