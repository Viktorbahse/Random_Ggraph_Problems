import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import networkx as nx

def generate_exponential_samples(n, lambda_param):
    samples = np.random.exponential(1/lambda_param, n)
    return samples

def generate_gamma_samples(n, lambda_param):
    alpha = 1 / 2
    samples = np.random.gamma(alpha, 1 / lambda_param, n)
    return samples

def generate_knn_graph(sample, k):
    sample = np.array(sample).reshape(-1, 1)

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(sample)

    distances, indices = knn.kneighbors(sample)

    G = nx.Graph()

    for i in range(len(sample)):
        G.add_node(i, pos=(sample[i][0], 0))
        for j in range(1, k):
            G.add_edge(i, indices[i][j])

    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    plt.title(f'KNN Graph (k={k})')
    plt.xlabel('Sample Values')
    plt.yticks([])
    plt.show()

    return G

def generate_distance_graph(sample, d):
    G = nx.Graph()

    for i in range(len(sample)):
        G.add_node(i, pos=(sample[i], 0))

    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            if abs(sample[i] - sample[j]) <= d:
                G.add_edge(i, j)

    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    plt.title(f'Distance Graph with d = {d}')
    plt.xlabel('Sample Values')
    plt.yticks([])
    plt.show()
    return G

n = 10  # количество реализаций
sample = [1, 2, 3, 9, 10, 11, 12]

d = 2  # фиксированный параметр d
generate_knn_graph(sample, d)

def max_node_degree(graph):
    if not graph:
        return 0

    degrees = dict(graph.degree())
    max_degree = max(degrees.values())
    return max_degree

def max_independent_set_size(graph):
    if not graph:
        return 0
    independent_set = nx.algorithms.approximation.maximum_independent_set(graph)
    return len(independent_set)