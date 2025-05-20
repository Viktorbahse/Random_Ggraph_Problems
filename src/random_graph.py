import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import networkx as nx

# Экспоненциальное распределение
def generate_exponential_samples(n, lambda_param):
    samples = np.random.exponential(1/lambda_param, n)
    return samples

# Гамма распределение
def generate_gamma_samples(n, lambda_param):
    alpha = 1 / 2
    samples = np.random.gamma(alpha, 1 / lambda_param, n)
    return samples

#Генерация Normal(0,σ); 
def generate_normal_samles(loc=0, scale=1, size):
    return np.random.normal(loc, sigma, size)

#Генерация Student-t(ν) 
def generate_standard_t_samles(df=3, size):
    return np.random.standard_t(df, size)

# Построение knn-графа
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

    # pos = nx.get_node_attributes(G, 'pos')
    # plt.figure(figsize=(10, 6))
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
    # plt.title(f'KNN Graph (k={k})')
    # plt.xlabel('Sample Values')
    # plt.yticks([])
    # plt.show()

    return G

# Построение дистанционного графа
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

# макс степень вершины
def max_node_degree(graph):
    if not graph:
        return 0

    degrees = dict(graph.degree())
    max_degree = max(degrees.values())
    return max_degree

# макс. независимое множество
def max_independent_set_size(graph):
    if not graph:
        return 0
    independent_set = nx.algorithms.approximation.maximum_independent_set(graph)
    return len(independent_set)

#δ(G) - минимальная степень
def max_node_degree(graph):
    if not graph:
        return 0

    degrees = dict(graph.degree())
    return min(degrees.values())

#χ(G) - Хроматическое число
def chromatic_number(graph):
    if not graph:
        return 0
    return nx.algorithms.coloring.greedy_color(graph, strategy='largest_first', interchange=True) + 1

# Эксперименты с графом knn и экспоненциальным распределением
n_samples = 4000
k_neighbors = 7
lambda_values = np.linspace(0.1, 2.5, 25)  # 15 значений lambda
max_degrees = []

for lambda_param in lambda_values:
    max_degrees_for_lambda = []
    for _ in range(1000):
        samples = generate_exponential_samples(n_samples, lambda_param)
        G = generate_knn_graph(samples, k_neighbors)
        max_degree = max_node_degree(G)
        max_degrees_for_lambda.append(max_degree)
    mean_max_degree = np.mean(max_degrees_for_lambda)
    max_degrees.append(mean_max_degree)

plt.figure(figsize=(10, 6))
plt.plot(lambda_values, max_degrees, marker='o')
plt.title('Зависимость математического ожидания максимальной степени от lambda')
plt.xlabel('Параметр lambda')
plt.ylabel('Математическое ожидание максимальной степени')
plt.grid()
plt.show()

# Эксперименты с графом knn и гамма-распределением

for lambda_param in lambda_values:
    max_degrees_for_lambda = []
    for _ in range(1000):
        samples = generate_gamma_samples(n_samples, lambda_param)
        G = generate_knn_graph(samples, k_neighbors)
        max_degree = max_node_degree(G)
        max_degrees_for_lambda.append(max_degree)
    mean_max_degree = np.mean(max_degrees_for_lambda)
    max_degrees.append(mean_max_degree)

plt.figure(figsize=(10, 6))
plt.plot(lambda_values, max_degrees, marker='o')
plt.title('Зависимость математического ожидания максимальной степени от lambda')
plt.xlabel('Параметр lambda')
plt.ylabel('Математическое ожидание максимальной степени')
plt.grid()
plt.show()

# Эксперименты с d-графом и экспоненциальным распределением
d_distance = 0.5
for lambda_param in lambda_values:
    max_degrees_for_lambda = []
    for _ in range(1000):
        samples = generate_exponential_samples(n_samples, lambda_param)
        G = generate_distance_graph(samples, d_distance)
        max_degree = max_node_degree(G)
        max_degrees_for_lambda.append(max_degree)
    mean_max_degree = np.mean(max_degrees_for_lambda)
    max_degrees.append(mean_max_degree)
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, max_degrees, marker='o')
plt.title('Зависимость математического ожидания максимальной степени от lambda (дистанционный граф)')
plt.xlabel('Параметр lambda')
plt.ylabel('Математическое ожидание максимальной степени')
plt.grid()
plt.show()

# Эксперименты с d-графом и гамма-распределением
d_distance = 0.5
for lambda_param in lambda_values:
    max_degrees_for_lambda = []
    for _ in range(1000):
        samples = generate_gamma_samples(n_samples, lambda_param)
        G = generate_distance_graph(samples, d_distance)
        max_degree = max_node_degree(G)
        max_degrees_for_lambda.append(max_degree)
    mean_max_degree = np.mean(max_degrees_for_lambda)
    max_degrees.append(mean_max_degree)
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, max_degrees, marker='o')
plt.title('Зависимость математического ожидания максимальной степени от lambda (дистанционный граф)')
plt.xlabel('Параметр lambda')
plt.ylabel('Математическое ожидание максимальной степени')
plt.grid()
plt.show()