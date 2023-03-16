import networkx as nx
import numpy as np
sizes = [20, 20, 20, 20, 20]
probs = [[0.3, 0.05, 0.02, 0.01, 0.2], 
        [0.05, 0.04, 0.03, 0.5, 0.07],
        [0.02, 0.03, 0.1, 0.05, 0.07],
        [0.01, 0.5, 0.05, 0.2, 0.07], 
        [0.2, 0.07, 0.07, 0.07, 0.2]]

G = nx.stochastic_block_model(sizes, probs, seed = 42)
print(len(G.edges))
N = len(G.edges)
edge_array = np.zeros((N, 4)).astype(int)
num_timesteps = 20

G = nx.stochastic_block_model(sizes, probs, seed = 42)

for i, edge in enumerate(G.edges):
    edge_array[i] = np.array([
        edge[0], edge[1], 1, 0
    ])


p = 0.1
for timestep in range(1, 20):
    G = nx.stochastic_block_model(sizes, probs, seed = 42)
    new_edge_array = np.zeros((N, 4)).astype(int)

    for i, edge in enumerate(G.edges):
        new_edge_array[i] = np.array([
        edge[0], edge[1], 1, timestep
    ])
    
    new_edges = edge_array.copy()
    indices = np.random.choice(range(len(new_edges)), p*len(new_edges))
    indices = torch.randperm(len(edges[:10000]))[:num_edges]

    new_edges[indices] = new_edge_array[indices]
edge_array = edge_array.astype(int)
np.savetxt("data/foo.csv", edge_array, fmt='%i', delimiter=",")

with open("data/foo.csv", "r") as f:
    line = f.read()
    print(line)
with open("data/foo.csv", "w") as f:
    line = "source,target,weight,time\n" + line

    f.write(line)