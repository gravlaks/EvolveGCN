import networkx as nx
import numpy as np
sizes = [20, 20, 20, 20, 20]
probs = [[0.5, 0.05, 0.02, 0.01, 0.2], 
        [0.05, 0.04, 0.03, 0.5, 0.07],
        [0.02, 0.03, 0.3, 0.05, 0.07],
        [0.01, 0.5, 0.05, 0.5, 0.07], 
        [0.2, 0.07, 0.07, 0.07, 0.4]]

G = nx.stochastic_block_model(sizes, probs, seed = 42)
print(len(G.edges))
N = len(G.edges)
edge_array = np.zeros((N, 4)).astype(int)
num_timesteps = 20
for i, edge in enumerate(G.edges):
    timestep = i//(N//num_timesteps)
    edge_array[i] = np.array([
        edge[0], edge[1], 1, timestep
    ])
edge_array = edge_array.astype(int)
np.savetxt("data/foo.csv", edge_array, fmt='%i', delimiter=",")

with open("data/foo.csv", "r") as f:
    line = f.read()
    print(line)
with open("data/foo.csv", "w") as f:
    line = "source,target,weight,time\n" + line

    f.write(line)