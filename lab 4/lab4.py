import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from multiprocessing import Pool, cpu_count
from tqdm import tqdm



def generate_points(n, min_coord=-100, max_coord=100):
    points = np.random.uniform(min_coord, max_coord, (n, 3))
    return points

def generate_graph(points, directed=False):
    n = len(points)
    distances = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    if directed:
        randomness = np.random.uniform(0, 1, (n, n))
        distances += randomness * distances
    return distances



class AntColony:
    def __init__(self, graph, n_ants=10, n_iterations=100, decay=0.5, alpha=1, beta=1):
        self.graph = graph
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.n_nodes = graph.shape[0]
        self.pheromone = np.ones((self.n_nodes, self.n_nodes))
    
    def probability(self, ant, candidates):
        pheromone = self.pheromone[ant.current, candidates]
        eta = 1.0 / self.graph[ant.current, candidates]
        numerator = pheromone ** self.alpha * eta ** self.beta
        denominator = numerator.sum()
        probabilities = numerator / denominator
        return probabilities
    
    def update_pheromone(self, delta_pheromones):
        for delta in delta_pheromones:
            self.pheromone = (1 - self.decay) * self.pheromone + delta
            self.pheromone = np.clip(self.pheromone, 0, 1)
    
    def solve(self):
        best_path = None
        best_distance = float('inf')
        with Pool(cpu_count()) as pool:
            for _ in tqdm(range(self.n_iterations)):
                ants = [Ant(self, i) for i in range(self.n_ants)]
                chunk_size = len(ants) // cpu_count()
                chunks = [ants[i:i+chunk_size] for i in range(0, len(ants), chunk_size)]
                results = pool.map(self.worker, chunks)
                delta_pheromones = [result[0] for result in results]
                self.update_pheromone(delta_pheromones)
                for result in results:
                    path, distance = result[1], result[2]
                    if distance < best_distance:
                        best_distance = distance
                        best_path = path
        return best_path, best_distance
    
    def worker(self, ants_subset):
        delta_pheromone = np.zeros_like(self.pheromone)
        best_distance = float('inf')
        best_path = None
        for ant in ants_subset:
            while not ant.is_complete():
                candidates = np.array([node for node in range(self.n_nodes) if node not in ant.tabu])
                probabilities = self.probability(ant, candidates)
                next_node = np.random.choice(candidates, p=probabilities)
                ant.move_to(next_node)
            delta_pheromone += self.calculate_delta_pheromone(ant)
            if ant.total_distance < best_distance:
                best_distance = ant.total_distance
                best_path = ant.path.copy()
        return delta_pheromone, best_path, best_distance
    
    def calculate_delta_pheromone(self, ant):
        delta = np.zeros_like(self.pheromone)
        for move in ant.path:
            i, j = move
            delta[i, j] += 1 / ant.total_distance
        return delta

class Ant:
    def __init__(self, colony, start_node):
        self.colony = colony
        self.tabu = [start_node]
        self.current = start_node
        self.path = []
        self.total_distance = 0
    
    def move_to(self, next_node):
        self.path.append((self.current, next_node))
        self.total_distance += self.colony.graph[self.current, next_node]
        self.tabu.append(next_node)
        self.current = next_node
    
    def is_complete(self):
        return len(self.tabu) == self.colony.n_nodes



def plot_initial_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')
    ax_button = plt.axes([0.7, 0.05, 0.15, 0.075])
    button = Button(ax_button, 'Find Path')
    status_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
    button.on_clicked(lambda event: plot_path(points, ax, status_text))
    plt.show()

def plot_path(points, ax, status_text):
    status_text.set_text("Computing path...")
    plt.draw()
    graph = generate_graph(points, directed=True)
    colony = AntColony(graph, n_ants=50, n_iterations=100)
    best_path, best_distance = colony.solve()
    path_indices = [edge[0] for edge in best_path] + [best_path[-1][1]]
    xs = [points[i][0] for i in path_indices]
    ys = [points[i][1] for i in path_indices]
    zs = [points[i][2] for i in path_indices]
    ax.plot(xs, ys, zs, c='b')
    status_text.set_text(f"Path length: {best_distance:.2f}")
    plt.draw()



def main():
    n_points = 200 
    points = generate_points(n_points)
    plot_initial_points(points)

if __name__ == "__main__":
    main()