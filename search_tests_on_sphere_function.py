from simulated_annealing import simulated_annealing
from naive_random_search import naive_random_search
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt


def sphere_function(x, y):
    return x**2 + y**2

# Define bounds for x and y
bounds = [(-10, 10), (-10, 10)]

# Run Naive Random Search
nrs_result, nrs_trajectory = naive_random_search(sphere_function, bounds, 1000)[:3], naive_random_search(sphere_function, bounds, 1000)[3]
print("Naive Random Search Result:", nrs_result)

# Run Simulated Annealing
sa_result, sa_trajectory = simulated_annealing(sphere_function, (0, 0), bounds, 1000, 100, 0.99)[:3], simulated_annealing(sphere_function, (0, 0), bounds, 1000, 100, 0.99)[3]
print("Simulated Annealing Result:", sa_result)

def plot_function_with_trajectory(func, title, trajectories, bounds=(-10, 10), resolution=400):
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    
    # Plot each trajectory
    for trajectory, label in trajectories:
        traj_x, traj_y = zip(*trajectory)
        traj_z = func(np.array(traj_x), np.array(traj_y))
        ax.plot(traj_x, traj_y, traj_z, '.-', label=label)
    
    ax.set_title(title + ' 3D')
    ax.legend()
    
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=50, cmap='viridis')
    fig.colorbar(contour, ax=ax2)
    
    # Plot each trajectory
    for trajectory, label in trajectories:
        traj_x, traj_y = zip(*trajectory)
        ax2.plot(traj_x, traj_y, '.-', label=label)
    
    ax2.set_title(title + ' Contour')
    ax2.legend()
    
    plt.show()

# Plot functions with trajectories
plot_function_with_trajectory(sphere_function, "Sphere Function", [(nrs_trajectory, "Naive Random Search"), (sa_trajectory, "Simulated Annealing")])
