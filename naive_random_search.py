import numpy as np

def naive_random_search(func, bounds, iterations):
    best_x, best_y = None, None
    best_value = float('inf')
    trajectory = []
    
    for _ in range(iterations):
        x = np.random.uniform(bounds[0][0], bounds[0][1])
        y = np.random.uniform(bounds[1][0], bounds[1][1])
        value = func(x, y)
        trajectory.append((x, y))
        if value < best_value:
            best_value = value
            best_x, best_y = x, y
    
    return best_x, best_y, best_value, trajectory