import numpy as np

def simulated_annealing(func, initial_point, bounds, max_iter, initial_temp, cooling_rate):
    current_x, current_y = initial_point
    current_value = func(current_x, current_y)
    temp = initial_temp
    trajectory = [(current_x, current_y)]
    
    for i in range(max_iter):
        next_x = current_x + np.random.uniform(-0.1, 0.1) * (bounds[0][1] - bounds[0][0])
        next_y = current_y + np.random.uniform(-0.1, 0.1) * (bounds[1][1] - bounds[1][0])
        next_value = func(next_x, next_y)
        trajectory.append((next_x, next_y))
        
        if next_value < current_value or np.random.random() < np.exp((current_value - next_value) / temp):
            current_x, current_y = next_x, next_y
            current_value = next_value
        
        temp *= cooling_rate
    
    return current_x, current_y, current_value, trajectory