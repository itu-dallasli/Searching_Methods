import time
import numpy as np
from agent import *

# Emir Arda Eker
# 150220331
# I added comments when necessary
# I couldn't find any other than Euclidian and Manhattan than learned a heuristic funtion named Octile from 3rd party resources :)

class BFSAgent(Agent):
    def __init__(self, matrix):
        """
            Initializes the BFS agent class.

            Args:
                matrix (list): Initial game matrix
        """
        # Initializing the parent class
        super().__init__(matrix)

    def tree_solve(self):
        """
            Solves the game using tree-based BFS algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        self.frontier = []  # Frontier set initialization by the initial state

        initial_node = Node(None,self.initial_matrix,None,0,0) # Null values for *g_score* and *h_score* creates type error so even if we don't use , I set to 0
        self.frontier.append(initial_node)

        yb, xb = len(self.initial_matrix[0]), len(self.initial_matrix[1])

        while self.frontier:

            self.maximum_node_in_memory = max(self.maximum_node_in_memory,len(self.frontier))

            current_node = self.frontier.pop(0)  # 

            self.explored_node += 1

            if self.check_value(current_node.matrix, self.desired_value):  # Checking if we have finished
                return self.get_moves(current_node)  # Solution moves 

            p1_p,p2_p = self.find_players_positions(current_node.matrix)  # 

            new_matrix = self.copy_matrix(current_node.matrix) 
            new_matrix[p1_p[0]][p1_p[1]] = 0
            new_matrix[p2_p[0]][p2_p[1]] = 0

            for action in self.actions:
                try:
                    new_x = p1_p[0] + action[0]
                    new_y = p1_p[1] + action[1]

                    if 0 <= new_x < len(current_node.matrix) and 0 <= new_y < len(current_node.matrix[0]) and current_node.matrix[new_x][new_y] != 4:
                        added_matrix = self.copy_matrix(new_matrix)
                        tmp_p1, tmp_p2 = self.find_players_positions(current_node.matrix)

                        tmp_p1[0] = new_x
                        tmp_p1[1] = new_y

                        # Determine the new position for player 2 based on the opposite action
                        p2_action = [-action[0], -action[1]]
                        obstacle_x = p2_p[0] + p2_action[0]
                        obstacle_y = p2_p[1] + p2_action[1]

                        if 0 <= obstacle_x < len(current_node.matrix) and 0 <= obstacle_y < len(current_node.matrix[0]) and current_node.matrix[obstacle_x][obstacle_y] != 4:
                            tmp_p2[0] = obstacle_x
                            tmp_p2[1] = obstacle_y

                        if tmp_p1 == tmp_p2:
                            added_matrix[tmp_p1[0]][tmp_p1[1]] = 3
                        else:
                            added_matrix[tmp_p1[0]][tmp_p1[1]] = self.player1
                            added_matrix[tmp_p2[0]][tmp_p2[1]] = self.player2

                        next_node = Node(current_node, added_matrix, action, 0, 0)
                        self.frontier.append(next_node)
                        self.generated_node += 1

                except IndexError:
                     pass

    def graph_solve(self):
        """
            Solves the game using graph-based BFS algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        """
            Solves the game using tree-based BFS algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        self.frontier = []  # Frontier set initialization by the initial state
        self.explored = [] 

        initial_node = Node(None, self.initial_matrix, None, 0, 0)

        self.frontier.append(initial_node)  # First node creation of Adam
        yb, xb = len(self.initial_matrix[0]), len(self.initial_matrix[1])  # Boundary recognition

        while self.frontier:

            self.maximum_node_in_memory = max(self.maximum_node_in_memory, len(self.explored))

            current_node = self.frontier.pop(0)  

            self.explored_node += 1

            if self.check_value(current_node.matrix, self.desired_value):  # Checking if we have finished
                return self.get_moves(current_node)  # Solution moves 

            p1_p, p2_p = self.find_players_positions(current_node.matrix)  

            new_matrix = self.copy_matrix(current_node.matrix)  # Mt matrix for keeping
            new_matrix[p1_p[0]][p1_p[1]] = 0
            new_matrix[p2_p[0]][p2_p[1]] = 0

            for action in self.actions:
                try:
                    new_x = p1_p[0] + action[0]
                    new_y = p1_p[1] + action[1]

                    if 0 <= new_x < len(current_node.matrix) and 0 <= new_y < len(current_node.matrix[0]) and \
                            current_node.matrix[new_x][new_y] != 4:
                        added_matrix = self.copy_matrix(new_matrix)
                        tmp_p1, tmp_p2 = self.find_players_positions(current_node.matrix)

                        tmp_p1[0] = new_x
                        tmp_p1[1] = new_y

                        # Determine the new position for player 2 based on the opposite action
                        p2_action = [-action[0], -action[1]]
                        obstacle_x = p2_p[0] + p2_action[0]
                        obstacle_y = p2_p[1] + p2_action[1]

                        if 0 <= obstacle_x < len(current_node.matrix) and 0 <= obstacle_y < len(
                                current_node.matrix[0]) and current_node.matrix[obstacle_x][obstacle_y] != 4:
                            tmp_p2[0] = obstacle_x
                            tmp_p2[1] = obstacle_y

                        if tmp_p1 == tmp_p2:
                            added_matrix[tmp_p1[0]][tmp_p1[1]] = 3
                        else:
                            added_matrix[tmp_p1[0]][tmp_p1[1]] = self.player1
                            added_matrix[tmp_p2[0]][tmp_p2[1]] = self.player2

                        next_node = Node(current_node, added_matrix, action, 0, 0)

                        checked = False
                        for item in self.explored:
                            if self.check_equal(item.matrix,
                                                next_node.matrix):  # Check for duplicate states in the explored list
                                checked = True

                        if not checked:  # If the state is not in the explored list, add it to frontier and explored
                            self.frontier.append(next_node)
                            self.explored.append(next_node)
                            self.generated_node += 1

                except IndexError:
                    pass


class DFSAgent(Agent):
    def __init__(self, matrix):
        """
            Initializes the DFS agent class.

            Args:
                matrix (list): Initial game matrix
        """
        # Initializing the parent class
        super().__init__(matrix)

    def tree_solve(self):
        """
            Solves the game using tree-based DFS algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        self.frontier = []  # Frontier set initialization by the initial state

        initial_node = Node(None, self.initial_matrix, None, 0, 0)
        self.frontier.append(initial_node)  # First node creation of Adam
        yb, xb = len(self.initial_matrix[0]), len(
            self.initial_matrix[1])  # Boundary setting

        while self.frontier:
            print(self.generated_node)
            self.maximum_node_in_memory = max(self.maximum_node_in_memory, len(self.frontier))

            current_node = self.frontier.pop()  # pop(0) is for BFS that is for DFS

            self.explored_node += 1

            if self.check_value(current_node.matrix,
                                self.desired_value):  # Checking if we have finished
                return self.get_moves(current_node)  # Solution moves 

            p1_p, p2_p = self.find_players_positions(current_node.matrix)  # 

            new_matrix = self.copy_matrix(
                current_node.matrix)  # Mt matrix for keeping
            new_matrix[p1_p[0]][p1_p[1]] = 0
            new_matrix[p2_p[0]][p2_p[1]] = 0

            # down,
            for action in self.actions:
                try:
                    new_x = p1_p[0] + action[0]
                    new_y = p1_p[1] + action[1]

                    if 0 <= new_x < len(current_node.matrix) and 0 <= new_y < len(current_node.matrix[0]) and \
                            current_node.matrix[new_x][new_y] != 4:
                        added_matrix = self.copy_matrix(new_matrix)
                        tmp_p1, tmp_p2 = self.find_players_positions(current_node.matrix)

                        tmp_p1[0] = new_x
                        tmp_p1[1] = new_y

                        # Determine the new position for player 2 based on the opposite action
                        p2_action = [-action[0], -action[1]]
                        obstacle_x = p2_p[0] + p2_action[0]
                        obstacle_y = p2_p[1] + p2_action[1]

                        if 0 <= obstacle_x < len(current_node.matrix) and 0 <= obstacle_y < len(
                                current_node.matrix[0]) and current_node.matrix[obstacle_x][obstacle_y] != 4:
                            tmp_p2[0] = obstacle_x
                            tmp_p2[1] = obstacle_y

                        if tmp_p1 == tmp_p2:
                            added_matrix[tmp_p1[0]][tmp_p1[1]] = 3
                        else:
                            added_matrix[tmp_p1[0]][tmp_p1[1]] = self.player1
                            added_matrix[tmp_p2[0]][tmp_p2[1]] = self.player2

                        next_node = Node(current_node, added_matrix, action, 0, 0)
                        self.frontier.append(next_node)
                        self.generated_node += 1

                except IndexError:
                    pass

    def graph_solve(self):
        """
            Solves the game using graph-based DFS algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        self.frontier = []  # Frontier set initialization by the initial state
        self.explored = []  # initialize the explored set

        initial_node = Node(None, self.initial_matrix, None, 0, 0)

        self.frontier.append(initial_node)  #First node creation of Ada
        yb, xb = len(self.initial_matrix[0]), len(self.initial_matrix[1])  # Boundary setting

        while self.frontier:

            self.maximum_node_in_memory = max(self.maximum_node_in_memory, len(self.explored))

            current_node = self.frontier.pop()  # take the last element (stack), only difference from BFS

            self.explored_node += 1

            if self.check_value(current_node.matrix,
                                self.desired_value):  # Checking if we have finished
                return self.get_moves(current_node)  # Solution moves 

            p1_p, p2_p = self.find_players_positions(current_node.matrix)  # 

            new_matrix = self.copy_matrix(
                current_node.matrix)  # Mt matrix for keeping
            new_matrix[p1_p[0]][p1_p[1]] = 0
            new_matrix[p2_p[0]][p2_p[1]] = 0

            for action in self.actions:
                try:
                    new_x = p1_p[0] + action[0]
                    new_y = p1_p[1] + action[1]

                    if 0 <= new_x < len(current_node.matrix) and 0 <= new_y < len(current_node.matrix[0]) and \
                            current_node.matrix[new_x][new_y] != 4:
                        added_matrix = self.copy_matrix(new_matrix)
                        tmp_p1, tmp_p2 = self.find_players_positions(current_node.matrix)

                        tmp_p1[0] = new_x
                        tmp_p1[1] = new_y

                        # Determine the new position for player 2 based on the opposite action
                        p2_action = [-action[0], -action[1]]
                        obstacle_x = p2_p[0] + p2_action[0]
                        obstacle_y = p2_p[1] + p2_action[1]

                        if 0 <= obstacle_x < len(current_node.matrix) and 0 <= obstacle_y < len(
                                current_node.matrix[0]) and current_node.matrix[obstacle_x][obstacle_y] != 4:
                            tmp_p2[0] = obstacle_x
                            tmp_p2[1] = obstacle_y

                        if tmp_p1 == tmp_p2:
                            added_matrix[tmp_p1[0]][tmp_p1[1]] = 3
                        else:
                            added_matrix[tmp_p1[0]][tmp_p1[1]] = self.player1
                            added_matrix[tmp_p2[0]][tmp_p2[1]] = self.player2

                        next_node = Node(current_node, added_matrix, action, 0, 0)

                        checked = False
                        for item in self.explored:
                            if self.check_equal(item.matrix,
                                                next_node.matrix):  # Check for duplicate states in the explored list
                                checked = True

                        if not checked:  # If the state is not in the explored list, add it to frontier and explored
                            self.frontier.append(next_node)
                            self.explored.append(next_node)
                            self.generated_node += 1

                except IndexError:
                    pass

class AStarAgent(Agent):
    def __init__(self, matrix):
        """
            Initializes the A* agent class.

            Args:
                matrix (list): Initial game matrix
        """
        super().__init__(matrix)
        
    def mnhtn(self, p1_p, p2_p):  #  Manhattan distance h function
        distance = abs(p1_p[0]-p2_p[0])+abs(p2_p[1]-p2_p[0])
        return distance

    def octile(self, p1_p, p2_p):  # Octile distance h function
        dx = abs(p1_p[0] - p2_p[0])
        dy = abs(p1_p[1] - p2_p[1])
        distance = max(dx, dy) + (2**(1/2) - 1) * min(dx, dy)
        return distance

    def euclid(self, p1_p, p2_p):  #  Euclidian distance h function
        distance = ((p1_p[0]-p2_p[0])**2+(p1_p[1]-p2_p[1])**2)**1/2
        return distance
    
    def tree_solve(self):
        """
            Solves the game using tree-based A* algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        self.frontier = PriorityQueue()  # Frontier setting

        initial_node = Node(None, self.initial_matrix, None, 0, 0)

        self.frontier.push(initial_node,0)  # First node creation of Adam

        yb, xb = len(self.initial_matrix[0]), len(self.initial_matrix[1])  # Boundary setting

        while not self.frontier.is_empty():

            self.maximum_node_in_memory = max(self.maximum_node_in_memory, self.frontier.size())

            current_node = self.frontier.pop()  
            step_cost = current_node.g_score  
            self.explored_node += 1

            if self.check_value(current_node.matrix,self.desired_value):  # Checking if we have finished
                return self.get_moves(current_node)  # Solution moves 

            p1_p, p2_p = self.find_players_positions(current_node.matrix)  

            new_matrix = self.copy_matrix(current_node.matrix)  # Mt matrix for keeping
            new_matrix[p1_p[0]][p1_p[1]] = 0
            new_matrix[p2_p[0]][p2_p[1]] = 0

            for action in self.actions:
                try:
                    new_x = p1_p[0] + action[0]
                    new_y = p1_p[1] + action[1]

                    if 0 <= new_x < len(current_node.matrix) and 0 <= new_y < len(current_node.matrix[0]) and \
                            current_node.matrix[new_x][new_y] != 4:
                        added_matrix = self.copy_matrix(new_matrix)
                        tmp_p1, tmp_p2 = self.find_players_positions(current_node.matrix)

                        tmp_p1[0] = new_x
                        tmp_p1[1] = new_y

                        # Determine the new position for player 2 based on the opposite action
                        p2_action = [-action[0], -action[1]]
                        obstacle_x = p2_p[0] + p2_action[0]
                        obstacle_y = p2_p[1] + p2_action[1]

                        if 0 <= obstacle_x < len(current_node.matrix) and 0 <= obstacle_y < len(
                                current_node.matrix[0]) and current_node.matrix[obstacle_x][obstacle_y] != 4:
                            tmp_p2[0] = obstacle_x
                            tmp_p2[1] = obstacle_y

                        if tmp_p1 == tmp_p2:
                            added_matrix[tmp_p1[0]][tmp_p1[1]] = 3
                        else:
                            added_matrix[tmp_p1[0]][tmp_p1[1]] = self.player1
                            added_matrix[tmp_p2[0]][tmp_p2[1]] = self.player2

                        h_score = self.mnhtn(tmp_p1, tmp_p2)

                        next_node = Node(current_node, added_matrix, [1, 0], step_cost + 1, h_score)  

                        priority = next_node.f_score
                        self.frontier.push(next_node, priority)  
                        self.generated_node += 1

                except IndexError:
                    pass

    def graph_solve(self):
        """
            Solves the game using graph-based A* algorithm.

            Returns:
                list: A list of tuples containing moves and state matrices in the solution.
        """
        self.frontier = PriorityQueue()  # Frontier setting
        self.explored = []
        initial_node = Node(None, self.initial_matrix, None, 0, 0)
        self.frontier.push(initial_node, 0)  # First node creation of Adam
        yb, xb = len(self.initial_matrix[0]), len(self.initial_matrix[1])  ## Boundary setting

        while self.frontier.is_empty() == 0:

            self.maximum_node_in_memory = max(self.maximum_node_in_memory, len(self.explored))

            current_node = self.frontier.pop()

            ###

            step_cost = current_node.g_score # We define a stepcost.

            ###
            
            self.explored.append(current_node)
            self.explored_node += 1

            if self.check_value(current_node.matrix,self.desired_value):
                return self.get_moves(current_node)

            p1_p, p2_p = self.find_players_positions(current_node.matrix)

            #Defining a new matrix
            new_matrix = self.copy_matrix(current_node.matrix)
            new_matrix[p1_p[0]][p1_p[1]] = 0
            new_matrix[p2_p[0]][p2_p[1]] = 0

            for action in self.actions:
                try:
                    new_x = p1_p[0] + action[0]
                    new_y = p1_p[1] + action[1]

                    if 0 <= new_x < len(current_node.matrix) and 0 <= new_y < len(current_node.matrix[0]) and \
                            current_node.matrix[new_x][new_y] != 4:
                        added_matrix = self.copy_matrix(new_matrix)
                        tmp_p1, tmp_p2 = self.find_players_positions(current_node.matrix)

                        tmp_p1[0] = new_x
                        tmp_p1[1] = new_y

                        p2_action = [-action[0], -action[1]]
                        obstacle_x = p2_p[0] + p2_action[0]
                        obstacle_y = p2_p[1] + p2_action[1]

                        if 0 <= obstacle_x < len(current_node.matrix) and 0 <= obstacle_y < len(
                                current_node.matrix[0]) and current_node.matrix[obstacle_x][obstacle_y] != 4:
                            tmp_p2[0] = obstacle_x
                            tmp_p2[1] = obstacle_y

                        if tmp_p1 == tmp_p2:
                            added_matrix[tmp_p1[0]][tmp_p1[1]] = 3
                        else:
                            added_matrix[tmp_p1[0]][tmp_p1[1]] = self.player1
                            added_matrix[tmp_p2[0]][tmp_p2[1]] = self.player2

                        h_score = self.mnhtn(tmp_p1, tmp_p2)

                        next_node = Node(current_node, added_matrix, [1, 0], step_cost + 1,h_score)  

                        checked = False
                        for item in self.explored:
                            if self.check_equal(item.matrix,next_node.matrix):  
                                checked = True
                                
                        if not checked:
                            priority = next_node.f_score
                            self.frontier.push(next_node, priority)  
                            self.generated_node += 1

                except IndexError:
                    pass
