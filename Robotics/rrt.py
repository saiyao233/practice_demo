# import numpy as np
# import matplotlib.pyplot as plt

# class RRT:
#     def __init__(self, start, goal, step_size=0.1, num_iterations=1000):
#         self.start = start
#         self.goal = goal
#         self.step_size = step_size
#         self.num_iterations = num_iterations
#         self.tree = [start]
#         self.edges = []

#     def distance(self, point1, point2):
#         return np.linalg.norm(np.array(point1) - np.array(point2))

#     def nearest_vertex(self, random_point):
#         distances = [self.distance(vertex, random_point) for vertex in self.tree]
#         nearest_vertex = self.tree[np.argmin(distances)]
#         return nearest_vertex

#     def steer(self, nearest, random_point):
#         if self.distance(nearest, random_point) < self.step_size:
#             return random_point
#         else:
#             theta = np.arctan2(random_point[1] - nearest[1], random_point[0] - nearest[0])
#             return (nearest[0] + self.step_size * np.cos(theta), nearest[1] + self.step_size * np.sin(theta))

#     def is_goal_reached(self, point):
#         return self.distance(point, self.goal) <= self.step_size

#     def generate_path(self):
#         for _ in range(self.num_iterations):
#             random_point = (np.random.rand() * 2 * np.pi, np.random.rand() * 2 * np.pi)
#             nearest = self.nearest_vertex(random_point)
#             new_point = self.steer(nearest, random_point)

#             self.tree.append(new_point)
#             self.edges.append((nearest, new_point))

#             if self.is_goal_reached(new_point):
#                 print("Goal reached!")
#                 return self.tree, self.edges

#         print("Goal not reached within the maximum iterations.")
#         return self.tree, self.edges

#     def plot(self, tree, edges):
#         for edge in edges:
#             plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'ro-')
#         plt.plot([self.start[0]], [self.start[1]], 'go')  # Start in green
#         plt.plot([self.goal[0]], [self.goal[1]], 'bo')   # Goal in blue
#         plt.show()

# # Parameters
# start = (0, 0)  # Start angle of the joints
# goal = (np.pi, np.pi)  # Goal angle of the joints
# rrt = RRT(start, goal)
# tree, edges = rrt.generate_path()
# rrt.plot(tree, edges)

# import numpy as np
# import matplotlib.pyplot as plt

# class RRT:
#     def __init__(self, start, goal, step_size=0.1, num_iterations=10000):
#         self.start = start
#         self.goal = goal
#         self.step_size = step_size
#         self.num_iterations = num_iterations
#         self.tree = [start]
#         self.edges = []

#     def distance(self, point1, point2):
#         return np.linalg.norm(np.array(point1) - np.array(point2))

#     def nearest_vertex(self, random_point):
#         distances = [self.distance(vertex, random_point) for vertex in self.tree]
#         nearest_vertex = self.tree[np.argmin(distances)]
#         return nearest_vertex

#     def steer(self, nearest, random_point):
#         direction = np.subtract(random_point, nearest)
#         norm = np.linalg.norm(direction)
#         if norm < self.step_size:
#             return random_point
#         else:
#             return np.add(nearest, self.step_size * direction / norm)

#     def is_goal_reached(self, point):
#         return self.distance(point, self.goal) <= self.step_size

#     def generate_path(self):
#         for _ in range(self.num_iterations):
#             random_point = tuple(np.random.rand(4) * 2 * np.pi)
#             nearest = self.nearest_vertex(random_point)
#             new_point = self.steer(nearest, random_point)

#             self.tree.append(new_point)
#             self.edges.append((nearest, new_point))

#             if self.is_goal_reached(new_point):
#                 print("Goal reached!")
#                 return self.tree, self.edges

#         print("Goal not reached within the maximum iterations.")
#         return self.tree, self.edges

#     def plot(self, tree, edges):
#         for edge in edges:
#             plt.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], 'ro-')  # Joint 1 angles
#             plt.plot([edge[0][2], edge[1][2]], [edge[0][3], edge[1][3]], 'bo-')  # Joint 2 angles
#         plt.plot([self.start[0]], [self.start[1]], 'go')  # Start in green
#         plt.plot([self.goal[0]], [self.goal[1]], 'bo')   # Goal in blue
#         plt.show()

# # Parameters
# start = (0, 0, 0.5, 0.5)  # Start angle of the joints for both robots
# goal = (np.pi, np.pi, 1, 1)  # Goal angle of the joints for both robots
# rrt = RRT(start, goal)
# tree, edges = rrt.generate_path()
# rrt.plot(tree, edges)

import numpy as np
import matplotlib.pyplot as plt

class RRT:
    def __init__(self, start, goal, step_size=0.1, num_iterations=5000):
        self.start = start
        self.goal = goal
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.tree = [start]
        self.edges = []

    def distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def nearest_vertex(self, random_point):
        distances = [self.distance(vertex, random_point) for vertex in self.tree]
        nearest_vertex = self.tree[np.argmin(distances)]
        return nearest_vertex

    def steer(self, nearest, random_point):
        direction = np.subtract(random_point, nearest)
        norm = np.linalg.norm(direction)
        if norm < self.step_size:
            return random_point
        else:
            return np.add(nearest, self.step_size * direction / norm)

    def is_goal_reached(self, point):
        return self.distance(point, self.goal) <= self.step_size

    def generate_path(self):
        for _ in range(self.num_iterations):
            random_point = tuple(np.random.rand(4) * 2 * np.pi)
            nearest = self.nearest_vertex(random_point)
            new_point = self.steer(nearest, random_point)

            self.tree.append(new_point)
            self.edges.append((nearest, new_point))

            if self.is_goal_reached(new_point):
                print("Goal reached!")
                return self.tree, self.edges

        print("Goal not reached within the maximum iterations.")
        return self.tree, self.edges

    def plot(self, tree, edges):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for edge in edges:
            ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], [edge[0][2], edge[1][2]], 'ro-')
        ax.set_xlabel('Theta 1')
        ax.set_ylabel('Theta 2')
        ax.set_zlabel('Theta 3')
        plt.show()

# Parameters
start = (0, 0, 0, 0)  # Start angle of the joints
goal = (np.pi/2, np.pi/2, np.pi/2, np.pi/2)  # Goal angle of the joints
rrt = RRT(start, goal)
tree, edges = rrt.generate_path()
rrt.plot(tree, edges)
