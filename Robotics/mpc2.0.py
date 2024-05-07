import numpy as np
from casadi import *

# 定义机械臂参数
L1, L2, L3 = 1.0, 1.0, 1.0  # 链长
theta1, theta2, theta3 = 0, 0, 0  # 初始关节角度

# 逆运动学函数：计算达到目标点的关节角度
def inverse_kinematics(x, y, z):
    theta3 = np.arccos((x**2 + y**2 + z**2 - L1**2 - L2**2 - L3**2) / (2 * L1 * L2))
    theta2 = np.arctan2(z, np.sqrt(x**2 + y**2)) - np.arcsin((L3 * np.sin(theta3)) / np.sqrt(x**2 + y**2 + z**2))
    theta1 = np.arctan2(y, x)
    return theta1, theta2, theta3

# 目标位置
target_x, target_y, target_z = 1.0, 1.0, 1.0
theta1, theta2, theta3 = inverse_kinematics(target_x, target_y, target_z)
print(theta1, theta2, theta3)
# 模型预测控制设置
N = 10  # 预测步长
T = 1.0  # 总控制时间
dt = T / N  # 步长

# 状态变量
theta = SX.sym('theta', 3)
theta_dot = SX.sym('theta_dot', 3)
print(theta,theta_dot)
# # 控制变量
# u = SX.sym('u', 3)

# # 状态方程
# state = vertcat(theta, theta_dot)
# dynamics = vertcat(theta_dot, u)

# # 目标函数和约束
# Q = np.diag([1000, 1000, 1000])
# R = np.diag([0.1, 0.1, 0.1])
# obj = 0  # 初始化目标函数
# g = []  # 初始化约束条件

# # 初始状态
# x0 = np.array([0, 0, 0, 0, 0, 0])

# # 循环构建优化问题
# for i in range(N):
#     obj += mtimes([(theta - np.array([theta1, theta2, theta3])), Q, (theta - np.array([theta1, theta2, theta3]))]) + mtimes([u, R, u])
#     x_next = state + dynamics * dt
#     state = x_next
#     theta = state[:3]
#     theta_dot = state[3:]
#     g.append(theta_dot)  # 添加速度约束

# # 优化问题设置
# OPT_variables = vertcat(theta, theta_dot, u)
# nlp_problem = {'f': obj, 'x': OPT_variables, 'g': vertcat(*g)}
# solver = nlpsol('solver', 'ipopt', nlp_problem)

# # 解决优化问题
# sol = solver(lbx=-np.inf, ubx=np.inf, lbg=-10, ubg=10, x0=x0)
# optimal_theta = sol['x']

# print("Optimized Joint Angles:", optimal_theta)
