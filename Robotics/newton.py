import numpy as np

# 设置机械臂参数
l1, l2 = 1, 1  # 两个关节的长度
xt, yt = 1, 1  # 目标位置

# 正向运动学函数
def forward_kinematics(theta):
    x = l1 * np.cos(theta[0]) + l2 * np.cos(theta[0] + theta[1])
    y = l1 * np.sin(theta[0]) + l2 * np.sin(theta[0] + theta[1])
    return np.array([x, y])

# 误差函数
def error_function(theta):
    pos = forward_kinematics(theta)
    return np.sum((np.array([xt, yt]) - pos)**2)

# 计算雅可比矩阵（数值方法）
def jacobian(theta, h=1e-6):
    J = np.zeros((2, 2))
    for i in range(len(theta)):
        print(i)
        theta_plus = np.array(theta, copy=True)
        theta_plus[i] += h
        pos_plus = forward_kinematics(theta_plus)
        theta_minus = np.array(theta, copy=True)
        theta_minus[i] -= h
        pos_minus = forward_kinematics(theta_minus)
        J[:, i] = (pos_plus - pos_minus) / (2 * h)
    return J

# 牛顿优化方法
def newton_method(theta_init, max_iter=10):
    theta = theta_init
    for i in range(max_iter):
        pos = forward_kinematics(theta)
        error = np.array([xt, yt]) - pos
        J = jacobian(theta)
        # 使用伪逆计算更新步长
        theta_update = np.linalg.pinv(J).dot(error)
        theta += theta_update
        # 如果误差足够小则停止
        if np.linalg.norm(error) < 1e-5:
            break
    return theta

# 初始关节角度
theta_init = np.array([0.1, 0.1])
# 运行牛顿方法
theta_solution = newton_method(theta_init)
print("Found joint angles:", theta_solution)
print("Target position:", forward_kinematics(theta_solution))
