import numpy as np
from scipy.optimize import minimize

# 定义机器人的两个连杆长度
l1, l2 = 1.0, 1.0

# 正运动学：根据关节角度计算末端执行器位置
def forward_kinematics(theta):
    x = l1 * np.cos(theta[0]) + l2 * np.cos(theta[0] + theta[1])
    y = l1 * np.sin(theta[0]) + l2 * np.sin(theta[0] + theta[1])
    return np.array([x, y])

# 逆运动学：根据末端位置求解关节角度
def inverse_kinematics(target):
    def ik_objective(theta):
        pos = forward_kinematics(theta)
        return np.linalg.norm(pos - target)
    
    initial_guess = np.array([0, 0])  # 逆运动学的初始猜测
    result = minimize(ik_objective, initial_guess, method='BFGS')
    return result.x

# 目标函数：最小化关节角速度总和（作为示例）
def trajectory_objective(theta, start_angles):
    # 加入了平滑运动的目标，这里简单地使用角度差的绝对值总和
    return np.sum(np.abs(theta - start_angles))

# 轨迹优化函数
def optimize_trajectory(start_angles, goal):
    # cons = {'type': 'ineq', 'fun': joint_limits_constraint}
    bounds = [(-np.pi, np.pi), (-np.pi/2, np.pi/2)]  # 关节角度限制
    result = minimize(trajectory_objective, start_angles, args=(start_angles,), bounds=bounds)
    return result

# 运行逆运动学和轨迹优化
target_position = np.array([1.5, 1.5])  # 目标末端位置
start_angles = inverse_kinematics(target_position)  # 逆运动学解
optimized_result = optimize_trajectory(start_angles, target_position)

print("Initial Joint Angles from IK:", start_angles)
print("Optimized Joint Angles:", optimized_result.x)
print("Final Position:", forward_kinematics(optimized_result.x))
