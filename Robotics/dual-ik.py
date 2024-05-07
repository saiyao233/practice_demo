import numpy as np

def forward_kinematics(theta, l1, l2):
    x = l1 * np.cos(theta[0]) + l2 * np.cos(theta[0] + theta[1])
    y = l1 * np.sin(theta[0]) + l2 * np.sin(theta[0] + theta[1])
    return np.array([x, y])

def jacobian(theta, l1, l2):
    J = np.zeros((2, 2))
    J[0, 0] = -l1 * np.sin(theta[0]) - l2 * np.sin(theta[0] + theta[1])
    J[0, 1] = -l2 * np.sin(theta[0] + theta[1])
    J[1, 0] = l1 * np.cos(theta[0]) + l2 * np.cos(theta[0] + theta[1])
    J[1, 1] = l2 * np.cos(theta[0] + theta[1])
    return J

def ik_solver(arms, targets, l1, l2, alpha=0.1, max_iterations=1000, tolerance=1e-3):
    thetas = [np.array([0.0, 0.0]), np.array([0.0, 0.0])]  # Initial angles for both arms
    
    for _ in range(max_iterations):
        errors = []
        for i, arm in enumerate(arms):
            print(i,arm)
            current_pos = forward_kinematics(thetas[i], l1, l2)
            error = targets[i] - current_pos
            errors.append(np.linalg.norm(error))
            print(errors)
            if np.linalg.norm(error) < tolerance:
                continue
            J = jacobian(thetas[i], l1, l2)
            # Compute pseudo-inverse of Jacobian
            J_pinv = np.linalg.pinv(J)
            # Update joint angles
            thetas[i] += alpha * J_pinv @ error
        
        if all(e < tolerance for e in errors):
            print("Convergence reached")
            break

    return thetas

# Define arm lengths
l1, l2 = 1.0, 1.0
# Define target positions for both arms
targets = [np.array([1.5, 0.5]), np.array([1.5, -0.5])]

# Call the IK solver
thetas = ik_solver(['arm1', 'arm2'], targets, l1, l2)
print("Theta for Arm 1:", thetas[0])
print("Theta for Arm 2:", thetas[1])
