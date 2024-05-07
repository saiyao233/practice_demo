import numpy as np

# Define robot parameters
link_lengths = [1.0, 1.0]  # Lengths of the arm segments
joint_limits = [(-np.pi, np.pi), (-np.pi/2, np.pi/2)]  # Joint limits
dt = 0.1  # Time step for simulation

# Initialize robot state
joint_angles = np.array([0, 0],dtype='float64')  # Initial joint angles
joint_velocities = np.array([0, 0],dtype='float64')  # Initial joint velocities

# Target position for the end effector
target_position = np.array([1.5, 0.5])

def inverse_kinematics(target, link_lengths):
    l1, l2 = link_lengths
    x, y = target
    cos_angle2 = (x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2)
    sin_angle2 = np.sqrt(1 - cos_angle2**2)
    angle2 = np.arctan2(sin_angle2, cos_angle2)
    
    k1 = l1 + l2 * cos_angle2
    k2 = l2 * sin_angle2
    angle1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    return np.array([angle1, angle2])

def model_predictive_control(current_angles, target_angles, current_velocities, n_steps=10, dt=0.1):
    # Predict future states over the horizon
    prediction_horizon = n_steps
    control_commands = np.zeros((prediction_horizon, len(current_angles)))
    predicted_angles = np.zeros((prediction_horizon + 1, len(current_angles)))
    predicted_angles[0] = current_angles

    # Simulate future states
    for i in range(prediction_horizon):
        # Simple proportional controller with damping (velocity control)
        control_input = 0.5 * (target_angles - predicted_angles[i]) - 0.1 * current_velocities
        predicted_angles[i + 1] = predicted_angles[i] + dt * current_velocities
        current_velocities += dt * control_input
        control_commands[i] = control_input

    # Cost function evaluation (sum of squared errors + effort)
    position_errors = np.linalg.norm(predicted_angles - target_angles, axis=1)
    control_efforts = np.linalg.norm(control_commands, axis=1)
    cost = np.sum(position_errors**2) + np.sum(control_efforts**2)

    # Return the first step of the optimal trajectory
    optimal_control = control_commands[0]
    return optimal_control

def forward_kinematics(link_lengths, joint_angles):
    l1, l2 = link_lengths
    theta1, theta2 = joint_angles
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return np.array([x, y])
# Main control loop
for _ in range(100):  # Run for a number of time steps
    # Compute the joint angles required to reach the target position
    desired_angles = inverse_kinematics(target_position, link_lengths)
    
    # Use MPC to optimize the trajectory
    control_commands = model_predictive_control(joint_angles, desired_angles, joint_velocities)
    
    # Apply the control commands
    joint_velocities += dt * control_commands
    joint_angles += dt * joint_velocities  # Update joint angles
    joint_angles = np.clip(joint_angles, joint_limits[0], joint_limits[1])  # Apply joint limits
    
    print("Joint Angles: ", joint_angles)
    print("End Effector Position: ", forward_kinematics(link_lengths, joint_angles))
    

    

