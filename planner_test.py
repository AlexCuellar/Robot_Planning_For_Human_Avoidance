from planner import Planner
import numpy as np
from visualize_environment import visualize

T = 20 # timesteps
T_length = .5 # Time length for each timestep
A = np.identity(4)
A[0, 2] = 1*T_length
A[1, 3] = 1*T_length
B = np.zeros((4, 2))
B[2, 0] = 1*T_length
B[3, 1] = 1*T_length

x0_bar = np.array([0, 0, 0, 0]) # The mean of x at timestep 0
Sigma_x0 = np.zeros((4, 4)) # Covariance matrix of x at timestep 0
Sigma_x0[0, 0] = .05
Sigma_x0[1, 1] = .05
Sigma_x0[2, 2] = .001
Sigma_x0[3, 3] = .001
Sigma_w = 0.00001*np.identity(4)# Noise covariance added to x at each timestep
Delta = 0.1 # Joint / overall chance constraint
xf_bar = np.array([10, 10, 0, 0]) # Final state
state_bounds = [(-40, 40), (-40, 40), (-15, 15), (-15, 15)] # Bounds for our state vector
input_bounds = [(-1.2, 1.2), (-1.2, 1.2)] # Bounds for our input vector

p = Planner(T, 4, 2, T_length, A, B, x0_bar, xf_bar, Sigma_x0, Sigma_w, Delta, state_bounds, input_bounds)

N = 15 # Num trajectories
data = np.zeros((N, T + 1, 2)) # array for trajectories
start = (2, 8) # Start point on dummy trajectories
travel = (7, -7) # x and y movement along each direction
# Genearte trajectories
for t in range(T + 1):
    for n in range(N):
        mean = np.array([start[0] + travel[0]*t/T, start[1] + travel[1]*t/T])
        cov = np.array([[1, 0], [0, 1]])
        data[n, t, :] = np.random.multivariate_normal(mean, cov)

# Generate objects from our dummy trajectories
p.include_trajectories(data)
# Plan with Delta = .1
p.plan_with_IRA()
x1 = p.get_x()

# Plan with Delta = .4
p.update_delta(.4)
p.reset_rmpc()
p.include_trajectories(data)
p.plan_with_IRA()
x2 = p.get_x()

# Plot both planned trajectories
objs = p.get_objects()
xs = [x1, x2]
colors = ["r", "g"]
visualize(xs, objs, colors)
