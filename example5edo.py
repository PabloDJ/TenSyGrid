import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the system of equations
def system(t, y):
    y1, y2, y3, y4, y5 = y
    dy1_dt = -0.5 * y1 + 0.1 * y2
    dy2_dt = 0.5 * y1 - 0.2 * y2 + 0.1 * y3
    dy3_dt = -0.3 * y3 + 0.05 * y1 * y2
    dy4_dt = 0.4 * y4 - 0.1 * y5**2
    dy5_dt = -0.2 * y5 + 0.1 * y1 * y4
    return [dy1_dt, dy2_dt, dy3_dt, dy4_dt, dy5_dt]

# Initial conditions and time span
y_initial = [1, 0, 0, 0, 0]
t_span = (0, 50)  # From time t=0 to t=10

# Solve the system
sol = solve_ivp(system, t_span, y_initial, t_eval=np.linspace(0, 50, 100))

# Plotting the results
plt.plot(sol.t, sol.y.T)
plt.xlabel('Time')
plt.ylabel('Variables y1 to y5')
plt.title('Solution of the System of ODEs')
plt.legend(['y1', 'y2', 'y3', 'y4', 'y5'])
plt.grid(True)
plt.show()
