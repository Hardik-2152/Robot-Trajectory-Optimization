import numpy as np
import matplotlib.pyplot as plt

T = 5.0
t = np.linspace(0, T, 100)

q1_start, q1_end = 0, 45
q2_start, q2_end = 0, -90

q1_linear = q1_start + (q1_end - q1_start) * (t / T)
q2_linear = q2_start + (q2_end - q2_start) * (t / T)

def cubic_trajectory(q_start, q_end, t, T):
    a0 = q_start
    a1 = 0
    a2 = 3 * (q_end - q_start) / (T**2)
    a3 = -2 * (q_end - q_start) / (T**3)
    return a0 + a1*t + a2*t**2 + a3*t**3

q1_cubic = cubic_trajectory(q1_start, q1_end, t, T)
q2_cubic = cubic_trajectory(q2_start, q2_end, t, T)

plt.figure()
plt.plot(t, q1_linear, label="q1 Linear")
plt.plot(t, q1_cubic, label="q1 Cubic")
plt.xlabel("Time (s)")
plt.ylabel("Joint Angle (deg)")
plt.title("Joint 1 Trajectory")
plt.legend()
plt.grid()

plt.figure()
plt.plot(t, q2_linear, label="q2 Linear")
plt.plot(t, q2_cubic, label="q2 Cubic")
plt.xlabel("Time (s)")
plt.ylabel("Joint Angle (deg)")
plt.title("Joint 2 Trajectory")
plt.legend()
plt.grid()

plt.show()
