import numpy as np
import matplotlib.pyplot as plt

def magnetic_field(I, dir_I, r_i, r):
    mu_0 = 4 * np.pi * 1e-7
    r_diff = r - r_i
    r_mag = np.linalg.norm(r_diff)
    
    if r_mag == 0:
        return np.array([0, 0, 0])
    
    dL_cross_r = np.cross(dir_I, r_diff)
    B = (mu_0 * I / (4 * np.pi * r_mag**3)) * dL_cross_r
    return B

def straight_wire_field(I, L, r):
    mu_0 = 4 * np.pi * 1e-7
    return (mu_0 * I) / (2 * np.pi * r)

def loop_magnetic_field(I, R, r_values):
    mu_0 = 4 * np.pi * 1e-7
    return [(mu_0 * I * R**2) / (2 * (R**2 + r**2)**(3/2)) for r in r_values]

i_values = [10]
r_values = np.linspace(0.01, 1, 100)
R_loop = 0.1

B_direct = [straight_wire_field(i_values[0], 1, r) for r in r_values]
B_loop = loop_magnetic_field(i_values[0], R_loop, r_values)

L = 1.0
errors = []
for r in r_values:
    B_calc = magnetic_field(i_values[0], np.array([0, 1, 0]), np.array([0, 0, 0]), np.array([r, 0, 0]))
    B_theory = straight_wire_field(i_values[0], L, r)
    error = np.linalg.norm(B_calc) / B_theory - 1
    errors.append(abs(error))

threshold_distance = r_values[np.argmin(np.array(errors) < 0.01)]
print(f"Minimum distance with error less than 1%: {threshold_distance:.3f} m")

plt.figure(figsize=(10, 5))
plt.plot(r_values, B_direct, label="Direct wire", linestyle="-")
plt.plot(r_values, B_loop, label="Coil (axis)", linestyle="-")
plt.xlabel("Distance (m)")
plt.ylabel("Magnetic field (Tesla)")
plt.title("Magnetic field distribution")
plt.legend()
plt.grid()
plt.show()