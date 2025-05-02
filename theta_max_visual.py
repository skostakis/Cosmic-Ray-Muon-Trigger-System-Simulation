import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
d = 1.0  # Vertical separation between plates (meters)
side = 1.0  # Plate side length (meters)
half_side = side / 2

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the two plates (top and bottom)
X_top, Y_top = np.meshgrid([-half_side, half_side], [-half_side, half_side])
ax.plot_surface(X_top, Y_top, np.zeros_like(X_top), alpha=0.5, color='blue')
ax.plot_surface(X_top, Y_top, -d * np.ones_like(X_top), alpha=0.5, color='red')

def plot_ray(ax, x0, y0, theta_deg, phi_deg, d, force_color=None):
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)
    
    t = np.linspace(0, d, 100)
    x = x0 + t * np.tan(theta) * np.cos(phi)
    y = y0 + t * np.tan(theta) * np.sin(phi)
    z = -t
    
    hit = (np.abs(x[-1]) <= half_side) & (np.abs(y[-1]) <= half_side)
    
    # Use force_color if provided, otherwise default to green/red
    color = force_color if force_color is not None else ('green' if hit else 'red')
    
    return ax.plot(x, y, z, color=color, linewidth=1.5)

# Correct θ_max using FULL DIAGONAL / d
full_diag = np.sqrt((side)**2 + (side)**2)
theta_max_deg = np.degrees(np.arctan(full_diag / d))

ray_lines = []
labels = []

# Ray at θ_max (blue color)
line = plot_ray(ax, x0=0.5, y0=0.5, theta_deg=theta_max_deg, phi_deg=225, d=d, force_color='blue')
ray_lines.append(line[0])
labels.append(f'θ = {theta_max_deg:.1f}° (Max Angle)')

# Ray exceeding θ_max (red)
line = plot_ray(ax, x0=0.5, y0=0.5, theta_deg=theta_max_deg + 5, phi_deg=225, d=d)
ray_lines.append(line[0])
labels.append(f'θ = {theta_max_deg + 5:.1f}° (Miss)')

# Vertical ray (green)
line = plot_ray(ax, x0=0, y0=0, theta_deg=0, phi_deg=0, d=d)
ray_lines.append(line[0])
labels.append('θ = 0° (Vertical)')

ax.legend(ray_lines, labels, loc='upper left')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_xlim(-half_side, half_side)
ax.set_ylim(-half_side, half_side)
ax.set_zlim(-d, 0)
ax.set_title(f'Trigger System Simulation (at z = 1m)')
ax.view_init(elev=20, azim=-45)
plt.tight_layout()
plt.show()
