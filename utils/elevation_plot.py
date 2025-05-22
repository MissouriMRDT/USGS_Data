import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
import sys
import matplotlib

matplotlib.use('TkAgg')

if len(sys.argv) != 2:
    print("Usage: python compare_roughness.py <file.csv>")
    sys.exit(1)

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

# Validate columns
required_columns = {'GridX', 'GridY', 'CellRoughness', 'TerrainRoughness'}
if not required_columns.issubset(df.columns):
    print("CSV must contain: GridX, GridY, CellRoughness, TerrainRoughness")
    sys.exit(1)

# Extract coordinates and values
x = df['GridX'].values
y = df['GridY'].values
z1 = df['CellRoughness'].values
z2 = df['TerrainRoughness'].values

# Create regular grid
grid_x, grid_y = np.meshgrid(
    np.linspace(x.min(), x.max(), 100),
    np.linspace(y.min(), y.max(), 100)
)

# Interpolate scattered data onto grid
Z1 = griddata((x, y), z1, (grid_x, grid_y), method='linear')
Z2 = griddata((x, y), z2, (grid_x, grid_y), method='linear')
Z_diff = Z1 - Z2

# --- Plotting ---
fig = plt.figure(figsize=(20, 15))

# 3D Plot: CellRoughness
ax1 = fig.add_subplot(3, 2, 1, projection='3d')
surf1 = ax1.plot_surface(grid_x, grid_y, Z1, cmap=cm.terrain, edgecolor='none')
ax1.set_title("Cell Roughness (3D)")
ax1.set_xlabel('GridX'); ax1.set_ylabel('GridY'); ax1.set_zlabel('Roughness')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# 2D Heatmap: CellRoughness
ax2 = fig.add_subplot(3, 2, 2)
heat1 = ax2.imshow(Z1, cmap='terrain', origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   aspect='auto')
ax2.set_title("Cell Roughness (Heatmap)")
ax2.set_xlabel("GridX"); ax2.set_ylabel("GridY")
fig.colorbar(heat1, ax=ax2, shrink=0.5)

# 3D Plot: TerrainRoughness
ax3 = fig.add_subplot(3, 2, 3, projection='3d')
surf2 = ax3.plot_surface(grid_x, grid_y, Z2, cmap=cm.viridis, edgecolor='none')
ax3.set_title("Terrain Roughness (3D)")
ax3.set_xlabel('GridX'); ax3.set_ylabel('GridY'); ax3.set_zlabel('Roughness')
fig.colorbar(surf2, ax=ax3, shrink=0.5)

# 2D Heatmap: TerrainRoughness
ax4 = fig.add_subplot(3, 2, 4)
heat2 = ax4.imshow(Z2, cmap='viridis', origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   aspect='auto')
ax4.set_title("Terrain Roughness (Heatmap)")
ax4.set_xlabel("GridX"); ax4.set_ylabel("GridY")
fig.colorbar(heat2, ax=ax4, shrink=0.5)

# 3D Plot: Difference
ax5 = fig.add_subplot(3, 2, 5, projection='3d')
surf3 = ax5.plot_surface(grid_x, grid_y, Z_diff, cmap=cm.coolwarm, edgecolor='none')
ax5.set_title("Difference (Cell - Terrain) (3D)")
ax5.set_xlabel('GridX'); ax5.set_ylabel('GridY'); ax5.set_zlabel('Δ Roughness')
fig.colorbar(surf3, ax=ax5, shrink=0.5)

# 2D Heatmap: Difference
ax6 = fig.add_subplot(3, 2, 6)
heat3 = ax6.imshow(Z_diff, cmap='coolwarm', origin='lower',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   aspect='auto')
ax6.set_title("Difference (Heatmap)")
ax6.set_xlabel("GridX"); ax6.set_ylabel("GridY")
fig.colorbar(heat3, ax=ax6, shrink=0.5)

plt.tight_layout()
plt.savefig("roughness_comparison.png", dpi=300)
print("✅ Saved as roughness_comparison.png")

plt.show()
