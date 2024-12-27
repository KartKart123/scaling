import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import json
import numpy as np

# Load data from all uploaded files
file_paths = [
    "models/data_1",
    "models/data_2",
    "models/data_3",
    "models/data_4",
    "models/data_5",
    "models/data_6",
    "models/data_7",
    "models/data_8",
    "models/data_9",
    "models/data_10",
]

data_list = []
for file_path in file_paths:
    with open(file_path, 'r') as f:
        data = json.load(f)
        data_list.append(data)
# Function to format numbers in a compact form (e.g., 1M, 1K)
def format_params(num):
    if num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.0f}K"
    else:
        return str(num)

# Extract the number of parameters for each dataset
params_list = [data[0]['params'] for data in data_list]
log_params = np.log10(params_list)  # Use log scale for coloring

# Normalize the colors based on log(params)
norm = Normalize(vmin=min(log_params), vmax=max(log_params))
cmap = plt.cm.viridis  # Choose a colormap
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Necessary for creating the colorbar

# Re-plot with formatted parameter labels
fig, ax = plt.subplots(figsize=(12, 7))

for idx, data in enumerate(data_list):
    x = [entry['compute'] for entry in data]
    y = [entry['loss'] for entry in data]
    color = cmap(norm(log_params[idx]))
    formatted_params = format_params(params_list[idx])
    ax.plot(x, y, label=f"{formatted_params}", color=color)

# Set log scale for x-axis
ax.set_xscale("log")

# Add labels, legend, and grid
ax.set_xlabel("Compute (FLOPs)", fontsize=12)
ax.set_ylabel("Test Loss", fontsize=12)
ax.set_title("Efficient Compute Allocation", fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# Add colorbar with proper positioning
cbar = fig.colorbar(sm, ax=ax, label="Log10(Non-embedding parameters)", orientation='vertical')

fig2, ax2 = plt.subplots(figsize=(12, 7))

for idx, data in enumerate(data_list):
    x = [entry['tokens'] for entry in data]
    y = [entry['loss'] for entry in data]
    color = cmap(norm(log_params[idx]))
    formatted_params = format_params(params_list[idx])
    ax2.plot(x, y, label=f"{formatted_params}", color=color)

# Set log scale for x-axis
ax2.set_xscale("log")

# Add labels, legend, and grid
ax2.set_xlabel("Tokens", fontsize=12)
ax2.set_ylabel("Test Loss", fontsize=12)
ax2.set_title("Sample Efficiency", fontsize=14)
ax2.legend(loc='upper right')
ax2.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# Add colorbar with proper positioning
cbar2 = fig2.colorbar(sm, ax=ax2, label="Log10(Non-embedding parameters)", orientation='vertical')

fig3, ax3 = plt.subplots(figsize=(12, 7))

for idx, data in enumerate(data_list):
    x = [entry['iter'] for entry in data]
    y = [entry['loss'] for entry in data]
    color = cmap(norm(log_params[idx]))
    formatted_params = format_params(params_list[idx])
    ax3.plot(x, y, label=f"{formatted_params}", color=color)

# Add labels, legend, and grid
ax3.set_xlabel("Steps", fontsize=12)
ax3.set_ylabel("Train Loss", fontsize=12)
ax3.set_title("Training", fontsize=14)
ax3.legend(loc='upper right')
ax3.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# Add colorbar with proper positioning
cbar3 = fig3.colorbar(sm, ax=ax3, label="Log10(Non-embedding parameters)", orientation='vertical')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the updated plot
plt.show()