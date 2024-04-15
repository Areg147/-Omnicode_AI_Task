import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plotter(array, ranges, image, dictionary, colors, labels, FRAME):
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    fig.suptitle(f'Frame {FRAME}', fontsize=20)

    # Histogram for temperature
    array = array[array >= 50]
    n, bins, patches = axes[0].hist(array[array != 0], bins=np.arange(min(array), max(array) + 1), color='blue', alpha=0.7)
    axes[0].set_xlim(0, 400)
    axes[0].set_ylim(0, 2000)
    axes[0].set_xlabel("temperature in celsius")
    axes[0].set_ylabel("frequency")
    axes[0].set_title("Stick Temperature Distribution")

    # Adding custom legend for specified ranges
    legend_elements = [
        Patch(facecolor='red', edgecolor='r', label='50-100'),
        Patch(facecolor='blue', edgecolor='b', label='100-150')
    ]
    axes[0].legend(handles=legend_elements, loc='upper left', title="Temperature Ranges")

# Example usage
array = np.random.randint(50, 200, size=1000)  # Random data for demonstration
plotter(array, None, None, None, None, None, 1)
plt.show()
