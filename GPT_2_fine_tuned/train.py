import matplotlib.pyplot as plt
import numpy as np

def plotter(array, ranges, colors, FRAME):
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    fig.suptitle(f'Frame {FRAME}', fontsize=20)

    # Histogram for temperature
    array = array[array >= 50]
    n, bins, patches = axes[0].hist(array[array != 0], bins=np.arange(min(array), max(array) + 1), color='blue', alpha=0.7)
    axes[0].set_xlim(0, 400)
    axes[0].set_ylim(0, 2000)
    axes[0].set_xlabel("Temperature in Celsius")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Stick Temperature Distribution")

    # Creating a legend for temperature ranges
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=color, label=f'{start} - {end} °C') for (start, end), color in zip(ranges, colors)]
    axes[0].legend(handles=legend_patches, loc='upper left', title='Temperature Ranges')

# Example data and calling the function
array = np.random.randint(50, 400, size=1000)  # Random data for demonstration
ranges = [(50, 100), (100, 150), (150
