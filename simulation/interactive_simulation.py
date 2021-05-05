import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from matplotlib import colors
import seaborn as sns


try:
    result_folder_name = sys.argv[1]
except:
    print("No folders were entered.")
    print("Closing program.")
    sys.exit(0)

file_path = os.path.dirname(__file__)

if result_folder_name in os.listdir(file_path):
    path = file_path + "/" + result_folder_name
else:
    print("No folder named as  '{}' folders in {}".format(
        result_folder_name, file_path))


data = [file for file in os.listdir(
    path) if os.path.isfile("{}/{}".format(path, file))]
lattice_data = [name for name in data if name[-4:]
                == ".npy" and name[:-4].isdigit()]
count_data = ["infected.npy",
              "susceptible.npy", "immune.npy", "steps.npy"]

# Load result files
try:
    infected = np.load("{}/infected.npy".format(path))
    susceptible = np.load("{}/susceptible.npy".format(path))
    immune = np.load("{}/immune.npy".format(path))
    steps = np.load("{}/steps.npy".format(path))
except Exception as e:
    print(e)

plt.ion()
sns.set_style("dark")
fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [4, 1]})
plt.tight_layout()

cmap = colors.ListedColormap(["black", "yellow", "red", "yellow"])

for i, step in enumerate(steps):
    if i == 0:
        grid = ax1.imshow(np.load("{}/{}.npy".format(path, step)),
                          interpolation='nearest', cmap=cmap, vmin=0, vmax=3)

        line1, = ax2.plot(steps[0], infected[0], color="red")
        line2, = ax2.plot(steps[0], susceptible[0], color="yellow")
        line3, = ax2.plot(steps[0], immune[0], color="green")
    else:
        grid.set_data(np.load("{}/{}.npy".format(path, step)))
        line1.set_data(steps[0: i+1], infected[0: i+1])
        line2.set_data(steps[0: i+1], susceptible[0: i+1])
        line3.set_data(steps[0: i+1], immune[0: i+1])
    # Update axis2 scale
    ax2.relim()
    ax2.autoscale_view()

    fig.canvas.draw()
    fig.canvas.flush_events()
