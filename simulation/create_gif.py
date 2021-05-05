import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import seaborn as sns


try:
    result_folder_name = sys.argv[1]
except:
    print("No folders were entered.")
    print("Closing program.")
    sys.exit(0)
try:
    result_file_name = sys.argv[2]
except:
    print("No result file name entered.")
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

# plt.ion()
sns.set_style("dark")
fig, (ax1, ax2) = plt.subplots(nrows=2, gridspec_kw={'height_ratios': [4, 1]})
plt.tight_layout()

cmap = colors.ListedColormap(["black", "yellow", "red", "green"])
grid = ax1.imshow(np.load(
    "{}/{}.npy".format(path, str(steps[0]))), interpolation="nearest", cmap=cmap, vmin=0, vmax=3)
line1, = ax2.plot(steps[0], infected[0], color="red")
line2, = ax2.plot(steps[0], susceptible[0], color="yellow")
line3, = ax2.plot(steps[0], immune[0], color="green")


def update(frame):
    grid.set_data(np.load("{}/{}.npy".format(path, str(steps[frame]))))
    line1.set_data(steps[0: frame+1], infected[0: frame+1])
    line2.set_data(steps[0: frame+1], susceptible[0: frame+1])
    line3.set_data(steps[0: frame+1], immune[0: frame+1])
    ax2.relim()
    ax2.autoscale_view()

    return [grid, line1, line2, line3]


animation = FuncAnimation(
    fig, update, frames=np.arange(0, len(steps)), blit=False, interval=100, repeat=True)


plt.show()

# Change parameters for your own preference
animation.save("{}/{}.gif".format(file_path, result_file_name), fps=6,
               bitrate=-1, codec="libx264", dpi=100)
