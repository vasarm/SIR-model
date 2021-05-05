import sys
import os

from simulation.simulation import Simulation
from simulation.array_from_picture import convert_img_to_array


try:
    # If picture file entered then use it
    lattice = None
    filename = sys.argv[1]
    file_path = os.path.dirname(__file__)
    lattice = convert_img_to_array(file_path, filename)

except IndexError:
    pass
except FileNotFoundError:
    print("Could not find entered file in this folder. Check file name and it's location.")


sim = Simulation(K=0.5, T=0.1, I=0.001, width=1000, height=1000)
sim.init(random=True, lattice=lattice)
answer = sim.run(number_of_steps=0, count_lattice_step=10, save=True)
sim.display_result(y_axis_type="%")
