import sys
import os
import json

from simulation.simulation import Simulation
from simulation.array_from_picture import convert_img_to_array


with open("config.json") as f:
    settings = json.load(f)

try:
    if settings["lattice_path"] != "-1":
        if os.path.isfile(settings["lattice_path"]):
            file_name = os.path.basename(settings["lattice_path"])
            file_path = os.path.dirname(settings["lattice_path"])
            lattice = convert_img_to_array(file_path, file_name)
            settings["random"] = False
        else:
            print("Enetered lattice path is not file.")
            sys.exit(-1)
except:
    print("Problem with settings 'lattice_path' variable. Set '-1' if no input.")
    sys.exit(-1)


sim = Simulation(K=settings["K"], T=settings["T"], I=settings["I"],
                 width=settings["width"], height=settings["height"])
sim.init(random=settings["random"], lattice=lattice)
sim.run(number_of_steps=settings["number_of_steps"],
        count_lattice_step=settings["count_lattice_step"], save=settings["save"])
if settings["display_result"]:
    sim.display_result(y_axis_type="%")
