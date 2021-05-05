# SIR-model

## Package dependencies
    * numpy
    * seaborn
    * matplotlib
    * pyopencl
    * Pillow (if want to imprt array from picture)

## ...
    Lattice array values:

    0 - wall
    1 - susceptible
    2 - infected
    3 - immune


## Usage

Create Simulation object and define parameters:
    
    1) T - probability that node transforms from 2 -> 3 (Gets immune after beeing infected). Must be between 0 < T <=1
    2) K - probaability that node transforms from 1 -> 3 (Gets infected). Must be between 0 < K <=1
    *3) I - probability that node transfroms from 1 -> 3 (Gets randomly immune). Default value = 0. Must be between 0 <= I <= 1

These parameters can also be functions of step value but then check if returned value is always between required interval. (Program does not explicitly control it.)

If using random lattice genration then define lattice parameters as well. If inserting your own lattice then program changes these values automatically.
    
    4) width: int, (default=100)
        lattice width

    5) height: int, (default=100)
        lattice height

```Python
    sim = Simulation(K, T, I, width)
```

Next initialize simulation which creates kernel and lattice.
    1) random : True/False
        Select if lattice generation is random or user provided. If user provided then set False. (Default: True)
    2) p : list [p_sus, p_inf, p_imm], (default [0.999, 0, 0.001])
        If lattice generation is random then insert probabilities to generate susceptible, infected, immune (accordingly to list) node on the lattice. Values must be < 1 and sum of all parameters must be. If p_inf (probability to generate infected node) = 0 then program takes next parameter which generates N infected nodes randomly on the lattice.
    3) cnt_infected : int, (default = 10)
        If p_inf == 0 then generate number of "cnt_infected" infected nodes on the lattice.
    4) lattice : np.array()
        If random == False then user must insert simulation lattice. Must be 2D array.
    5) count_method : string, (defualt = "gpu):
        What method to use for counting nodes on lattice. "gpu" uses gpu implematation. "cpu1" reads memory from GPU memory and reads states on CPU. (With bigger lattices GPU method is faster) 

```Python
    sim.init(random=True, p=[0.999, 0, 0.001], cnt_infected=10)
```

Finally run the simulation. Takes in three parameters:
    1) number_of_steps : int, (default 1000)
        How many simulation steps will be performed on the lattice. If number_of_steps == 0 then run simulation till no infected nodes are left.
    2) count_lattice_step : int, (default 10)
        After these steps count node states on the array and adds to result list.
    3) save : True/False, (default False)
        When program counts node states read current lattice from GPU and save as an array in the program folder. This method creates in the file path new folder where results are saved. Required if to convert simulation to gif.

```Python
    sim.run(number_of_steps=0, count_lattice_step = 10, save=True)
```

Finally it is possible to plot the result. Can change y-axis range:
y_axis_type="abs" - On the y axis number of nodes are displayed
y_axis_type="%" - On the y axis percentage of nodes are displayed 


```
sim.display_result(y_axis_type="abs")
```


