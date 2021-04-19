# https://indico.cern.ch/event/93877/contributions/2118070/attachments/1104200/1575343/acat3_revised_final.pdf


# Random number generator
# https://www.cems.uwe.ac.uk/~irjohnso/coursenotes/ufeen8-15-m/p1192-parkmiller.pdf

import pyopencl as cl
import pyopencl.array

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import time

from kernels import kernel1, counter_kernel


class Simulation:

    def __init__(self, K, T, width=100, height=100):
        """[summary]

        Parameters
        ----------
        K : int, float, (any single digit)
            Probability that one node infects.
        T : inf, float, (any single digit)
            Probability that nodes gets immune after being infected
        width : int, optional
            width of the lattice, by default 100
        height : int, optional
            height of the lattice, by default 100

        Raises
        ------
        RuntimeError
            Temporary error to avoid that program will take too much RAM (when creating numpy array).
        """
        if width * height > 500000000:
            raise RuntimeError(
                "Error. Memory usage might be too big.")  # < 6 GB total
        # Initiate shape array
        self.shape = np.array([0, 0], dtype=np.int)

        self.T = np.float32(T)
        self.K = np.float32(K)
        self.width = np.uint(width)
        self.height = np.uint(height)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @width.setter
    def width(self, new_width):
        if isinstance(new_width, (int, np.uint)) and new_width > 0:
            self.shape[1] = np.int(new_width)
            self._width = np.uint(new_width)
        else:
            raise TypeError("Width must be positive integer.")

    @height.setter
    def height(self, new_height):
        if isinstance(new_height, (int, np.uint)) and new_height > 0:
            self.shape[0] = np.int(new_height)
            self._height = np.uint(new_height)
        else:
            raise TypeError("Height must be positive integer.")

    ############################
    ####    Class mehtods   ####
    ############################

    def _lattice_padding(self, array, pad_width=1, constant_values=0):
        return np.pad(array, pad_width=pad_width, constant_values=constant_values)

    def _set_initial_random_numbers(self):
        """
        Create initial random number array for probability calculation.

        Returns
        -------
        ndarray
            Return ndarray with dtype = unsigned int.
        """
        return np.random.randint(low=0, high=4294967295, size=self.shape, dtype=np.uint)

    def create_lattice(self, random=True, p=[0.999, 0, 0.001], n_infected=10, user_lattice=None):
        """
        Create simulation lattice. User can define multiple modes:
        1) random = True, p = [susceptible, infected, immune] : generate random lattice with given probabilities
        2) random = True, p = [susceptible, 0, immune], n_infected = N : Insert exactly N infected nodes
        3) user_lattice = np.ndarray : User defined lattice. For correct simulation use only defined node values.

        Values on lattice: 0-padding, 1-susceptible, 2-infected, 3-immune

        Parameters
        ----------
        random : bool, optional
            If random lattice is generated, by default True
        p : list, optional
            Probabilities for initial state generation, by default [0.999, 0, 0.001]
        n_infected : int, optional
            [description], by default 10
        user_lattice : [type], optional
            [description], by default None

        Returns
        -------
        np.ndarray
            Padded simulation lattice (np.ndarray where dtype=np.byte).

        Raises
        ------
        RuntimeError
            [description]
        ValueError
            [description]
        ValueError
            [description]
        """
        if np.sum(p) != 1:
            raise RuntimeError(
                "Sum of probabilities to generating certain node must be 1.")
        for p_i in p:
            if p_i < 0:
                raise ValueError(
                    "Probability must have value between 0 <= p <= 1")

        if random:
            # Create fully random lattice

            # User can define the number of infected
            if n_infected is not None:
                if p[1] != 0:
                    raise ValueError(
                        "If generating certain number of infected nodes then probability to generate infected must be zero.")

            # Create random lattice
            lattice = np.random.choice(
                np.array([1, 2, 3], dtype=np.ubyte), p=p, size=self.shape)

            # If user defined the number of infected nodes then insert them to lattice.
            if n_infected is not None:
                for x in range(n_infected):
                    coord_x = np.random.randint(0, self.width-1)
                    coord_y = np.random.randint(0, self.height-1)
                    lattice[coord_y, coord_x] = 2

        elif user_lattice is not None:
            # If user has defined lattice then use it.
            self.width = user_lattice.shape[1]
            self.height = user_lattice.shape[0]
            lattice = user_lattice.astype(np.ubyte)
        else:
            raise RuntimeError("Can't generate lattice - no method defined.")

        # Lattice padding
        lattice = self._lattice_padding(lattice, constant_values=np.ubyte(0))
        return lattice

    def _init_kernel(self, count_method="gpu"):
        # Initiate kernel ready for calculation
        self.ctx = cl.create_some_context()
        self.kernel = cl.Program(self.ctx, kernel1()).build()
        # self.kernel2 = cl.Program(self.ctx, kernel1()).build()
        self.kernel_count = cl.Program(self.ctx, counter_kernel()).build()
        self.queue = cl.CommandQueue(self.ctx)
        self.mem_flags = cl.mem_flags

        # Create arrays for buffer
        # Buffers
        # Two lattice state buffers. After every epoch change initial and result buffer.
        self.state1_buf = cl.Buffer(self.ctx, self.mem_flags.READ_WRITE | self.mem_flags.COPY_HOST_PTR,
                                    hostbuf=self.state1)
        self.state2_buf = cl.Buffer(self.ctx, self.mem_flags.READ_WRITE | self.mem_flags.COPY_HOST_PTR,
                                    hostbuf=self.state2)
        # Random number buffer
        self.random_buf = cl.Buffer(self.ctx, self.mem_flags.READ_WRITE | self.mem_flags.COPY_HOST_PTR,
                                    hostbuf=self.random_states)

        # Create buffer for counter according to counter method.
        if count_method == "gpu":
            self.counter = np.array([0, 0], dtype=np.uint32)
            self.counter_buf = cl.Buffer(self.ctx, self.mem_flags.READ_WRITE | self.mem_flags.COPY_HOST_PTR,
                                         hostbuf=self.counter)

    def _count_state(self, i, method="gpu"):
        """Method to count current lattice state.

        Parameters
        ----------
        method : str, optional
            Choose which method to use, by default "gpu"
        """
        if method == "gpu":
            # Let GPU count occurances of 2 and 3.
            self.counter = np.array([0, 0], dtype=np.uint32)
            cl.enqueue_copy(self.queue, self.counter_buf, self.counter)
            if i % 2 == 1:
                self.kernel_count.count_state(self.queue, np.flip(self.shape), None,
                                              self.width, self.state2_buf, self.counter_buf).wait()
            else:
                self.kernel_count.count_state(self.queue, np.flip(self.shape), None,
                                              self.width, self.state1_buf, self.counter_buf).wait()
            cl.enqueue_copy(self.queue, self.counter, self.counter_buf).wait()
            self.state_results.append(list(self.counter))
        if method == "cpu1":
            # Numpy method to fast count numbers. Only 2 and are needed.
            if i % 2 == 1:
                cl.enqueue_copy(self.queue, self.temp_state_array,
                                self.state2_buf).wait()
            else:
                cl.enqueue_copy(self.queue, self.temp_state_array,
                                self.state1_buf).wait()
            counted = np.bincount(
                self.temp_state_array.reshape(-1), minlength=4)[2:4]
            self.state_results.append(list(counted))

    def cycle(self, steps, save=10, count_method="gpu"):
        """Runs kernel num_step of times. As it uses only two buffers it changes between these after every step.
            If epoch%2 == 0 then initial state = state1 and result = state2
            If epoch%2 == 1 then initital state = state2 and result = state1
        Parameters
        ----------
        steps : int
            Number of steps.
        """
        # Count 0-th step.
        self._count_state(0, method=count_method)

        # If steps=0 then simulation runst till no ill nodes are left.
        if steps == 0:
            if self.T == 0:
                raise RuntimeError(
                    "No nodes can ever get cured. Infinite cycle")
            # First state index

            i = 1
            while self.state_results[-1][0] != 0:
                if i % 2 == 1:
                    event = self.kernel.run_cycle(self.queue, np.flip(self.shape), None, self.width, self.height, self.K, self.T,
                                                  self.random_buf, self.state1_buf, self.state2_buf)
                else:
                    event = self.kernel.run_cycle(self.queue, np.flip(self.shape), None, self.width, self.height, self.K, self.T,
                                                  self.random_buf, self.state2_buf, self.state1_buf)
                event.wait()

                # Save first state and then every "save"-th value.
                # For example save=10: 10, 20, 30, etc...
                if i % save == 0:
                    self._count_state(i, method=count_method)
                i += 1

            self.number_of_steps = i-1

        elif steps > 0 and isinstance(steps, int):
            for i in range(1, steps+1):
                # np.flip(self.shape + 2) as numpy shape uses width = shape[1] element and height = shape[0] element. Opencl has it's differently
                if i % 2 == 1:
                    event = self.kernel.run_cycle(self.queue, np.flip(self.shape), None, self.width, self.height, self.K, self.T,
                                                  self.random_buf, self.state1_buf, self.state2_buf)
                else:
                    event = self.kernel.run_cycle(self.queue, np.flip(self.shape), None, self.width, self.height, self.K, self.T,
                                                  self.random_buf, self.state2_buf, self.state1_buf)
                event.wait()

                # Save first state and then every "save"-th value.
                # For example save=10: 10, 20, 30, etc...
                if i % save == 0:
                    self._count_state(i, method=count_method)
        else:
            raise RuntimeError("Not valid simulation parameters.")

        # Count last state as well if needed.
        if steps % save != 0:
            self._count_state(i, method=count_method)

    def run(self, steps=10, save=10, count_method="gpu", **kwargs):
        """
        Initiates OpenCL kernel and runs it num_epoch of times. Every save_state it saves the current array state.

        Parameters
        ----------
        steps : int
            Number of simulation steps, by default 100
        save : int, optional
            After every N steps save current state, by default 10

        Returns
        -------
        [type]
            [description]
        """
        # num_epoch - how many cycles simulation will run the cycle. (Not implemented: 0=Run till approx. no ill nodes are left. )
        # save_state - after every "save_state" saves the current state values. (number of ill and number of immune)
        # At the moment save state returns array

        self.steps = steps
        self.save = save

        # *** Generate initial states for lattices and random numbers***
        self.state1 = self.create_lattice(random=kwargs.get("random", True), p=kwargs.get("p", [0.999, 0, 0.001]),
                                          n_infected=kwargs.get("n_infected", 10), user_lattice=kwargs.get("user_lattice", None))
        self.state2 = np.array(self.state1, copy=True)

        # Array for results
        self.state_results = []
        # Array for counting results
        if count_method == "cpu1":
            self.temp_state_array = np.zeros_like(self.state1)

        # Random number array for random number buffer.
        self.random_states = self._lattice_padding(
            self._set_initial_random_numbers())
        self.step_res = np.empty_like(self.state1, dtype=np.ubyte)
        # Initiate Python wrapper for OpenCL
        self._init_kernel(count_method=count_method)

        # Release random states memory from RAM as it may take a lot of RAM.
        del(self.random_states)

        # Run kernel and lattice simulation
        self.cycle(steps=self.steps, save=self.save, count_method=count_method)

    def plot_result(self, ptype="abs"):
        """Plot simulation results

        Parameters
        ----------
        ptype : str, optional
            Type for plotting y-values. 
            "abs" - y axis shows number of nodes on lattice.
            "percentage" - y axis shows percentage of nodes to the total value of nodes.
            , by default "abs".

        Raises
        ------
        RuntimeError
            [description]
        """
        #fig, ax = plt.subplots(figsize=(15, 7))
        sns.set_style("darkgrid")
        sns.set_context("talk", font_scale=1)

        if len(self.state_results) == 0:
            raise RuntimeError("No results. Run simulation first")

        if self.steps == 0:
            x_axis_values = np.unique(np.concatenate(
                [np.arange(0, self.number_of_steps, self.save), [self.number_of_steps]]))
        else:
            x_axis_values = np.unique(np.concatenate(
                [np.arange(0, self.steps, self.save), [self.steps]]))
        y_sus = []
        y_ill = []
        y_immune = []
        total_nodes = self.width * self.height

        for step_res in self.state_results:
            y_ill.append(step_res[0])
            y_immune.append(step_res[1])
            y_sus.append(total_nodes-step_res[0]-step_res[1])

        if ptype == "percentage":
            y_sus = np.array(y_sus)/total_nodes
            y_ill = np.array(y_ill)/total_nodes
            y_immune = np.array(y_immune)/total_nodes

        data = pd.DataFrame(
            {"Susceptible": y_sus, "Infected": y_ill, "Immune": y_immune})
        sns.lineplot(x=x_axis_values, y=y_sus,
                     label="Susceptible", color="blue", alpha=0.7)
        sns.lineplot(x=x_axis_values, y=y_ill,
                     label="Infected", color="red", alpha=0.7)
        sns.lineplot(x=x_axis_values, y=y_immune,
                     label="Immune", color="green", alpha=0.7)

        plt.show()


sim = Simulation(K=0.5, T=0.04, width=10000, height=10000)
answer = sim.run(steps=100000, save=100, random=True, count_method="gpu")
sim.plot_result(ptype="percentage")
# print(sim.shape)
