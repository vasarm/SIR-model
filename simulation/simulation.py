import time
import os
import types

import numpy as np
import pyopencl as cl

import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    from kernels import load_kernel1, load_counter_kernel
else:
    from .kernels import load_kernel1, load_counter_kernel


class Simulation:

    def __init__(self, K, T, I=0, width=100, height=100):
        """
        Parameters
        ----------
        K : int, float, float32
            Probability that node infects.
        T: int, float, float32
            Probability that node gets immune after being infected.
        I: float32, function
            Must be float32 or function (dependent on step value) which returns float32 type value between 0< I <= 1.
            Probability to get immune without being infected.
        width: int, optional
            width of the array without padding (width of the lattice)
        height: int, optional
            height of the array without padding (height of the lattice)
        """
        # width*height must be <= 50_000_000
        # Hard coded limit as one file is ~50 MB. Change if you want.
        self.max_array_size_to_save = 50_000_000

        # Init lattice shape
        self.shape = np.array([0, 0], dtype=np.int64)

        # Init probability parameters
        self.T = T
        self.K = K
        self.I = I

        # Init width and height
        # Changing width or height automatically changes self.shape as well
        self.width = np.uint(width)
        self.height = np.uint(height)

        # Lattices
        self.state1 = None
        self.state2 = None

        # Other arrays
        self.random_numbers = None  # For generating random numbers
        self.counter_array = None  # For reading data from counter_buf

        # Counter method
        self.count_methods = ["gpu", "cpu1"]
        self.count_method = "gpu"

        # Kernel attributes
        self.kernel = None
        self.counter_kernel = None
        self.queue = None

        # Buffers
        self.state1_buf = None  # State 1 buffer
        self.state2_buf = None  # State 2 buffer
        self.random_buf = None  # Random numbers buffer
        self.counter_buf = None  # GPU lattice reading buffer. Changed if reading method = "gpu"

        # Result attributes
        self.susceptible_list = None
        self.infected_list = None
        self.immune_list = None
        self.step_list = None

    @property
    def T(self):
        return self._T

    @property
    def K(self):
        return self._K

    @property
    def I(self):
        return self._I

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @T.setter
    def T(self, new_T):
        if isinstance(new_T, (int, float, np.float32)) and 0 < new_T <= 1:
            self._T = lambda x: np.float32(new_T)
        elif isinstance(new_T, types.FunctionType):
            self._T = lambda x: np.float32(new_T(x))
        elif not isinstance(new_T, (int, float, np.float32)):
            raise TypeError("T must be integer or float.")
        elif new_T > 1 or new_T <= 0:
            raise ValueError("T must be 0 < T <= 1")

    @K.setter
    def K(self, new_K):
        if isinstance(new_K, (int, float, np.float32)) and 0 < new_K <= 1:
            self._K = lambda x: np.float32(new_K)
        elif isinstance(new_K, types.FunctionType):
            self._K = lambda x: np.float32(new_K(x))
        elif not isinstance(new_K, (int, float, np.float32)):
            raise TypeError("T must be integer or float.")
        elif new_K > 1 or new_K <= 0:
            raise ValueError("T must be 0 < T <= 1")

    @I.setter
    def I(self, new_I):
        if isinstance(new_I, (int, float, np.float32)) and 0 <= new_I <= 1:
            self._I = lambda x: np.float32(new_I)
        elif isinstance(new_I, types.FunctionType):
            self._I = lambda x: np.float32(new_I(x))
        elif not isinstance(new_I, (int, float, np.float32, types.FunctionType)):
            raise TypeError("I must be integer or float or function.")
        elif isinstance(new_I, (int, float, np.float32)) and (new_I > 1 or new_I <= 0):
            raise ValueError("I must be 0 <= I <= 1")

    @width.setter
    def width(self, new_width):
        if isinstance(new_width, (int, np.uint)) and new_width > 0:
            self.shape[1] = np.int64(new_width)
            self._width = np.uint(new_width)
        else:
            raise TypeError("Width must be positive integer.")

    @height.setter
    def height(self, new_height):
        if isinstance(new_height, (int, np.uint)) and new_height > 0:
            self.shape[0] = np.int64(new_height)
            self._height = np.uint(new_height)
        else:
            raise TypeError("Height must be positive integer.")

    ############################
    ####    Class mehtods   ####
    ############################

    def _pad_array(self, array, pad_width=1, constant_values=0):
        return np.pad(array, pad_width=pad_width, constant_values=constant_values)

    def create_lattice(self, random=None, p=None, cnt_infected=None,
                       lattice=None):
        """
        Create lattice on which simulation is performed. Also adds padding. Lattice values are
        0 - wall/padding, 1 - susceptible, 2 - infected, 3 - immune

        Parameters
        ----------
        random: bool, optional
            Boolean to select lattice creating method. If random then
            create array randomly, by default True
        p: list, optional
            If random == True then this array is used to define probabilities for
            node values. p[0] : 1, p[1] : 2, p[2] : 3.
            If [p1] = 0 then generate lattice useing p[0] and p[2] and then insert
            given number of infected nodes in the lattice; by default [0.999, 0, 0.001]

        cnt_infected: int, optional
            If p[1] == 0 then generate number of infected nodes in lattice.
            p[1] and cnt_infected can't be zero both at the same time

        lattice: array, optional
            if random is set False then it is possible to insert own lattice.

        Returns
        -------
        np.array

        """
        if random is None:
            random = True
        if p is None:
            p = [0.999, 0, 0.001]
        if cnt_infected is None:
            cnt_infected = 10

        if random:
            # Possible node values on lattice
            nodes = [1, 2, 3]

            if p[1] == 0 and cnt_infected == 0:
                raise ValueError(
                    "Probability to generate infected node and cnt_infected can't be 0 at the same time.")
            elif p[1] != 0:
                if np.sum(p) != 1.0:
                    raise ValueError("Sum of probabilities must be 1.")

            lattice = np.random.choice(
                nodes, p=p, size=self.shape)

            if p[1] == 0:
                generate_x_coordinates = np.random.randint(
                    0, self.width-1, size=cnt_infected)
                generate_y_coordinates = np.random.randint(
                    0, self.height - 1, size=cnt_infected)
                coordinates = list(
                    zip(generate_x_coordinates, generate_y_coordinates))
                for x, y in coordinates:
                    lattice[y, x] = 2

        elif not random:
            lattice = np.array(lattice)
            self.width = lattice.shape[1]
            self.height = lattice.shape[0]
        # Add padding:
        lattice = self._pad_array(lattice)
        lattice = lattice.astype(np.ubyte)
        return lattice

    def _generate_random_numbers(self):
        # Random number type is unsigned int -> max value = 2^32 - 1
        return self._pad_array(np.random.randint(low=0, high=4294967295, size=self.shape, dtype=np.uint))

    def _check_device_memory(self, device):
        device_memory = device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
        # Count the size of three large arrays + 8 (size of counter array)
        array_memory = self.state1.nbytes + \
            self.state2.nbytes + self.random_numbers.nbytes + 8
        return array_memory > device_memory

    def _init_kernel(self):

        ctx = cl.create_some_context()
        if self._check_device_memory(ctx.devices[0]):
            raise RuntimeError(
                "Selected device has not enough memory for these arrays.")
        # Load kernel text from other file and build kernel wrapper
        self.kernel = cl.Program(ctx, load_kernel1()).build()
        self.counter_kernel = cl.Program(ctx, load_counter_kernel()).build()

        self.queue = cl.CommandQueue(ctx)
        mem_flags = cl.mem_flags

        # Buffers
        self.state1_buf = cl.Buffer(ctx, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR,
                                    hostbuf=self.state1)
        self.state2_buf = cl.Buffer(ctx, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR,
                                    hostbuf=self.state2)
        self.random_buf = cl.Buffer(ctx, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR,
                                    hostbuf=self.random_numbers)

        # If using gpu to count states then create buffer for that
        if self.count_method == "gpu":
            self.counter_buf = cl.Buffer(ctx, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR,
                                         hostbuf=self.counter_array)

    def _save_counted_result(self, step_number):
        infected = self.counter_array[0]
        immune = self.counter_array[1]
        self.infected_list.append(infected)
        self.immune_list.append(immune)
        self.susceptible_list.append(
            self.width*self.height - infected - immune)
        self.step_list.append(step_number)

    def _cpu1_count(self, buf_id):
        if buf_id == 1:
            return np.bincount(self.state1.reshape(-1), minlength=4)[2:4]
        elif buf_id == 2:
            return np.bincount(self.state2.reshape(-1), minlength=4)[2:4]
        else:
            raise ValueError("Buffer id must be 1 or 2.")

    def _count_and_save_state(self, i, save=False, folder=None):
        if self.count_method == "gpu":
            self.counter_array = np.array([0, 0], dtype=np.uint32)
            cl.enqueue_copy(self.queue, self.counter_buf, self.counter_array)
            if i % 2 == 1:
                self.counter_kernel.count_state(self.queue, np.flip(self.shape), None,
                                                self.width, self.state2_buf, self.counter_buf).wait()
                if save:
                    cl.enqueue_copy(self.queue, self.state2,
                                    self.state2_buf).wait()
                    self._save_state(self.state2, folder, i)

            else:
                self.counter_kernel.count_state(self.queue, np.flip(self.shape), None,
                                                self.width, self.state1_buf, self.counter_buf).wait()
                if save:
                    cl.enqueue_copy(self.queue, self.state1,
                                    self.state1_buf).wait()
                    self._save_state(self.state1, folder, i)

            cl.enqueue_copy(self.queue, self.counter_array,
                            self.counter_buf).wait()

            # Save results
            self._save_counted_result(i)
        elif self.count_method == "cpu1":
            if i % 2 == 1:
                cl.enqueue_copy(self.queue, self.state2,
                                self.state2_buf).wait()
                self.counter_array = self._cpu1_count(2)
                if save:
                    self._save_state(self.state2, folder, i)

            else:
                cl.enqueue_copy(self.queue, self.state1,
                                self.state1_buf).wait()
                self.counter_array = self._cpu1_count(1)
                if save:
                    self._save_state(self.state2, folder, i)

            self._save_counted_result(i)

    def _create_result_folder(self):
        path = os.path.dirname(__file__)
        # name[6:] because len(result) = 6
        result_directories = [name[6:] for name in os.listdir(
            path) if os.path.isdir("{}/{}".format(path, name)) and "result" in name]
        # All directories which have number in results
        numbers = [int(number)
                   for number in result_directories if number.isdigit()]
        if len(numbers) == 0:
            name = "result1"
        else:
            free_numbers = set(
                np.arange(1, np.max(numbers) + 2)) - set(numbers)
            new_index = int(min(free_numbers))
            name = "result{}".format(new_index)
        os.mkdir("{}/{}".format(path, name))

        return "{}/{}".format(path, name)

    def _save_state(self, state, folder, i):
        np.save("{}/{}".format(folder, i), state)

    def init(self, random=None, p=None, cnt_infected=None, lattice=None, count_method=None):

        if count_method is not None and count_method in self.count_methods:
            self.count_method = count_method

        # Initialize result arrays
        self.susceptible_list = []
        self.infected_list = []
        self.immune_list = []
        self.step_list = []

        self.state1 = self.create_lattice(random=random, p=p, cnt_infected=cnt_infected,
                                          lattice=lattice)
        self.state2 = np.array(self.state1, copy=True)
        self.random_numbers = self._generate_random_numbers()

        # Node counter array
        self.counter_array = np.array([0, 0], dtype=np.uint32)

        # Initiate kernel
        self._init_kernel()

        # Delete random number array to delete memory from computer memory
        del self.random_numbers

    def _run_one_step(self, step_id, count_step, save, folder=None):
        """
        Runs one step and saves if condition is met.

        Parameters
        ----------

        Returns
        -------

        """
        if step_id % 2 == 1:
            event = self.kernel.run_cycle(self.queue, np.flip(self.shape), None, self.width, self.height,
                                          self.K(step_id), self.T(
                                              step_id), self.I(step_id),
                                          self.random_buf, self.state1_buf, self.state2_buf)
        else:
            event = self.kernel.run_cycle(self.queue, np.flip(self.shape), None, self.width, self.height,
                                          self.K(step_id), self.T(
                                              step_id), self.I(step_id),
                                          self.random_buf, self.state2_buf, self.state1_buf)
        event.wait()

        if step_id % count_step == 0:
            self._count_and_save_state(step_id, save, folder=folder)

    def run(self, number_of_steps=1000, count_lattice_step=10, save=False):
        if save:
            if self.width * self.height > self.max_array_size_to_save:
                raise RuntimeError(
                    "Lattice width*height is bigger than set max array size. Saved files get too big. Change cap limit if you want to run program with these parameters.")
            folder = self._create_result_folder()
        else:
            folder = None

        # Count initialized state
        start_time = time.perf_counter()
        self._count_and_save_state(0, save, folder)
        if number_of_steps < 0:
            raise ValueError("Number of steps must be >= 0.")
        if not isinstance(number_of_steps, int):
            raise TypeError("Number of steps must be integer.")
        if count_lattice_step <= 0:
            raise ValueError("Lattice counting step must be > 0")
        if not isinstance(count_lattice_step, int):
            raise TypeError("Lattice counting step must be integer.")
        if not isinstance(save, bool):
            raise TypeError("Save must be boolean.")

        if number_of_steps == 0:  # Run till no nodes are ill
            i = 1
            while self.infected_list[-1] != 0:
                self._run_one_step(i, count_lattice_step,
                                   save=save, folder=folder)
                i += 1

        else:
            for i in range(1, number_of_steps+1):
                self._run_one_step(i, count_lattice_step,
                                   save=save, folder=folder)

        # Count last state as well if needed.
        if number_of_steps % count_lattice_step != 0:
            self._count_state(i)
        end_time = time.perf_counter()
        print("Simulation run {} seconds.".format(end_time-start_time))

        if save:
            # Save step counts as well
            np.save("{}/infected".format(folder), np.array(self.infected_list))
            np.save("{}/susceptible".format(folder),
                    np.array(self.susceptible_list))
            np.save("{}/immune".format(folder), np.array(self.immune_list))
            np.save("{}/steps".format(folder), np.array(self.step_list))

    def display_result(self, y_axis_type="abs"):
        ytypes = ["abs", "%"]
        if y_axis_type not in ytypes:
            raise ValueError("ytype must be one of {}".format(ytypes))

        sns.set_style("darkgrid")
        sns.set_context("talk", font_scale=1)

        if self.step_list is None or len(self.step_list) == 0:
            raise RuntimeError(
                "No results. Initialize and run simulation first")

        if y_axis_type == "abs":
            sns.lineplot(x=self.step_list, y=self.susceptible_list,
                         label="Susceptible", color="blue", alpha=0.7)
            sns.lineplot(x=self.step_list, y=self.infected_list,
                         label="Infected", color="red", alpha=0.7)
            sns.lineplot(x=self.step_list, y=self.immune_list,
                         label="Immune", color="green", alpha=0.7)
            plt.ylabel("Number of infected nodes.")
        elif y_axis_type == "%":
            sns.lineplot(x=self.step_list, y=np.array(self.susceptible_list) / (self.width * self.height),
                         label="Susceptible", color="blue", alpha=0.7)
            sns.lineplot(x=self.step_list, y=np.array(self.infected_list) / (self.width * self.height),
                         label="Infected", color="red", alpha=0.7)
            sns.lineplot(x=self.step_list, y=np.array(self.immune_list) / (self.width * self.height),
                         label="Immune", color="green", alpha=0.7)
            plt.ylabel("% of infected nodes.")
        plt.title("Infection dynamics graph".format(self.K, self.T))
        plt.xlabel("Step number")

        plt.show()


if __name__ == "__main__":
    lattice = np.full(shape=(1000, 1000), fill_value=1)
    lattice[0:500, 500] = 0
    lattice[123, 965] = 2
    #lattice[123, 966] = 3

    sim = Simulation(K=0.5, T=0.1, I=0.001, width=1000, height=1000)
    sim.init(random=False, lattice=lattice)
    answer = sim.run(number_of_steps=0, count_lattice_step=10, save=True)
    sim.display_result(y_axis_type="%")
