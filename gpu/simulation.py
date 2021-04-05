# https://indico.cern.ch/event/93877/contributions/2118070/attachments/1104200/1575343/acat3_revised_final.pdf


# Random number generator
# https://www.cems.uwe.ac.uk/~irjohnso/coursenotes/ufeen8-15-m/p1192-parkmiller.pdf

import pyopencl as cl
import numpy as np
import time

from kernels import kernel1


class Simulation:

    def __init__(self, K, T, width=100, height=100):
        if width * height > 500000000:
            raise RuntimeError(
                "Error. Memory usage might be too big.")  # < 6 GB total
        self.T = np.float32(T)
        self.K = np.float32(K)
        self.width = np.uint(width)
        self.height = np.uint(height)
        self.shape = np.array([height, width], dtype=np.int)

    def create_random_states(self):
        return np.pad(np.random.randint(low=0, high=4294967295, size=self.shape, dtype=np.uint),
                      pad_width=1, constant_values=0)

    def create_lattice(self, random=True, n_ill=10, custom=None):
        # Create fully random lattice
        if random:
            lattice = np.pad(np.random.choice(np.array([1, 3], dtype=np.byte), p=[0.9, 0.1], size=self.shape),
                             pad_width=1, constant_values=0)
            for x in range(n_ill):
                coord_x = np.random.randint(1, self.width)
                coord_y = np.random.randint(1, self.height)
                lattice[coord_x, coord_y] = 2
        # No immune persons.
        else:
            lattice = np.full(shape=self.shape, fill_value=1, dtype=np.byte)
            for x in range(n_ill):
                coord_x = np.random.randint(0, self.width)
                coord_y = np.random.randint(0, self.height)
                lattice[coord_x, coord_y] = 2
        return lattice

    def setup(self, random, n_ill):
        self.ctx = cl.create_some_context()
        self.kernel = cl.Program(self.ctx, kernel1()).build()
        #self.kernel2 = cl.Program(self.ctx, kernel1()).build()
        self.queue = cl.CommandQueue(self.ctx)
        self.mem_flags = cl.mem_flags

        # Create arrays for buffer
        self.random_states = self.create_random_states()
        self.state1 = self.create_lattice(random=True, n_ill=10)
        self.state2 = np.copy(self.state1)

        # Buffers
        self.state1_buf = cl.Buffer(self.ctx, self.mem_flags.READ_WRITE | self.mem_flags.COPY_HOST_PTR,
                                    hostbuf=self.state1)
        self.state2_buf = cl.Buffer(self.ctx, self.mem_flags.READ_WRITE | self.mem_flags.COPY_HOST_PTR,
                                    hostbuf=self.state2)
        self.random_buf = cl.Buffer(self.ctx, self.mem_flags.READ_WRITE | self.mem_flags.COPY_HOST_PTR,
                                    hostbuf=self.random_states)

    def cycle(self, num_epoch):
        # Temporary. To display result.
        print("Start")
        print(self.state1[1:-1, 1:-1])
        print(self.random_states[1:-1, 1:-1])

        for i in range(num_epoch):
            print("i =", i)
            if i % 2 == 0:
                self.kernel.run_cycle(self.queue, np.flip(self.shape + 2), None, self.width, self.height, self.K, self.T,
                                      self.random_buf, self.state1_buf, self.state2_buf)
                cl.enqueue_copy(self.queue, self.state2,
                                self.state2_buf)
                # Temporary. To display result.
                print(self.state2[1:-1, 1:-1])
            else:
                self.kernel.run_cycle(self.queue, np.flip(self.shape + 2), None, self.width, self.height, self.K, self.T,
                                      self.random_buf, self.state2_buf, self.state1_buf)
                cl.enqueue_copy(self.queue, self.state1,
                                self.state1_buf)
                # Temporary. To display result.
                print(self.state1[1:-1, 1:-1])
            cl.enqueue_copy(self.queue, self.random_states,
                            self.random_buf)
            # Temporary. To display result.
            print(self.random_states[1:-1, 1:-1])

    def run(self, num_epoch=100, save_state=10, random=True, n_ill=10):
        # num_epoch - how many cycles simulation will run the cycle. (Not implemented: 0=Run till approx. no ill nodes are left. )
        # save_state - after every "save_state" saves the current state values. (number of ill and number of immune)
        # At the moment save state returns array
        self.setup(random, n_ill)

        self.cycle(num_epoch)
        if num_epoch % 2 == 0:
            cl.enqueue_copy(self.queue, self.state2, self.state2_buf)
            return self.state2
        else:
            cl.enqueue_copy(self.queue, self.state1, self.state1_buf)
            return self.state1


sim = Simulation(K=1, T=1, width=3, height=3)
answer = sim.run(num_epoch=10)
