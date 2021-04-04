# https://indico.cern.ch/event/93877/contributions/2118070/attachments/1104200/1575343/acat3_revised_final.pdf


# Random number generator
# https://www.cems.uwe.ac.uk/~irjohnso/coursenotes/ufeen8-15-m/p1192-parkmiller.pdf

import pyopencl as cl
import numpy as np
import time
opencl_kernel = """

int random(unsigned int seed){
    unsigned int t = seed ^ (seed << 11);
    unsigned int number = seed ^ (seed >> 19) ^ (t ^ (t >> 8));
    return number;
}

float return_probability(int id, __global unsigned int * rand){
    unsigned int seed = rand[id];
    unsigned int random_number = random(seed);
    rand[id] = seed;
    float probability = (float) seed/4294967295;
    return probability;
}

__kernel void model(const int width, const int height, const float K, const float T,
                     __global unsigned int *rand, __global unsigned char *data, __global unsigned char *result){
        
    unsigned int y = get_global_id(1);
    unsigned int x = get_global_id(0);

    // Take padding into account (width+2)
    int index = (width+2) * y + x;
    // If data[index] == 3 or data[index] == 0 then do nothing. 0 - wall, 3 - immune
    
    if(data[index] == 2){
        float probability = return_probability(index, rand);
        if (probability > K){
            result[index] = (char) 2;
        }
        else{
            result[index] = (char) 3;
        }

    }
    else if (data[index] == 1){
        //count the number of ill nodes near the node
        //ill node has value 2
        char count = 0;
        if (data[index-1] == 2){
            count +=1;
        }
        if (data[index+1] == 2){
            count +=1;
        }
        if (data[index-width] == 2){
            count +=1;
        }
        if (data[index+width] == 2){
            count +=1;
        }
        
        if (count == 0){
            result[index] = (char) 1;
        }
        else{
            float prob_to_infect = (float) 1 - pow(1-K, count);
            float probability = return_probability(index, rand);

            if (probability > prob_to_infect){
                result[index] = (char) 1;
            }
            else{
                result[index] = (char) 2;
            }
        }
    }
}
"""
T = np.float32(0.5)
K = np.float32(0.3)

width = np.uint(100)
height = np.uint(100)
shape = np.array([height, width], dtype=np.int)

initial_random_states = np.pad(np.random.randint(low=0, high=4294967295, size=shape, dtype=np.uint),
                               pad_width=1, constant_values=0)
initial_states = np.pad(np.random.choice(np.array([1, 2, 3], dtype=np.byte), size=shape),
                        pad_width=1, constant_values=0)
initial_states_2 = np.copy(initial_states)
print(shape)
print(shape+2)

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])


kernel = cl.Program(context, opencl_kernel).build()
#kernel2 = cl.Program(context, opencl_kernel).build()
queue = cl.CommandQueue(context)
mem_flags = cl.mem_flags

state_1_buf = cl.Buffer(context, mem_flags.READ_WRITE |
                        mem_flags.COPY_HOST_PTR, hostbuf=initial_states)
state_2_buf = cl.Buffer(
    context, mem_flags.READ_WRITE |
    mem_flags.COPY_HOST_PTR, hostbuf=initial_states_2)
random_buf = cl.Buffer(
    context, mem_flags.READ_WRITE |
    mem_flags.COPY_HOST_PTR,  hostbuf=initial_random_states)

start = time.perf_counter()
kernel.model(
    queue, np.flip(shape+2), None, width, height, K, T, random_buf, state_1_buf, state_2_buf)
end = time.perf_counter()
print(end-start)
#cl.enqueue_copy(queue, initial_states_2, state_2_buf)
# kernel.model(
#    queue, np.flip(shape), None, np.int32(W), np.int32(H), state_1_buf, state_2_buf)


#cl.enqueue_copy(queue,  initial_states, state_2_buf)
