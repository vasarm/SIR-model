

def kernel1():
    kernel = """

    int random(unsigned int seed){
        unsigned int t = seed ^ (seed << 11);
        unsigned int number = seed ^ (seed >> 19) ^ (t ^ (t >> 8));
        return number;
    }

    float return_probability(int id, __global unsigned int * rand){
        unsigned int seed = rand[id];
        unsigned int random_number = random(seed);
        rand[id] = random_number;
        float probability = (float) seed/4294967295;
        return probability;
    }

    __kernel void run_cycle(const int width, const int height, const float K, const float T,
                        __global unsigned int *rand, __global unsigned char *data, __global unsigned char *result){
        // state is an array [number of ill, number of immune]
        // width and height are for non padded arrays, so add/substract 2 if using.

        unsigned int y = get_global_id(1);
        unsigned int x = get_global_id(0);

        // Take padding into account (width+2)
        int index = (width+2) * y + x;
        // If data[index] == 3 or data[index] == 0 then do nothing. 0 - wall, 3 - immune
        if (data[index] == 3){
            result[index] = (char) 3;
        } 
        // Get immune
        else if(data[index] == 2){
            float probability = return_probability(index, rand);
            // If generated probability (assumed uniform) is bigger than T, then do not get immune
            if (probability <= T){
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
            if (data[index-width-2] == 2){
                count +=1;
            }
            if (data[index+width+2] == 2){
                count +=1;
            }
            
            if (count != 0){
                float prob_to_infect = (float) 1 - pow((float) 1-K, count);
                float probability = return_probability(index, rand);

                // If generated probabilty (let's assume uniform) is bigger than calculated probability, then do not get infected
                if (probability <= prob_to_infect){
                    result[index] = (char) 2;
                }
            }
        }
    }
    """
    return kernel


def counter_kernel():
    kernel = """
    #pragma OPENCL EXTENSION cl_khr_global_int64_extended_atomics : enable
    __kernel void count_state(const int width, __global unsigned char *final_state, __global unsigned long *counter){
        
        // Initiate counter
        counter[0] = 0;
        counter[1] = 0;

        unsigned int y = get_global_id(1);
        unsigned int x = get_global_id(0);

        int index = (width+2) * y + x;
        
        // Let it count
        if (final_state[index] == 2){
            atomic_add(counter[0], (unsigned long) 1);
        }
        else if (final_state[index] == 3){
            atomic_add(counter[1], (unsigned long) 1);
        }
    }
    """
    return kernel
