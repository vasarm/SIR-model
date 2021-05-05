

def load_kernel1():
    kernel = """

    // bitwise 
    int random(unsigned int seed){
        unsigned int t = seed ^ (seed << 11);
        unsigned int number = seed ^ (seed >> 19) ^ (t ^ (t >> 8));
        return number;
    }

    float return_probability(int id, __global unsigned int * random_number_array){
        unsigned int seed = random_number_array[id];
        unsigned int random_number = random(seed);
        random_number_array[id] = random_number;
        float probability = (float) seed/4294967295;
        return probability;
    }

    __kernel void run_cycle(const int width, const int height, const float K, const float T,
                        __global unsigned int *rand, __global unsigned char *data, __global unsigned char *result){
        // state is an array [number of ill, number of immune]
        // width and height are for non padded arrays, so add/substract 2 if using.

        // Indices
        unsigned int y = get_global_id(1);
        unsigned int x = get_global_id(0);
        // Random numbers
        __global unsigned int *random_numbers_array = rand;

        // Take padding into account (width+2).
        // y = [0, width], so add 1 to iterate over [1, width+1]
        // x = [0, height], so add 1 to iterate over [1, height+1]
        int index = (width+2) * (y+1) + x+1;



        // If data[index] == 3 or data[index] == 0 then do nothing. 0 - wall, 3 - immune
        if (data[index] == 3){
            result[index] = (unsigned char) 3;
        } 
        // Get immune
        else if(data[index] == 2){
            float probability = return_probability(index, random_numbers_array);
            // If generated probability (assumed uniform) is bigger than T, then do not get immune
            if (probability <= T){
                result[index] = (unsigned char) 3;
            }
            else {
                result[index] = (unsigned char) 2;
            }
        }
        else if (data[index] == 1){
            //count the number of ill nodes near the node
            //ill node has value 2
            unsigned char count = 0;
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
                float probability = return_probability(index, random_numbers_array);

                // If generated probabilty (let's assume uniform) is bigger than calculated probability, then do not get infected
                if (probability <= prob_to_infect){
                    result[index] = (unsigned char) 2;
                }
            }
        }
    }
    """
    return kernel


def load_counter_kernel():
    kernel = """
    #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
    void increase(volatile __global uint *counter, int pos){
        atom_inc(&counter[pos]);
    }
    __kernel void count_state(const int width, __global unsigned char *final_state, volatile __global uint *counter){
        // Initiate counter
        
        volatile __global uint *temp_counter = counter;
        unsigned int y = get_global_id(1);
        unsigned int x = get_global_id(0);
        int index = (width+2) * (y+1) + x + 1;
        
        // Let it count
        if (final_state[index] == 2){
            increase(temp_counter, (int) 0);
        }
        else if (final_state[index] == 3){
            increase(temp_counter, (int) 1);
        }
    }
    """
    return kernel
