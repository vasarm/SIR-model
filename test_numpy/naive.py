import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation


def cycle(matrix, T, K):
    """matrix should be (n_person+1) * (n_person+1) matrix where elements at the border are all -1.
    Easier to check for neighbours at the borders.
    
    Notation:
    1: Healthy (immune)
    0: Healthy (not immune)
    2: Infected
    """ 
    height, width = matrix.shape
    height -= 2
    width -= 2
    # Make copy to have one matrix, which is static.
    copy_matrix = np.array(matrix, copy=True)

    for i in range(1, height+1):
        for j in range(1, width+1):
            if copy_matrix[i,j] == 1:
                continue
            elif copy_matrix[i,j] == 2:
                # Probabilty to get well is T and to stay sick is 1-T
                matrix[i,j] = np.random.choice([2, 1], size=1, p=[1-T, T])
                continue
            else:
                prob = 1
                if copy_matrix[i-1, j] == 2:
                    prob = prob * (1-K)
                if copy_matrix[i+1, j] == 2:
                    prob = prob * (1-K)
                if copy_matrix[i, j-1] == 2:
                    prob = prob * (1-K)
                if copy_matrix[i, j+1] == 2:
                    prob = prob * (1-K)
                # Probability to get sick is 1 - Π(1-K(i,j)) where K(i,j)=K if neighbour
                matrix[i,j] = np.random.choice([0, 2], size=1, p=[prob, 1-prob])

def animate(i, image, matrix, T, K):
    cycle(matrix, T, K)
    image.set_array(matrix[1:-1, 1:-1])
    return matrix

T = 0.2
K = 0.4
# Dimensions
height = 20
width = 15
# Set up persons matrix and select which random person is ill
persons = np.zeros((height, width))
ill_h = np.random.random_integers(low=0, high=height-1, size=1)
ill_w = np.random.random_integers(low=0, high=width-1, size=1)
persons[ill_h, ill_w] = 2
# Pad borders with -1 values
persons = np.pad(persons, 1, constant_values = -1)


# Display 
# white - not infected, green - immune, red - infected
cmap = colors.ListedColormap(['w','g','r'])
fig = plt.figure()
im = plt.imshow(persons[1:-1, 1:-1], cmap=cmap)

# Interval = refresh rate in ms
ani = animation.FuncAnimation(fig, animate, fargs =(im, persons, T, K),
                                    interval=300)
plt.show()