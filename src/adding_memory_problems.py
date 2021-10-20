# Created by moritz (wolter@cs.uni-bonn.de)
import numpy as np

def generate_data_adding(time_steps: int, n_data: int) -> tuple:
    """
    Generate data for the adding problem.

    Source: https://github.com/amarshah/complex_RNN/blob/master/adding_problem.py

    Args:
        time_steps: The number of time steps we would like to consider.
        n_data: the number of sequences we would like to consider overall.

    Returns:
        tuple: x [time_steps, n_data, 2] input array,
               y [n_data, 1] output array.
    """
    x = np.asarray(np.zeros((time_steps, int(n_data), 2)),
                   dtype=np.float)
    # this should be low=-1!? According to hochreiter et al?!
    x[:, :, 0] = np.asarray(np.random.uniform(low=0.,
                                              high=1.,
                                              size=(time_steps, n_data)),
                            dtype=np.float)
    inds = np.asarray(np.random.randint(time_steps/2, size=(n_data, 2)))
    inds[:, 1] += int(time_steps/2)

    for i in range(int(n_data)):
        x[inds[i, 0], i, 1] = 1.0
        x[inds[i, 1], i, 1] = 1.0

    y = (x[:, :, 0] * x[:, :, 1]).sum(axis=0)
    y = np.reshape(y, (n_data, 1))
    return x, y


def generate_data_memory(time_steps: int, n_data: int, n_sequence: int) -> tuple:
    """ Generate data for the memory problem.

    Args:
        time_steps (int): The length of the problem.
        n_data (int): the number of samples we would like to generate overall.
        n_sequence (int): The length of the sequence to memorize.

    Returns:
        tuple: The input and desired output values.
    """    
    seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, time_steps - 1))
    zeros2 = np.zeros((n_data, time_steps))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int32')
    return x, y