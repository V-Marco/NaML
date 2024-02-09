import numpy as np
from scipy.signal import convolve, lfilter
from scipy.signal.windows import exponential

def inhomogeneous_poisson_through_num_points_for_window_one(lambdas):
    t = np.zeros(len(lambdas))
    for i, lambd in enumerate(lambdas):
        num_points = np.random.poisson(lambd)
        if num_points > 0: t[i] = 1
    return t

def generate_pink_noise(num_obs, mean = 1, std = 0.5):

    white_noise = np.random.normal(mean, std, num_obs + 2000)

    A = [1, -2.494956002, 2.017265875, -0.522189400]
    B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]

    pink_noise = lfilter(B, A, white_noise)[2000:]

    return pink_noise

def generate_exponential_current(spike_train):
    win = exponential(M = 141, center = 0, tau = 20, sym = False)
    current = convolve(spike_train, win, "full")
    return current

def generate_current_inj_matrix(mfr, N_syn, T):
    matrix = np.zeros((N_syn, T))
    for i in range(N_syn):
        train = inhomogeneous_poisson_through_num_points_for_window_one(generate_pink_noise(num_obs = T, mean = mfr) / 1000)
        matrix[i, :] = generate_exponential_current(train)[:T]
    return matrix