import sys
sys.path.append("../")

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from network import train
from cell import NaP_K_Cell

from neuron import h
from neuron.units import ms
from multiprocessing import Pool

sim_time_ms = 5000
delay_ms = 100
win_size = 80

def _simulate_with_CI(params):
    h.load_file('stdrun.hoc')
    cell = NaP_K_Cell()
    cell.set_I(params)
    h.tstop = sim_time_ms * ms
    h.dt = 0.1
    h.run()
    V = cell.get_V()[::int(1 / h.dt)][:sim_time_ms]
    spike_times = cell.get_spike_times()
    I = cell.get_I()[::int(1 / h.dt)][:sim_time_ms]
    return I, V, spike_times

def generate_CI_dataset(size, current_type, random_state):
    pool = Pool(processes = 6)
    all_CI_params = []
    for _ in range(size):
        np.random.seed(random_state.randint(50, 450))
        if current_type == "const":
            params = [(np.random.randint(-50, 50) / 1000, sim_time_ms - delay_ms, delay_ms)]
        all_CI_params.append(params)
    
    result = pool.map(_simulate_with_CI, all_CI_params)
    pool.close()
    pool.join()
    I, V, spikes = np.zeros((size, sim_time_ms)), np.zeros((size, sim_time_ms)), np.zeros((size, sim_time_ms))
    for i in range(size):
        I[i] = result[i][0]
        V[i] = result[i][1]
        spikes[i, result[i][2]] = 1

    return I, V, spikes


def generate_dataset(size, N_exc_syn, N_inh_syn, current_type, random_state):

    N_syn = N_exc_syn + N_inh_syn
    I_input = np.zeros((size, N_syn, sim_time_ms))
    V_ouput = np.zeros((size, sim_time_ms))
    spikes_out = np.zeros((size, sim_time_ms))

    for i in tqdm(range(size)):
        np.random.seed(random_state.randint(50, 450))

        if current_type == "const":
            CI = np.ones((N_syn, sim_time_ms)) * np.random.randint(-20, 90) / 10 / N_syn
        I_input[i] = CI

        #manager = Manager()
        return_dict = {} #manager.dict()
        #barrier = Barrier(2)
        
        #p = Process(target = simulate, args = (return_dict))
        #p.start()
        #barrier.wait()
        #p.join()
        #p.terminate()
        simulate(return_dict)

        V_out, spikes = return_dict['V'], return_dict['spike_times']
        V_ouput[i] = V_out.flatten()
        if len(spikes) > 0:
            spikes_out[i, spikes] = 1
    
    return I_input, V_ouput, spikes_out

if __name__ == "__main__":

    os.system(f"nrnivmodl ../modfiles_hay > /dev/null 2>&1")
    random_state = np.random.RandomState(123)
    I_input, V_ouput, spikes_out = generate_CI_dataset(10, "const", random_state)
    print(np.where(spikes_out[0] > 0)[0])
    print(V_ouput.shape)
    plt.plot(V_ouput[0])
    plt.show()

    os.system("rm -r x86_64")