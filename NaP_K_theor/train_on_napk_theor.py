import sys
sys.path.append("../")
import numpy as np

from simple_models import NaP_Kdr
from voltage_current import CurrentInjection
from network import TCNN, count_parameters, train

from multiprocessing import Pool

import torch
import pickle

sim_time_ms = 1000

def _simulate(I):
    model = NaP_Kdr(V_init = -60, n_init = 0.1)
    V = model.simulate(CurrentInjection.constant(I, sim_time_ms))
    return V

def generate_batch(size):
    np.random.seed(np.random.randint(50, 450))
    pool = Pool(size)
    Is = np.random.randint(-50, 50, size) / 10
    Vs = pool.map(_simulate, Is)

    I_in = np.ones((size, sim_time_ms)) * Is.reshape((-1, 1))
    V_out = np.vstack(Vs)
    spikes_out = np.zeros((size, sim_time_ms))
    for i in range(len(Vs)):
        spikes_out[i, np.where(Vs[i] > -30)] = 1
    return np.expand_dims(I_in, 1), V_out, spikes_out


if __name__ == "__main__":

    model = TCNN(in_channels = 1, kernel_size = 80, num_layers = 1)
    print("Number of parameters: ", count_parameters(model))

    all_train_loss = []
    all_test_loss = []

    for batch_num in range(1):
        I_in_train, V_out_train, spikes_out_train = generate_batch(16)
        I_in_test, V_out_test, spikes_out_test = generate_batch(16)
        train_loss_history, test_loss_history = train(
                num_epoch = 16,
                model = model,
                device_name = "cpu",
                lr = 0.001,
                I_train = I_in_train,
                V_train = V_out_train,
                spike_times_train = spikes_out_train,
                I_test = I_in_test,
                V_test = V_out_test,
                spike_times_test = spikes_out_test,
                sim_time_ms = sim_time_ms)
        all_train_loss.extend(train_loss_history)
        all_test_loss.extend(test_loss_history)

    with open("train_loss.pickle", "wb") as file:
        pickle.dump(all_train_loss, file)
    with open("test_loss.pickle", "wb") as file:
        pickle.dump(all_test_loss, file)
    torch.save(model, "nak_model.pt")

