import sys, os
import numpy as np
import torch
import pickle

sys.path.append("../")
from network import train, count_parameters
from analysis import DataReader

from scipy.signal.windows import exponential
from scipy.signal import convolve


N_syn = 30076
t_stop = 4000
win_size = 100
device = 'cuda'

train_sim_range = (0, 4800)
test_sim_range = (4800, 5000)

dt = 0.1
save_every_ms = 1000

num_passes = 3

def get_I_input(path):
    with open(os.path.join(path, "exc_spike_trains.pickle"), "rb") as file:
        exc_st = pickle.load(file)
    with open(os.path.join(path, "inh_spike_trains.pickle"), "rb") as file:
        inh_st = pickle.load(file)
    with open(os.path.join(path, "soma_spike_trains.pickle"), "rb") as file:
        soma_st = pickle.load(file)

    I_input = np.zeros((N_syn, t_stop))
    win = exponential(M = 141, center = 0, tau = 20, sym = False)
    for i in range(N_syn):
        if len(exc_st[i]) > 0:
            I_input[i, exc_st[i]] = 1
        elif len(inh_st[i]) > 0:
            I_input[i, inh_st[i]] = 1
        elif len(soma_st[i]) > 0:
            I_input[i, soma_st[i]] = 1
        I_input[i] = convolve(I_input[i], win, "full")[:t_stop]
    
    return I_input.reshape((1, N_syn, t_stop))

def get_V_output(path):
    V_ouput = DataReader.read_data(path, "v.h5", save_every_ms, dt, t_stop)[0]
    return V_ouput.reshape(1, t_stop)

def get_spike_times(path):
    spike_times = DataReader.read_data(path, "soma_spikes.h5", save_every_ms, dt, t_stop)
    spike_times_cont = np.zeros(t_stop)
    spike_times_cont[spike_times.astype('int')] = 1
    return spike_times_cont.reshape(1, t_stop)

class IFDNN(torch.nn.Module):

    def __init__(self, N_syn, win_size):
        super().__init__()
        self.win_size = win_size
        self.conv1 = torch.nn.Conv1d(in_channels = N_syn, out_channels = 8, kernel_size = self.win_size, padding = "valid")
        self.conv2 = torch.nn.Conv1d(in_channels = 8, out_channels = 8, kernel_size = self.win_size, padding = "valid")
        self.conv3 = torch.nn.Conv1d(in_channels = 8, out_channels = 1, kernel_size = self.win_size, padding = "valid")
        self.spike_conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1, padding = "valid")
        self.v_conv = torch.nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 1, padding = "valid")

    def forward(self, I_input):
        # Causal padding, so that the kernel uses [... t-1] to predict [t]
        I_input = torch.nn.functional.pad(I_input, (self.win_size - 1, 0))
        out = self.conv1(I_input)
        out = torch.nn.functional.pad(out, (self.win_size - 1, 0))
        out = self.conv2(out)
        out = torch.nn.functional.pad(out, (self.win_size - 1, 0))
        out = self.conv3(out)

        spike_out = self.spike_conv(out)
        spike_out = torch.nn.functional.logsigmoid(spike_out)

        v_out = self.v_conv(out)

        return v_out, spike_out
    
if __name__ == "__main__":
    
    model = IFDNN(N_syn = N_syn, win_size = win_size)
    print("Number of parameters: ", count_parameters(model))

    all_train_loss = []
    all_test_loss = []

    for pass_num in range(num_passes):
        print("Pass: ", pass_num)
        print("----------")
        for sim_num in range(*train_sim_range):

            train_path = f'sims/sim_{sim_num}'
            I_input_train = get_I_input(train_path)
            V_ouput_train = get_V_output(train_path)
            spike_times_train = get_spike_times(train_path)

            # Choose a random simulation from the test pool
            test_path = f'sims/sim_{np.random.randint(*test_sim_range)}'
            I_input_test = get_I_input(test_path)
            V_ouput_test = get_V_output(test_path)
            spike_times_test = get_spike_times(test_path)

            train_loss_history, test_loss_history = train(
                num_epoch = 1,
                model = model,
                device_name = device,
                lr = 0.0001,
                I_train = I_input_train,
                V_train = V_ouput_train,
                spike_times_train = spike_times_train,
                I_test = I_input_test,
                V_test = V_ouput_test,
                spike_times_test = spike_times_test,
                sim_time_ms = t_stop
            )

            all_train_loss.extend(train_loss_history)
            all_test_loss.extend(test_loss_history)
    
            with open("train_loss.pickle", "wb") as file:
                pickle.dump(all_train_loss, file)
            with open("test_loss.pickle", "wb") as file:
                pickle.dump(all_test_loss, file)
            torch.save(model, "L5_model.pt")
