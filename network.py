import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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

class TCNN(torch.nn.Module):

    def __init__(self, in_channels, kernel_size, num_layers, inter_channels = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_layers = torch.nn.ModuleList()
        if num_layers == 1:
            self.conv_layers.append(torch.nn.Conv1d(
                in_channels = in_channels, 
                out_channels = 1, 
                kernel_size = kernel_size, 
                padding = 'valid'))
        else:
            self.conv_layers.append(torch.nn.Conv1d(
                in_channels = in_channels, 
                out_channels = inter_channels, 
                kernel_size = kernel_size, 
                padding = 'valid'))
            for _ in range(num_layers - 2):
                self.conv_layers.append(torch.nn.Conv1d(
                in_channels = inter_channels, 
                out_channels = inter_channels, 
                kernel_size = kernel_size, 
                padding = 'valid'))
            self.conv_layers.append(torch.nn.Conv1d(
                in_channels = inter_channels, 
                out_channels = 1, 
                kernel_size = kernel_size, 
                padding = 'valid'))
        
        # Post-processing layers
        self.spike_conv = torch.nn.Conv1d(
            in_channels = 1, 
            out_channels = 1, 
            kernel_size = 1, 
            padding = "valid")
        self.v_conv = torch.nn.Conv1d(
            in_channels = 1, 
            out_channels = 1, 
            kernel_size = 1, 
            padding = "valid")
    
    def forward(self, I_input):
        out = I_input
        for layer in self.conv_layers:
            # Causal padding, so that the kernel uses [... t-1] to predict [t]
            out = torch.nn.functional.pad(out, (self.kernel_size - 1, 0))
            out = layer(out)
        
        # Spike log probs
        spike_out = self.spike_conv(out)
        spike_out = torch.nn.functional.logsigmoid(spike_out)

        # Voltage
        v_out = self.v_conv(out)

        return v_out, spike_out
        

def train(
        num_epoch, 
        model, 
        device_name, 
        lr, 
        I_train, 
        V_train, 
        spike_times_train, 
        I_test, 
        V_test, 
        spike_times_test, 
        sim_time_ms):

    device = torch.device(device_name)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    I_train = torch.Tensor(I_train).float()
    V_train = torch.tensor(V_train).float()
    spike_times_train = torch.tensor(spike_times_train).float()
    I_test = torch.Tensor(I_test).float()
    V_test = torch.tensor(V_test).float()
    spike_times_test = torch.tensor(spike_times_test).float()

    train_loss_history = []
    test_loss_history = []
    for epoch in range(num_epoch):
        
        model.train(True)
        model.zero_grad()

        V_out, spike_log_probs = model(I_train)
        V_out = V_out[:, 0, :].float()
        spike_log_probs = spike_log_probs[:, 0, :].float()

        loss_spike = 0
        for t in range(sim_time_ms):
            loss_spike += torch.nn.functional.binary_cross_entropy_with_logits(spike_log_probs[:, t], spike_times_train[:, t])

        loss_v = torch.nn.functional.mse_loss(V_out, V_train)
        full_train_loss = loss_spike.float() + loss_v.float()

        full_train_loss.backward()
        optimizer.step()

        model.train(False)
        V_out, spike_log_probs = model(I_test)
        V_out = V_out[:, 0, :].float()
        spike_log_probs = spike_log_probs[:, 0, :].float()

        loss_spike = 0
        for t in range(sim_time_ms):
            loss_spike += torch.nn.functional.binary_cross_entropy_with_logits(spike_log_probs[:, t], spike_times_test[:, t])
        
        acc = accuracy_score(spike_times_test.detach().numpy().flatten(), spike_log_probs.detach().numpy().flatten() > 0.5)
        try:
            auc_roc = roc_auc_score(spike_times_test.detach().numpy().flatten(), np.exp(spike_log_probs.detach().numpy().flatten()))
        except:
            auc_roc = -1

        loss_v = torch.nn.functional.mse_loss(V_out, V_test)
        full_test_loss = loss_spike.float() + loss_v.float()

        print(f"""[Epoch {epoch}]:
              train_loss = {full_train_loss.detach().cpu().numpy()},
              test_MSE = {loss_v.detach().cpu().numpy()},
              test_BCE = {loss_spike.detach().cpu().numpy()},
              test_loss = {full_test_loss.detach().cpu().numpy()},
              test_acc = {acc}
              test_AUC = {auc_roc}""")
        train_loss_history.append(full_train_loss.detach().cpu().numpy())
        test_loss_history.append(full_test_loss.detach().cpu().numpy())

    return train_loss_history, test_loss_history