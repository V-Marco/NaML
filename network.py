import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    
    I_train = torch.Tensor(I_train).float().to(device)
    V_train = torch.tensor(V_train).float().to(device)
    spike_times_train = torch.tensor(spike_times_train).float().to(device)
    I_test = torch.Tensor(I_test).float().to(device)
    V_test = torch.tensor(V_test).float().to(device)
    spike_times_test = torch.tensor(spike_times_test).float().to(device)

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
        
        acc = accuracy_score(spike_times_test.cpu().detach().numpy().flatten(), spike_log_probs.cpu().detach().numpy().flatten() > 0.5)
        auc_roc = roc_auc_score(spike_times_test.cpu().detach().numpy().flatten(), np.exp(spike_log_probs.cpu().detach().numpy().flatten()))

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
