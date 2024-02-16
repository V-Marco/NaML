import numpy as np

class NaP_Kdr:

    def __init__(self, V_init = -20, n_init = 0.1) -> None:
        self.C = 1
        self.EL = -80
        self.gL = 8
        self.gNa = 20
        self.gK = 10
        self.ENa = 60
        self.EK = -90

        self.V = V_init
        self.n = n_init
        self.dt = 0.01

    def update_n(self):
        self.n += (self.n_inf(self.V) - self.n) / self.tau(self.V) * self.dt

    def update_V(self, I):
        V_dot = I - self.gL * (self.V - self.EL) - self.gNa * self.m_inf(self.V) * (self.V - self.ENa) - self.gK * self.n * (self.V - self.EK)
        V_dot /= self.C
        self.V += V_dot * self.dt
    
    def m_inf(self, V):
        V_half = -20
        k = 15
        return 1 / (1 + np.exp((V_half - V) / k))
    
    def n_inf(self, V):
        V_half = -25
        k = 5
        return 1 / (1 + np.exp((V_half - V) / k))
    
    def tau(self, V):
        return 0.152
    
    def simulate(self, I):
        I_dt = np.repeat(I, int(1 / self.dt))
        V_out = np.zeros_like(I_dt)
        for i in range(len(V_out)):
            self.update_n()
            self.update_V(I_dt[i])
            V_out[i] = self.V
        return V_out[::int(1 / self.dt)]