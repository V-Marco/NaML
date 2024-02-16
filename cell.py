import numpy as np
from neuron import h

class Cell:

    def __init__(self):

        # Create sections
        self.soma = h.Section(name = 'soma')

        # Geometry & biophysics
        self.soma.L = 50 # (um)
        self.soma.diam = 30 # (um)
        self.soma.Ra = 100 # (ohm-cm)
        self.soma.nseg = 1
        self.soma.cm = 1 # (uF/cm2)

        # Channels 
        self.soma.insert('pas')
        self.soma(0.5).pas.e = -90 # (mV)
        self.soma(0.5).pas.g = 0.0000338 # (S/cm2)

        # Connectivity
        self.exc_synapses = []
        self.inh_synapses = []
        self.netcons = []
        
        # --- Recorders

        # Spikes
        self.spike_detector = h.NetCon(self.soma(0.5)._ref_v, None, sec = self.soma)
        self.spike_times = h.Vector()
        self.spike_detector.record(self.spike_times)

        # Soma voltage
        self.V = h.Vector().record(self.soma(0.5)._ref_v)

        # Current
        self._ci = []
        self.I = []

    def set_synapses(self, N_exc, N_inh) -> None:
        for _ in range(N_exc): self.exc_synapses.append(h.Exp2Syn(self.soma(0.5)))
        for _ in range(N_inh): self.inh_synapses.append(h.Exp2Syn(self.soma(0.5)))
        for syn in self.exc_synapses: syn.e = 0
        for syn in self.inh_synapses: syn.e = -90
        
    def set_I(self, params: list) -> None:
        '''
        params: list
            [(amp, dur, delay)]
        '''
        for i in range(len(params)):
            amp, dur, delay = params[i]
            cur_inj = h.IClamp(self.soma(0.5))
            cur_inj.amp = amp
            cur_inj.dur = dur
            cur_inj.delay = delay
            self.I.append(h.Vector().record(cur_inj._ref_i))
            self._ci.append(cur_inj)

    def get_V(self):
        return self.V.as_numpy()

    def get_I(self):
        full_I = np.hstack([I_vec.as_numpy() for I_vec in self.I]).flatten()
        return full_I
    
    def get_spike_times(self):
        return self.spike_times.as_numpy().astype(int)

class NaP_K_Cell(Cell):

    def __init__(self):
        super().__init__()
        self.soma.insert("kdr")
        self.soma.insert("nap")