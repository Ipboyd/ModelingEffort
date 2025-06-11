
import torch
import torch.nn as nn
import genPoissonTimes
import genPoissonInputs
import matplotlib.pyplot as plt
import pdb
from memory_profiler import profile
import gc
from torch.cuda.amp import autocast
import torch.profiler


#torch.autograd.set_detect_anomaly(True)


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, prev, threshold):
        ctx.save_for_backward(input)
        #if((input >= threshold) and (prev < threshold)):
        #print(((input >= threshold) and (prev < threshold)).float())
        return ((input >= threshold) and (prev < threshold)).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (1.0 / (1.0 + torch.abs(input)) ** 2)
        return grad_input, None, None



class LIF_ODE(nn.Module):
    def __init__(self):
        super().__init__()
        
        

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(self.device)
        #print(trial_num)

        # Learnable Parameters

        self.R1On_On_PSC_gSYN = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))
        self.S1OnOff_On_PSC_gSYN = nn.Parameter(torch.tensor(0.085, dtype=torch.float32))
        self.R1On_S1OnOff_PSC_gSYN = nn.Parameter(torch.tensor(0.025, dtype=torch.float32))
        self.R1Off_S1OnOff_PSC_gSYN = nn.Parameter(torch.tensor(0.025, dtype=torch.float32))
        self.R1Off_Off_PSC_gSYN = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))
        self.S1OnOff_Off_PSC_gSYN = nn.Parameter(torch.tensor(0.045, dtype=torch.float32))
        self.R2On_R1On_PSC_gSYN = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))
        self.S2OnOff_R1On_PSC_gSYN = nn.Parameter(torch.tensor(0.085, dtype=torch.float32))
        self.R2On_S2OnOff_PSC_gSYN = nn.Parameter(torch.tensor(0.025, dtype=torch.float32))
        self.R2Off_S2OnOff_PSC_gSYN = nn.Parameter(torch.tensor(0.025, dtype=torch.float32))
        self.R2Off_R1Off_PSC_gSYN = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))
        self.S2OnOff_R1Off_PSC_gSYN = nn.Parameter(torch.tensor(0.045, dtype=torch.float32))

        # Non-learnable Parameters
        self.tspan = torch.tensor([0.1, 3500.0], dtype=torch.float32)
        self.dt = torch.tensor(0.1, dtype=torch.float32)
        self.On_C = torch.tensor(0.1, dtype=torch.float32)
        self.On_g_L = torch.tensor(0.005, dtype=torch.float32)
        self.On_E_L = torch.tensor(-65.0, dtype=torch.float32)
        self.On_noise = torch.tensor(0.0, dtype=torch.float32)
        self.On_t_ref = torch.tensor(1.0, dtype=torch.float32)
        self.On_E_k = torch.tensor(-80.0, dtype=torch.float32)
        self.On_tau_ad = torch.tensor(5.0, dtype=torch.float32)
        self.On_g_inc = torch.tensor(0.0, dtype=torch.float32)
        self.On_Itonic = torch.tensor(0.0, dtype=torch.float32)
        self.On_V_thresh = torch.tensor(-47.0, dtype=torch.float32)
        self.On_V_reset = torch.tensor(-54.0, dtype=torch.float32)
        self.On_Npop = int(1.0)
        self.Off_C = torch.tensor(0.1, dtype=torch.float32)
        self.Off_g_L = torch.tensor(0.005, dtype=torch.float32)
        self.Off_E_L = torch.tensor(-65.0, dtype=torch.float32)
        self.Off_noise = torch.tensor(0.0, dtype=torch.float32)
        self.Off_t_ref = torch.tensor(1.0, dtype=torch.float32)
        self.Off_E_k = torch.tensor(-80.0, dtype=torch.float32)
        self.Off_tau_ad = torch.tensor(5.0, dtype=torch.float32)
        self.Off_g_inc = torch.tensor(0.0, dtype=torch.float32)
        self.Off_Itonic = torch.tensor(0.0, dtype=torch.float32)
        self.Off_V_thresh = torch.tensor(-47.0, dtype=torch.float32)
        self.Off_V_reset = torch.tensor(-54.0, dtype=torch.float32)
        self.Off_Npop = int(1.0)
        self.R1On_C = torch.tensor(0.1, dtype=torch.float32)
        self.R1On_g_L = torch.tensor(0.005, dtype=torch.float32)
        self.R1On_E_L = torch.tensor(-65.0, dtype=torch.float32)
        self.R1On_noise = torch.tensor(0.0, dtype=torch.float32)
        self.R1On_t_ref = torch.tensor(1.0, dtype=torch.float32)
        self.R1On_E_k = torch.tensor(-80.0, dtype=torch.float32)
        self.R1On_tau_ad = torch.tensor(100.0, dtype=torch.float32)
        self.R1On_g_inc = torch.tensor(0.0003, dtype=torch.float32)
        self.R1On_Itonic = torch.tensor(0.0, dtype=torch.float32)
        self.R1On_V_thresh = torch.tensor(-47.0, dtype=torch.float32)
        self.R1On_V_reset = torch.tensor(-54.0, dtype=torch.float32)
        self.R1On_Npop = int(1.0)
        self.R1Off_C = torch.tensor(0.1, dtype=torch.float32)
        self.R1Off_g_L = torch.tensor(0.005, dtype=torch.float32)
        self.R1Off_E_L = torch.tensor(-65.0, dtype=torch.float32)
        self.R1Off_noise = torch.tensor(0.0, dtype=torch.float32)
        self.R1Off_t_ref = torch.tensor(1.0, dtype=torch.float32)
        self.R1Off_E_k = torch.tensor(-80.0, dtype=torch.float32)
        self.R1Off_tau_ad = torch.tensor(100.0, dtype=torch.float32)
        self.R1Off_g_inc = torch.tensor(0.0003, dtype=torch.float32)
        self.R1Off_Itonic = torch.tensor(0.0, dtype=torch.float32)
        self.R1Off_V_thresh = torch.tensor(-47.0, dtype=torch.float32)
        self.R1Off_V_reset = torch.tensor(-54.0, dtype=torch.float32)
        self.R1Off_Npop = int(1.0)
        self.S1OnOff_C = torch.tensor(0.1, dtype=torch.float32)
        self.S1OnOff_g_L = torch.tensor(0.01, dtype=torch.float32)
        self.S1OnOff_E_L = torch.tensor(-57.0, dtype=torch.float32)
        self.S1OnOff_noise = torch.tensor(0.0, dtype=torch.float32)
        self.S1OnOff_t_ref = torch.tensor(0.5, dtype=torch.float32)
        self.S1OnOff_E_k = torch.tensor(-80.0, dtype=torch.float32)
        self.S1OnOff_tau_ad = torch.tensor(5.0, dtype=torch.float32)
        self.S1OnOff_g_inc = torch.tensor(0.0, dtype=torch.float32)
        self.S1OnOff_Itonic = torch.tensor(0.0, dtype=torch.float32)
        self.S1OnOff_V_thresh = torch.tensor(-47.0, dtype=torch.float32)
        self.S1OnOff_V_reset = torch.tensor(-52.0, dtype=torch.float32)
        self.S1OnOff_Npop = int(1.0)
        self.R2On_C = torch.tensor(0.1, dtype=torch.float32)
        self.R2On_g_L = torch.tensor(0.005, dtype=torch.float32)
        self.R2On_E_L = torch.tensor(-65.0, dtype=torch.float32)
        self.R2On_noise = torch.tensor(0.0, dtype=torch.float32)
        self.R2On_t_ref = torch.tensor(1.0, dtype=torch.float32)
        self.R2On_E_k = torch.tensor(-80.0, dtype=torch.float32)
        self.R2On_tau_ad = torch.tensor(100.0, dtype=torch.float32)
        self.R2On_g_inc = torch.tensor(0.0003, dtype=torch.float32)
        self.R2On_Itonic = torch.tensor(0.0, dtype=torch.float32)
        self.R2On_V_thresh = torch.tensor(-47.0, dtype=torch.float32)
        self.R2On_V_reset = torch.tensor(-54.0, dtype=torch.float32)
        self.R2On_Npop = int(1.0)
        self.R2Off_C = torch.tensor(0.1, dtype=torch.float32)
        self.R2Off_g_L = torch.tensor(0.005, dtype=torch.float32)
        self.R2Off_E_L = torch.tensor(-65.0, dtype=torch.float32)
        self.R2Off_noise = torch.tensor(0.0, dtype=torch.float32)
        self.R2Off_t_ref = torch.tensor(1.0, dtype=torch.float32)
        self.R2Off_E_k = torch.tensor(-80.0, dtype=torch.float32)
        self.R2Off_tau_ad = torch.tensor(100.0, dtype=torch.float32)
        self.R2Off_g_inc = torch.tensor(0.0003, dtype=torch.float32)
        self.R2Off_Itonic = torch.tensor(0.0, dtype=torch.float32)
        self.R2Off_V_thresh = torch.tensor(-47.0, dtype=torch.float32)
        self.R2Off_V_reset = torch.tensor(-54.0, dtype=torch.float32)
        self.R2Off_Npop = int(1.0)
        self.S2OnOff_C = torch.tensor(0.1, dtype=torch.float32)
        self.S2OnOff_g_L = torch.tensor(0.01, dtype=torch.float32)
        self.S2OnOff_E_L = torch.tensor(-57.0, dtype=torch.float32)
        self.S2OnOff_noise = torch.tensor(0.0, dtype=torch.float32)
        self.S2OnOff_t_ref = torch.tensor(0.5, dtype=torch.float32)
        self.S2OnOff_E_k = torch.tensor(-80.0, dtype=torch.float32)
        self.S2OnOff_tau_ad = torch.tensor(5.0, dtype=torch.float32)
        self.S2OnOff_g_inc = torch.tensor(0.0, dtype=torch.float32)
        self.S2OnOff_Itonic = torch.tensor(0.0, dtype=torch.float32)
        self.S2OnOff_V_thresh = torch.tensor(-47.0, dtype=torch.float32)
        self.S2OnOff_V_reset = torch.tensor(-52.0, dtype=torch.float32)
        self.S2OnOff_Npop = int(1.0)
        self.On_On_IC_locNum = torch.tensor(15.0, dtype=torch.float32)
        self.On_On_IC_label = 'on'
        self.On_On_IC_t_ref = torch.tensor(1.0, dtype=torch.float32)
        self.On_On_IC_t_ref_rel = torch.tensor(1.0, dtype=torch.float32)
        self.On_On_IC_rec = torch.tensor(2.0, dtype=torch.float32)
        self.On_On_IC_g_postIC = torch.tensor(0.17, dtype=torch.float32)
        self.On_On_IC_E_exc = torch.tensor(0.0, dtype=torch.float32)
        self.Off_Off_IC_locNum = torch.tensor(15.0, dtype=torch.float32)
        self.Off_Off_IC_label = 'off'
        self.Off_Off_IC_t_ref = torch.tensor(1.0, dtype=torch.float32)
        self.Off_Off_IC_t_ref_rel = torch.tensor(1.0, dtype=torch.float32)
        self.Off_Off_IC_rec = torch.tensor(2.0, dtype=torch.float32)
        self.Off_Off_IC_g_postIC = torch.tensor(0.17, dtype=torch.float32)
        self.Off_Off_IC_E_exc = torch.tensor(0.0, dtype=torch.float32)
        self.R1On_On_PSC_ESYN = torch.tensor(0.0, dtype=torch.float32)
        self.R1On_On_PSC_tauD = torch.tensor(1.5, dtype=torch.float32)
        self.R1On_On_PSC_tauR = torch.tensor(0.7, dtype=torch.float32)
        self.R1On_On_PSC_delay = torch.tensor(0.0, dtype=torch.float32)
        self.R1On_On_PSC_fF = torch.tensor(0.0, dtype=torch.float32)
        self.R1On_On_PSC_fP = torch.tensor(0.1, dtype=torch.float32)
        self.R1On_On_PSC_tauF = torch.tensor(180.0, dtype=torch.float32)
        self.R1On_On_PSC_tauP = torch.tensor(30.0, dtype=torch.float32)
        self.R1On_On_PSC_maxF = torch.tensor(4.0, dtype=torch.float32)
        self.S1OnOff_On_PSC_ESYN = torch.tensor(0.0, dtype=torch.float32)
        self.S1OnOff_On_PSC_tauD = torch.tensor(1.0, dtype=torch.float32)
        self.S1OnOff_On_PSC_tauR = torch.tensor(0.1, dtype=torch.float32)
        self.S1OnOff_On_PSC_delay = torch.tensor(0.0, dtype=torch.float32)
        self.S1OnOff_On_PSC_fF = torch.tensor(0.0, dtype=torch.float32)
        self.S1OnOff_On_PSC_fP = torch.tensor(0.2, dtype=torch.float32)
        self.S1OnOff_On_PSC_tauF = torch.tensor(180.0, dtype=torch.float32)
        self.S1OnOff_On_PSC_tauP = torch.tensor(80.0, dtype=torch.float32)
        self.S1OnOff_On_PSC_maxF = torch.tensor(4.0, dtype=torch.float32)
        self.R1On_S1OnOff_PSC_ESYN = torch.tensor(-80.0, dtype=torch.float32)
        self.R1On_S1OnOff_PSC_tauD = torch.tensor(4.5, dtype=torch.float32)
        self.R1On_S1OnOff_PSC_tauR = torch.tensor(1.0, dtype=torch.float32)
        self.R1On_S1OnOff_PSC_delay = torch.tensor(0.0, dtype=torch.float32)
        self.R1On_S1OnOff_PSC_fF = torch.tensor(0.0, dtype=torch.float32)
        self.R1On_S1OnOff_PSC_fP = torch.tensor(0.5, dtype=torch.float32)
        self.R1On_S1OnOff_PSC_tauF = torch.tensor(180.0, dtype=torch.float32)
        self.R1On_S1OnOff_PSC_tauP = torch.tensor(120.0, dtype=torch.float32)
        self.R1On_S1OnOff_PSC_maxF = torch.tensor(4.0, dtype=torch.float32)
        self.R1Off_S1OnOff_PSC_ESYN = torch.tensor(-80.0, dtype=torch.float32)
        self.R1Off_S1OnOff_PSC_tauD = torch.tensor(4.5, dtype=torch.float32)
        self.R1Off_S1OnOff_PSC_tauR = torch.tensor(1.0, dtype=torch.float32)
        self.R1Off_S1OnOff_PSC_delay = torch.tensor(0.0, dtype=torch.float32)
        self.R1Off_S1OnOff_PSC_fF = torch.tensor(0.0, dtype=torch.float32)
        self.R1Off_S1OnOff_PSC_fP = torch.tensor(0.5, dtype=torch.float32)
        self.R1Off_S1OnOff_PSC_tauF = torch.tensor(180.0, dtype=torch.float32)
        self.R1Off_S1OnOff_PSC_tauP = torch.tensor(120.0, dtype=torch.float32)
        self.R1Off_S1OnOff_PSC_maxF = torch.tensor(4.0, dtype=torch.float32)
        self.R1Off_Off_PSC_ESYN = torch.tensor(0.0, dtype=torch.float32)
        self.R1Off_Off_PSC_tauD = torch.tensor(1.5, dtype=torch.float32)
        self.R1Off_Off_PSC_tauR = torch.tensor(0.7, dtype=torch.float32)
        self.R1Off_Off_PSC_delay = torch.tensor(0.0, dtype=torch.float32)
        self.R1Off_Off_PSC_fF = torch.tensor(0.0, dtype=torch.float32)
        self.R1Off_Off_PSC_fP = torch.tensor(0.1, dtype=torch.float32)
        self.R1Off_Off_PSC_tauF = torch.tensor(180.0, dtype=torch.float32)
        self.R1Off_Off_PSC_tauP = torch.tensor(30.0, dtype=torch.float32)
        self.R1Off_Off_PSC_maxF = torch.tensor(4.0, dtype=torch.float32)
        self.S1OnOff_Off_PSC_ESYN = torch.tensor(0.0, dtype=torch.float32)
        self.S1OnOff_Off_PSC_tauD = torch.tensor(1.0, dtype=torch.float32)
        self.S1OnOff_Off_PSC_tauR = torch.tensor(0.1, dtype=torch.float32)
        self.S1OnOff_Off_PSC_delay = torch.tensor(0.0, dtype=torch.float32)
        self.S1OnOff_Off_PSC_fF = torch.tensor(0.0, dtype=torch.float32)
        self.S1OnOff_Off_PSC_fP = torch.tensor(0.0, dtype=torch.float32)
        self.S1OnOff_Off_PSC_tauF = torch.tensor(180.0, dtype=torch.float32)
        self.S1OnOff_Off_PSC_tauP = torch.tensor(80.0, dtype=torch.float32)
        self.S1OnOff_Off_PSC_maxF = torch.tensor(4.0, dtype=torch.float32)
        self.R2On_R1On_PSC_ESYN = torch.tensor(0.0, dtype=torch.float32)
        self.R2On_R1On_PSC_tauD = torch.tensor(1.5, dtype=torch.float32)
        self.R2On_R1On_PSC_tauR = torch.tensor(0.7, dtype=torch.float32)
        self.R2On_R1On_PSC_delay = torch.tensor(0.0, dtype=torch.float32)
        self.R2On_R1On_PSC_fF = torch.tensor(0.0, dtype=torch.float32)
        self.R2On_R1On_PSC_fP = torch.tensor(0.1, dtype=torch.float32)
        self.R2On_R1On_PSC_tauF = torch.tensor(180.0, dtype=torch.float32)
        self.R2On_R1On_PSC_tauP = torch.tensor(30.0, dtype=torch.float32)
        self.R2On_R1On_PSC_maxF = torch.tensor(4.0, dtype=torch.float32)
        self.S2OnOff_R1On_PSC_ESYN = torch.tensor(0.0, dtype=torch.float32)
        self.S2OnOff_R1On_PSC_tauD = torch.tensor(1.0, dtype=torch.float32)
        self.S2OnOff_R1On_PSC_tauR = torch.tensor(0.1, dtype=torch.float32)
        self.S2OnOff_R1On_PSC_delay = torch.tensor(0.0, dtype=torch.float32)
        self.S2OnOff_R1On_PSC_fF = torch.tensor(0.0, dtype=torch.float32)
        self.S2OnOff_R1On_PSC_fP = torch.tensor(0.2, dtype=torch.float32)
        self.S2OnOff_R1On_PSC_tauF = torch.tensor(180.0, dtype=torch.float32)
        self.S2OnOff_R1On_PSC_tauP = torch.tensor(80.0, dtype=torch.float32)
        self.S2OnOff_R1On_PSC_maxF = torch.tensor(4.0, dtype=torch.float32)
        self.R2On_S2OnOff_PSC_ESYN = torch.tensor(-80.0, dtype=torch.float32)
        self.R2On_S2OnOff_PSC_tauD = torch.tensor(4.5, dtype=torch.float32)
        self.R2On_S2OnOff_PSC_tauR = torch.tensor(1.0, dtype=torch.float32)
        self.R2On_S2OnOff_PSC_delay = torch.tensor(0.0, dtype=torch.float32)
        self.R2On_S2OnOff_PSC_fF = torch.tensor(0.0, dtype=torch.float32)
        self.R2On_S2OnOff_PSC_fP = torch.tensor(0.5, dtype=torch.float32)
        self.R2On_S2OnOff_PSC_tauF = torch.tensor(180.0, dtype=torch.float32)
        self.R2On_S2OnOff_PSC_tauP = torch.tensor(120.0, dtype=torch.float32)
        self.R2On_S2OnOff_PSC_maxF = torch.tensor(4.0, dtype=torch.float32)
        self.R2Off_S2OnOff_PSC_ESYN = torch.tensor(-80.0, dtype=torch.float32)
        self.R2Off_S2OnOff_PSC_tauD = torch.tensor(4.5, dtype=torch.float32)
        self.R2Off_S2OnOff_PSC_tauR = torch.tensor(1.0, dtype=torch.float32)
        self.R2Off_S2OnOff_PSC_delay = torch.tensor(0.0, dtype=torch.float32)
        self.R2Off_S2OnOff_PSC_fF = torch.tensor(0.0, dtype=torch.float32)
        self.R2Off_S2OnOff_PSC_fP = torch.tensor(0.5, dtype=torch.float32)
        self.R2Off_S2OnOff_PSC_tauF = torch.tensor(180.0, dtype=torch.float32)
        self.R2Off_S2OnOff_PSC_tauP = torch.tensor(120.0, dtype=torch.float32)
        self.R2Off_S2OnOff_PSC_maxF = torch.tensor(4.0, dtype=torch.float32)
        self.R2Off_R1Off_PSC_ESYN = torch.tensor(0.0, dtype=torch.float32)
        self.R2Off_R1Off_PSC_tauD = torch.tensor(1.5, dtype=torch.float32)
        self.R2Off_R1Off_PSC_tauR = torch.tensor(0.7, dtype=torch.float32)
        self.R2Off_R1Off_PSC_delay = torch.tensor(0.0, dtype=torch.float32)
        self.R2Off_R1Off_PSC_fF = torch.tensor(0.0, dtype=torch.float32)
        self.R2Off_R1Off_PSC_fP = torch.tensor(0.1, dtype=torch.float32)
        self.R2Off_R1Off_PSC_tauF = torch.tensor(180.0, dtype=torch.float32)
        self.R2Off_R1Off_PSC_tauP = torch.tensor(30.0, dtype=torch.float32)
        self.R2Off_R1Off_PSC_maxF = torch.tensor(4.0, dtype=torch.float32)
        self.S2OnOff_R1Off_PSC_ESYN = torch.tensor(0.0, dtype=torch.float32)
        self.S2OnOff_R1Off_PSC_tauD = torch.tensor(1.0, dtype=torch.float32)
        self.S2OnOff_R1Off_PSC_tauR = torch.tensor(0.1, dtype=torch.float32)
        self.S2OnOff_R1Off_PSC_delay = torch.tensor(0.0, dtype=torch.float32)
        self.S2OnOff_R1Off_PSC_fF = torch.tensor(0.0, dtype=torch.float32)
        self.S2OnOff_R1Off_PSC_fP = torch.tensor(0.0, dtype=torch.float32)
        self.S2OnOff_R1Off_PSC_tauF = torch.tensor(180.0, dtype=torch.float32)
        self.S2OnOff_R1Off_PSC_tauP = torch.tensor(80.0, dtype=torch.float32)
        self.S2OnOff_R1Off_PSC_maxF = torch.tensor(4.0, dtype=torch.float32)
        self.R2On_R2On_iNoise_V3_FR = torch.tensor(8.0, dtype=torch.float32)
        self.R2On_R2On_iNoise_V3_sigma = torch.tensor(0.0, dtype=torch.float32)
        self.R2On_R2On_iNoise_V3_dt = torch.tensor(0.1, dtype=torch.float32)
        self.R2On_R2On_iNoise_V3_nSYN = torch.tensor(0.015, dtype=torch.float32)
        self.R2On_R2On_iNoise_V3_simlen = torch.tensor(35000.0, dtype=torch.float32)
        self.R2On_R2On_iNoise_V3_tauD_N = torch.tensor(1.5, dtype=torch.float32)
        self.R2On_R2On_iNoise_V3_tauR_N = torch.tensor(0.7, dtype=torch.float32)
        self.R2On_R2On_iNoise_V3_E_exc = torch.tensor(0.0, dtype=torch.float32)
        self.ROn_X_PSC3_netcon = torch.tensor(1.0, dtype=torch.float32)
        self.ROn_SOnOff_PSC3_netcon = torch.tensor(1.0, dtype=torch.float32)
        self.C_ROn_PSC3_netcon = torch.tensor(1.0, dtype=torch.float32)

        T = len(torch.arange(self.tspan[0],self.tspan[1],self.dt, dtype=torch.float32))
        #print(trial_num)

        # Fixed Params
        self.On_R = 1/self.On_g_L
        self.On_tau = self.On_C*self.On_R
        self.On_Imask = torch.ones(1,self.On_Npop)
        self.Off_R = 1/self.Off_g_L
        self.Off_tau = self.Off_C*self.Off_R
        self.Off_Imask = torch.ones(1,self.Off_Npop)
        self.R1On_R = 1/self.R1On_g_L
        self.R1On_tau = self.R1On_C*self.R1On_R
        self.R1On_Imask = torch.ones(1,self.R1On_Npop)
        self.R1Off_R = 1/self.R1Off_g_L
        self.R1Off_tau = self.R1Off_C*self.R1Off_R
        self.R1Off_Imask = torch.ones(1,self.R1Off_Npop)
        self.S1OnOff_R = 1/self.S1OnOff_g_L
        self.S1OnOff_tau = self.S1OnOff_C*self.S1OnOff_R
        self.S1OnOff_Imask = torch.ones(1,self.S1OnOff_Npop)
        self.R2On_R = 1/self.R2On_g_L
        self.R2On_tau = self.R2On_C*self.R2On_R
        self.R2On_Imask = torch.ones(1,self.R2On_Npop)
        self.R2Off_R = 1/self.R2Off_g_L
        self.R2Off_tau = self.R2Off_C*self.R2Off_R
        self.R2Off_Imask = torch.ones(1,self.R2Off_Npop)
        self.S2OnOff_R = 1/self.S2OnOff_g_L
        self.S2OnOff_tau = self.S2OnOff_C*self.S2OnOff_R
        self.S2OnOff_Imask = torch.ones(1,self.S2OnOff_Npop)
        self.On_On_IC_netcon = +1.000000000000000e+00
        self.Off_Off_IC_netcon = +1.000000000000000e+00
        self.R1On_On_PSC_netcon = torch.eye(self.On_Npop, self.R1On_Npop)
        self.R1On_On_PSC_scale = (self.R1On_On_PSC_tauD/self.R1On_On_PSC_tauR)**(self.R1On_On_PSC_tauR/(self.R1On_On_PSC_tauD-self.R1On_On_PSC_tauR))
        self.S1OnOff_On_PSC_netcon = torch.eye(self.On_Npop, self.S1OnOff_Npop)
        self.S1OnOff_On_PSC_scale = (self.S1OnOff_On_PSC_tauD/self.S1OnOff_On_PSC_tauR)**(self.S1OnOff_On_PSC_tauR/(self.S1OnOff_On_PSC_tauD-self.S1OnOff_On_PSC_tauR))
        self.R1On_S1OnOff_PSC_netcon = torch.eye(self.S1OnOff_Npop, self.R1On_Npop)
        self.R1On_S1OnOff_PSC_scale = (self.R1On_S1OnOff_PSC_tauD/self.R1On_S1OnOff_PSC_tauR)**(self.R1On_S1OnOff_PSC_tauR/(self.R1On_S1OnOff_PSC_tauD-self.R1On_S1OnOff_PSC_tauR))
        self.R1Off_S1OnOff_PSC_netcon = torch.eye(self.S1OnOff_Npop, self.R1Off_Npop)
        self.R1Off_S1OnOff_PSC_scale = (self.R1Off_S1OnOff_PSC_tauD/self.R1Off_S1OnOff_PSC_tauR)**(self.R1Off_S1OnOff_PSC_tauR/(self.R1Off_S1OnOff_PSC_tauD-self.R1Off_S1OnOff_PSC_tauR))
        self.R1Off_Off_PSC_netcon = torch.eye(self.Off_Npop, self.R1Off_Npop)
        self.R1Off_Off_PSC_scale = (self.R1Off_Off_PSC_tauD/self.R1Off_Off_PSC_tauR)**(self.R1Off_Off_PSC_tauR/(self.R1Off_Off_PSC_tauD-self.R1Off_Off_PSC_tauR))
        self.S1OnOff_Off_PSC_netcon = torch.eye(self.Off_Npop, self.S1OnOff_Npop)
        self.S1OnOff_Off_PSC_scale = (self.S1OnOff_Off_PSC_tauD/self.S1OnOff_Off_PSC_tauR)**(self.S1OnOff_Off_PSC_tauR/(self.S1OnOff_Off_PSC_tauD-self.S1OnOff_Off_PSC_tauR))
        self.R2On_R1On_PSC_netcon = torch.eye(self.R1On_Npop, self.R2On_Npop)
        self.R2On_R1On_PSC_scale = (self.R2On_R1On_PSC_tauD/self.R2On_R1On_PSC_tauR)**(self.R2On_R1On_PSC_tauR/(self.R2On_R1On_PSC_tauD-self.R2On_R1On_PSC_tauR))
        self.S2OnOff_R1On_PSC_netcon = torch.eye(self.R1On_Npop, self.S2OnOff_Npop)
        self.S2OnOff_R1On_PSC_scale = (self.S2OnOff_R1On_PSC_tauD/self.S2OnOff_R1On_PSC_tauR)**(self.S2OnOff_R1On_PSC_tauR/(self.S2OnOff_R1On_PSC_tauD-self.S2OnOff_R1On_PSC_tauR))
        self.R2On_S2OnOff_PSC_netcon = torch.eye(self.S2OnOff_Npop, self.R2On_Npop)
        self.R2On_S2OnOff_PSC_scale = (self.R2On_S2OnOff_PSC_tauD/self.R2On_S2OnOff_PSC_tauR)**(self.R2On_S2OnOff_PSC_tauR/(self.R2On_S2OnOff_PSC_tauD-self.R2On_S2OnOff_PSC_tauR))
        self.R2Off_S2OnOff_PSC_netcon = torch.eye(self.S2OnOff_Npop, self.R2Off_Npop)
        self.R2Off_S2OnOff_PSC_scale = (self.R2Off_S2OnOff_PSC_tauD/self.R2Off_S2OnOff_PSC_tauR)**(self.R2Off_S2OnOff_PSC_tauR/(self.R2Off_S2OnOff_PSC_tauD-self.R2Off_S2OnOff_PSC_tauR))
        self.R2Off_R1Off_PSC_netcon = torch.eye(self.R1Off_Npop, self.R2Off_Npop)
        self.R2Off_R1Off_PSC_scale = (self.R2Off_R1Off_PSC_tauD/self.R2Off_R1Off_PSC_tauR)**(self.R2Off_R1Off_PSC_tauR/(self.R2Off_R1Off_PSC_tauD-self.R2Off_R1Off_PSC_tauR))
        self.S2OnOff_R1Off_PSC_netcon = torch.eye(self.R1Off_Npop, self.S2OnOff_Npop)
        self.S2OnOff_R1Off_PSC_scale = (self.S2OnOff_R1Off_PSC_tauD/self.S2OnOff_R1Off_PSC_tauR)**(self.S2OnOff_R1Off_PSC_tauR/(self.S2OnOff_R1Off_PSC_tauD-self.S2OnOff_R1Off_PSC_tauR))
        self.R2On_R2On_iNoise_V3_netcon = torch.eye(self.R2On_Npop, self.R2On_Npop)
        self.R2On_R2On_iNoise_V3_token = genPoissonTimes.gen_poisson_times(self.R2On_Npop,self.R2On_R2On_iNoise_V3_dt,self.R2On_R2On_iNoise_V3_FR,self.R2On_R2On_iNoise_V3_sigma,self.R2On_R2On_iNoise_V3_simlen)
        self.R2On_R2On_iNoise_V3_scale = (self.R2On_R2On_iNoise_V3_tauD_N/self.R2On_R2On_iNoise_V3_tauR_N)**(self.R2On_R2On_iNoise_V3_tauR_N/(self.R2On_R2On_iNoise_V3_tauD_N-self.R2On_R2On_iNoise_V3_tauR_N))

    @profile
    def forward(self):
        
        #State Variables
            
        T = len(torch.arange(self.tspan[0],self.tspan[1],self.dt, dtype=torch.float32))
        helper = torch.arange(self.tspan[0],self.tspan[1],self.dt, dtype=torch.float32)

        


        #Monitors

        On_V_spikes = []
        Off_V_spikes = []
        R1On_V_spikes = []
        R1Off_V_spikes = []
        S1OnOff_V_spikes = []
        R2On_V_spikes = []
        R2Off_V_spikes = []
        S2OnOff_V_spikes = []


        #ODEs

        spike_holderOn = torch.full((T-1,),0.0)

        spike_holderOff = torch.full((T-1,),0.0)

        spike_holderR1On = torch.full((T-1,),0.0)

        spike_holderR2On = torch.full((T-1,),0.0)

        spike_holderR2Off = torch.full((T-1,),0.0)

        spike_holderR1Off = torch.full((T-1,),0.0)

        spike_holderS2OnOff = torch.full((T-1,),0.0)

        spike_holderS1OnOff = torch.full((T-1,),0.0)

        #On_V_spk_sum = torch.tensor(0.0)

        #Off_V_spk_sum = torch.tensor(0.0)

        #R1On_V_spk_sum = torch.tensor(0.0)

        #R1Off_V_spk_sum = torch.tensor(0.0)

        #R2On_V_spk_sum = torch.tensor(0.0)

        #R2Off_V_spk_sum = torch.tensor(0.0)

        #S1OnOff_V_spk_sum = torch.tensor(0.0)

        #S2OnOff_V_spk_sum = torch.tensor(0.0)

        for num_trials_count in range(2):

            #print('made it here')
            On_tspike = -1e32*torch.ones(5,self.On_Npop)
            On_buffer_index = torch.ones(1,self.On_Npop)
            On_V_spikes_holder = []
            Off_tspike = -1e32*torch.ones(5,self.Off_Npop)
            Off_buffer_index = torch.ones(1,self.Off_Npop)
            Off_V_spikes_holder = []
            R1On_tspike = -1e32*torch.ones(5,self.R1On_Npop)
            R1On_buffer_index = torch.ones(1,self.R1On_Npop)
            R1On_V_spikes_holder = []
            R1Off_tspike = -1e32*torch.ones(5,self.R1Off_Npop)
            R1Off_buffer_index = torch.ones(1,self.R1Off_Npop)
            R1Off_V_spikes_holder = []
            S1OnOff_tspike = -1e32*torch.ones(5,self.S1OnOff_Npop)
            S1OnOff_buffer_index = torch.ones(1,self.S1OnOff_Npop)
            S1OnOff_V_spikes_holder = []
            R2On_tspike = -1e32*torch.ones(5,self.R2On_Npop)
            R2On_buffer_index = torch.ones(1,self.R2On_Npop)
            R2On_V_spikes_holder = []
            R2Off_tspike = -1e32*torch.ones(5,self.R2Off_Npop)
            R2Off_buffer_index = torch.ones(1,self.R2Off_Npop)
            R2Off_V_spikes_holder = []
            S2OnOff_tspike = -1e32*torch.ones(5,self.S2OnOff_Npop)
            S2OnOff_buffer_index = torch.ones(1,self.S2OnOff_Npop)
            S2OnOff_V_spikes_holder = []
            On_On_IC_iIC = torch.zeros(T, self.On_Npop)
            Off_Off_IC_iIC = torch.zeros(T, self.Off_Npop)
            R1On_On_PSC_syn = torch.zeros(T, self.R1On_Npop)
            S1OnOff_On_PSC_syn = torch.zeros(T, self.S1OnOff_Npop)
            R1On_S1OnOff_PSC_syn = torch.zeros(T, self.R1On_Npop)
            R1Off_S1OnOff_PSC_syn = torch.zeros(T, self.R1Off_Npop)
            R1Off_Off_PSC_syn = torch.zeros(T, self.R1Off_Npop)
            S1OnOff_Off_PSC_syn = torch.zeros(T, self.S1OnOff_Npop)
            R2On_R1On_PSC_syn = torch.zeros(T, self.R2On_Npop)
            S2OnOff_R1On_PSC_syn = torch.zeros(T, self.S2OnOff_Npop)
            R2On_S2OnOff_PSC_syn = torch.zeros(T, self.R2On_Npop)
            R2Off_S2OnOff_PSC_syn = torch.zeros(T, self.R2Off_Npop)
            R2Off_R1Off_PSC_syn = torch.zeros(T, self.R2Off_Npop)
            S2OnOff_R1Off_PSC_syn = torch.zeros(T, self.S2OnOff_Npop)
            On_V = [self.On_E_L, self.On_E_L]
            On_g_ad = [0,0]
            Off_V = [self.Off_E_L, self.Off_E_L]
            Off_g_ad = [0,0]
            R1On_V = [self.R1On_E_L, self.R1On_E_L]
            R1On_g_ad = [0,0]
            R1Off_V = [self.R1Off_E_L, self.R1Off_E_L]
            R1Off_g_ad = [0,0]
            S1OnOff_V = [self.S1OnOff_E_L, self.S1OnOff_E_L]
            S1OnOff_g_ad = [0,0]
            R2On_V = [self.R2On_E_L, self.R2On_E_L]
            R2On_g_ad = [0,0]
            R2Off_V = [self.R2Off_E_L, self.R2Off_E_L]
            R2Off_g_ad = [0,0]
            S2OnOff_V = [self.S2OnOff_E_L, self.S2OnOff_E_L]
            S2OnOff_g_ad = [0,0]
            R1On_On_PSC_s = [0,0]
            R1On_On_PSC_x = [0,0]
            R1On_On_PSC_F = [1,1]
            R1On_On_PSC_P = [1,1]
            R1On_On_PSC_q = [1,1]
            S1OnOff_On_PSC_s = [0,0]
            S1OnOff_On_PSC_x = [0,0]
            S1OnOff_On_PSC_F = [1,1]
            S1OnOff_On_PSC_P = [1,1]
            S1OnOff_On_PSC_q = [1,1]
            R1On_S1OnOff_PSC_s = [0,0]
            R1On_S1OnOff_PSC_x = [0,0]
            R1On_S1OnOff_PSC_F = [1,1]
            R1On_S1OnOff_PSC_P = [1,1]
            R1On_S1OnOff_PSC_q = [1,1]
            R1Off_S1OnOff_PSC_s = [0,0]
            R1Off_S1OnOff_PSC_x = [0,0]
            R1Off_S1OnOff_PSC_F = [1,1]
            R1Off_S1OnOff_PSC_P = [1,1]
            R1Off_S1OnOff_PSC_q = [1,1]
            R1Off_Off_PSC_s = [0,0]
            R1Off_Off_PSC_x = [0,0]
            R1Off_Off_PSC_F = [1,1]
            R1Off_Off_PSC_P = [1,1]
            R1Off_Off_PSC_q = [1,1]
            S1OnOff_Off_PSC_s = [0,0]
            S1OnOff_Off_PSC_x = [0,0]
            S1OnOff_Off_PSC_F = [1,1]
            S1OnOff_Off_PSC_P = [1,1]
            S1OnOff_Off_PSC_q = [1,1]
            R2On_R1On_PSC_s = [0,0]
            R2On_R1On_PSC_x = [0,0]
            R2On_R1On_PSC_F = [1,1]
            R2On_R1On_PSC_P = [1,1]
            R2On_R1On_PSC_q = [1,1]
            S2OnOff_R1On_PSC_s = [0,0]
            S2OnOff_R1On_PSC_x = [0,0]
            S2OnOff_R1On_PSC_F = [1,1]
            S2OnOff_R1On_PSC_P = [1,1]
            S2OnOff_R1On_PSC_q = [1,1]
            R2On_S2OnOff_PSC_s = [0,0]
            R2On_S2OnOff_PSC_x = [0,0]
            R2On_S2OnOff_PSC_F = [1,1]
            R2On_S2OnOff_PSC_P = [1,1]
            R2On_S2OnOff_PSC_q = [1,1]
            R2Off_S2OnOff_PSC_s = [0,0]
            R2Off_S2OnOff_PSC_x = [0,0]
            R2Off_S2OnOff_PSC_F = [1,1]
            R2Off_S2OnOff_PSC_P = [1,1]
            R2Off_S2OnOff_PSC_q = [1,1]
            R2Off_R1Off_PSC_s = [0,0]
            R2Off_R1Off_PSC_x = [0,0]
            R2Off_R1Off_PSC_F = [1,1]
            R2Off_R1Off_PSC_P = [1,1]
            R2Off_R1Off_PSC_q = [1,1]
            S2OnOff_R1Off_PSC_s = [0,0]
            S2OnOff_R1Off_PSC_x = [0,0]
            S2OnOff_R1Off_PSC_F = [1,1]
            S2OnOff_R1Off_PSC_P = [1,1]
            S2OnOff_R1Off_PSC_q = [1,1]
            R2On_R2On_iNoise_V3_sn = [0, 0]
            R2On_R2On_iNoise_V3_xn = [0, 0]


       
            
            #Delcare Inputs
            self.On_On_IC_input = genPoissonInputs.gen_poisson_inputs(num_trials_count,self.On_On_IC_locNum,self.On_On_IC_label,self.On_On_IC_t_ref,self.On_On_IC_t_ref_rel,self.On_On_IC_rec)
            self.Off_Off_IC_input = genPoissonInputs.gen_poisson_inputs(num_trials_count,self.Off_Off_IC_locNum,self.Off_Off_IC_label,self.Off_Off_IC_t_ref,self.Off_Off_IC_t_ref_rel,self.Off_Off_IC_rec)

            for t in range(1,T):
                #print('hello2')

                On_V_k1 = ( (self.On_E_L-On_V[-1]) - self.On_R*On_g_ad[-1]*(On_V[-1]-self.On_E_k) - self.On_R*((((self.On_On_IC_g_postIC*(self.On_On_IC_input[t]*self.On_On_IC_netcon)*(On_V[-1]-self.On_On_IC_E_exc))))) + self.On_R*self.On_Itonic*self.On_Imask  ) / self.On_tau
                On_g_ad_k1 = -On_g_ad[-1] / self.On_tau_ad
                Off_V_k1 = ( (self.Off_E_L-Off_V[-1]) - self.Off_R*Off_g_ad[-1]*(Off_V[-1]-self.Off_E_k) - self.Off_R*((((self.Off_Off_IC_g_postIC*(self.Off_Off_IC_input[t]*self.Off_Off_IC_netcon)*(Off_V[-1]-self.Off_Off_IC_E_exc))))) + self.Off_R*self.Off_Itonic*self.Off_Imask  ) / self.Off_tau
                Off_g_ad_k1 = -Off_g_ad[-1] / self.Off_tau_ad
                R1On_V_k1 = ( (self.R1On_E_L-R1On_V[-1]) - self.R1On_R*R1On_g_ad[-1]*(R1On_V[-1]-self.R1On_E_k) - self.R1On_R*((((self.R1On_On_PSC_gSYN*(R1On_On_PSC_s[-1]*self.R1On_On_PSC_netcon)*(R1On_V[-1]-self.R1On_On_PSC_ESYN))))+((((self.R1On_S1OnOff_PSC_gSYN*(R1On_S1OnOff_PSC_s[-1]*self.R1On_S1OnOff_PSC_netcon)*(R1On_V[-1]-self.R1On_S1OnOff_PSC_ESYN)))))) + self.R1On_R*self.R1On_Itonic*self.R1On_Imask  ) / self.R1On_tau
                R1On_g_ad_k1 = -R1On_g_ad[-1] / self.R1On_tau_ad
                R1Off_V_k1 = ( (self.R1Off_E_L-R1Off_V[-1]) - self.R1Off_R*R1Off_g_ad[-1]*(R1Off_V[-1]-self.R1Off_E_k) - self.R1Off_R*((((self.R1Off_S1OnOff_PSC_gSYN*(R1Off_S1OnOff_PSC_s[-1]*self.R1Off_S1OnOff_PSC_netcon)*(R1Off_V[-1]-self.R1Off_S1OnOff_PSC_ESYN))))+((((self.R1Off_Off_PSC_gSYN*(R1Off_Off_PSC_s[-1]*self.R1Off_Off_PSC_netcon)*(R1Off_V[-1]-self.R1Off_Off_PSC_ESYN)))))) + self.R1Off_R*self.R1Off_Itonic*self.R1Off_Imask  ) / self.R1Off_tau
                R1Off_g_ad_k1 = -R1Off_g_ad[-1] / self.R1Off_tau_ad
                S1OnOff_V_k1 = ( (self.S1OnOff_E_L-S1OnOff_V[-1]) - self.S1OnOff_R*S1OnOff_g_ad[-1]*(S1OnOff_V[-1]-self.S1OnOff_E_k) - self.S1OnOff_R*((((self.S1OnOff_On_PSC_gSYN*(S1OnOff_On_PSC_s[-1]*self.S1OnOff_On_PSC_netcon)*(S1OnOff_V[-1]-self.S1OnOff_On_PSC_ESYN))))+((((self.S1OnOff_Off_PSC_gSYN*(S1OnOff_Off_PSC_s[-1]*self.S1OnOff_Off_PSC_netcon)*(S1OnOff_V[-1]-self.S1OnOff_Off_PSC_ESYN)))))) + self.S1OnOff_R*self.S1OnOff_Itonic*self.S1OnOff_Imask  ) / self.S1OnOff_tau
                S1OnOff_g_ad_k1 = -S1OnOff_g_ad[-1] / self.S1OnOff_tau_ad
                R2On_V_k1 = ( (self.R2On_E_L-R2On_V[-1]) - self.R2On_R*R2On_g_ad[-1]*(R2On_V[-1]-self.R2On_E_k) - self.R2On_R*((((self.R2On_R1On_PSC_gSYN*(R2On_R1On_PSC_s[-1]*self.R2On_R1On_PSC_netcon)*(R2On_V[-1]-self.R2On_R1On_PSC_ESYN))))+((((self.R2On_S2OnOff_PSC_gSYN*(R2On_S2OnOff_PSC_s[-1]*self.R2On_S2OnOff_PSC_netcon)*(R2On_V[-1]-self.R2On_S2OnOff_PSC_ESYN))))+((((self.R2On_R2On_iNoise_V3_nSYN*(R2On_R2On_iNoise_V3_sn[-1]*self.R2On_R2On_iNoise_V3_netcon)*(R2On_V[-1]-self.R2On_R2On_iNoise_V3_E_exc))))))) + self.R2On_R*self.R2On_Itonic*self.R2On_Imask  ) / self.R2On_tau
                R2On_g_ad_k1 = -R2On_g_ad[-1] / self.R2On_tau_ad
                R2Off_V_k1 = ( (self.R2Off_E_L-R2Off_V[-1]) - self.R2Off_R*R2Off_g_ad[-1]*(R2Off_V[-1]-self.R2Off_E_k) - self.R2Off_R*((((self.R2Off_S2OnOff_PSC_gSYN*(R2Off_S2OnOff_PSC_s[-1]*self.R2Off_S2OnOff_PSC_netcon)*(R2Off_V[-1]-self.R2Off_S2OnOff_PSC_ESYN))))+((((self.R2Off_R1Off_PSC_gSYN*(R2Off_R1Off_PSC_s[-1]*self.R2Off_R1Off_PSC_netcon)*(R2Off_V[-1]-self.R2Off_R1Off_PSC_ESYN)))))) + self.R2Off_R*self.R2Off_Itonic*self.R2Off_Imask  ) / self.R2Off_tau
                R2Off_g_ad_k1 = -R2Off_g_ad[-1] / self.R2Off_tau_ad
                S2OnOff_V_k1 = ( (self.S2OnOff_E_L-S2OnOff_V[-1]) - self.S2OnOff_R*S2OnOff_g_ad[-1]*(S2OnOff_V[-1]-self.S2OnOff_E_k) - self.S2OnOff_R*((((self.S2OnOff_R1On_PSC_gSYN*(S2OnOff_R1On_PSC_s[-1]*self.S2OnOff_R1On_PSC_netcon)*(S2OnOff_V[-1]-self.S2OnOff_R1On_PSC_ESYN))))+((((self.S2OnOff_R1Off_PSC_gSYN*(S2OnOff_R1Off_PSC_s[-1]*self.S2OnOff_R1Off_PSC_netcon)*(S2OnOff_V[-1]-self.S2OnOff_R1Off_PSC_ESYN)))))) + self.S2OnOff_R*self.S2OnOff_Itonic*self.S2OnOff_Imask  ) / self.S2OnOff_tau
                S2OnOff_g_ad_k1 = -S2OnOff_g_ad[-1] / self.S2OnOff_tau_ad
                R1On_On_PSC_s_k1 = ( self.R1On_On_PSC_scale * R1On_On_PSC_x[-1] - R1On_On_PSC_s[-1] )/self.R1On_On_PSC_tauR
                R1On_On_PSC_x_k1 = -R1On_On_PSC_x[-1]/self.R1On_On_PSC_tauD
                R1On_On_PSC_F_k1 = (1 - R1On_On_PSC_F[-1])/self.R1On_On_PSC_tauF
                R1On_On_PSC_P_k1 = (1 - R1On_On_PSC_P[-1])/self.R1On_On_PSC_tauP
                R1On_On_PSC_q_k1 = 0
                S1OnOff_On_PSC_s_k1 = ( self.S1OnOff_On_PSC_scale * S1OnOff_On_PSC_x[-1] - S1OnOff_On_PSC_s[-1] )/self.S1OnOff_On_PSC_tauR
                S1OnOff_On_PSC_x_k1 = -S1OnOff_On_PSC_x[-1]/self.S1OnOff_On_PSC_tauD
                S1OnOff_On_PSC_F_k1 = (1 - S1OnOff_On_PSC_F[-1])/self.S1OnOff_On_PSC_tauF
                S1OnOff_On_PSC_P_k1 = (1 - S1OnOff_On_PSC_P[-1])/self.S1OnOff_On_PSC_tauP
                S1OnOff_On_PSC_q_k1 = 0
                R1On_S1OnOff_PSC_s_k1 = ( self.R1On_S1OnOff_PSC_scale * R1On_S1OnOff_PSC_x[-1] - R1On_S1OnOff_PSC_s[-1] )/self.R1On_S1OnOff_PSC_tauR
                R1On_S1OnOff_PSC_x_k1 = -R1On_S1OnOff_PSC_x[-1]/self.R1On_S1OnOff_PSC_tauD
                R1On_S1OnOff_PSC_F_k1 = (1 - R1On_S1OnOff_PSC_F[-1])/self.R1On_S1OnOff_PSC_tauF
                R1On_S1OnOff_PSC_P_k1 = (1 - R1On_S1OnOff_PSC_P[-1])/self.R1On_S1OnOff_PSC_tauP
                R1On_S1OnOff_PSC_q_k1 = 0
                R1Off_S1OnOff_PSC_s_k1 = ( self.R1Off_S1OnOff_PSC_scale * R1Off_S1OnOff_PSC_x[-1] - R1Off_S1OnOff_PSC_s[-1] )/self.R1Off_S1OnOff_PSC_tauR
                R1Off_S1OnOff_PSC_x_k1 = -R1Off_S1OnOff_PSC_x[-1]/self.R1Off_S1OnOff_PSC_tauD
                R1Off_S1OnOff_PSC_F_k1 = (1 - R1Off_S1OnOff_PSC_F[-1])/self.R1Off_S1OnOff_PSC_tauF
                R1Off_S1OnOff_PSC_P_k1 = (1 - R1Off_S1OnOff_PSC_P[-1])/self.R1Off_S1OnOff_PSC_tauP
                R1Off_S1OnOff_PSC_q_k1 = 0
                R1Off_Off_PSC_s_k1 = ( self.R1Off_Off_PSC_scale * R1Off_Off_PSC_x[-1] - R1Off_Off_PSC_s[-1] )/self.R1Off_Off_PSC_tauR
                R1Off_Off_PSC_x_k1 = -R1Off_Off_PSC_x[-1]/self.R1Off_Off_PSC_tauD
                R1Off_Off_PSC_F_k1 = (1 - R1Off_Off_PSC_F[-1])/self.R1Off_Off_PSC_tauF
                R1Off_Off_PSC_P_k1 = (1 - R1Off_Off_PSC_P[-1])/self.R1Off_Off_PSC_tauP
                R1Off_Off_PSC_q_k1 = 0
                S1OnOff_Off_PSC_s_k1 = ( self.S1OnOff_Off_PSC_scale * S1OnOff_Off_PSC_x[-1] - S1OnOff_Off_PSC_s[-1] )/self.S1OnOff_Off_PSC_tauR
                S1OnOff_Off_PSC_x_k1 = -S1OnOff_Off_PSC_x[-1]/self.S1OnOff_Off_PSC_tauD
                S1OnOff_Off_PSC_F_k1 = (1 - S1OnOff_Off_PSC_F[-1])/self.S1OnOff_Off_PSC_tauF
                S1OnOff_Off_PSC_P_k1 = (1 - S1OnOff_Off_PSC_P[-1])/self.S1OnOff_Off_PSC_tauP
                S1OnOff_Off_PSC_q_k1 = 0
                R2On_R1On_PSC_s_k1 = ( self.R2On_R1On_PSC_scale * R2On_R1On_PSC_x[-1] - R2On_R1On_PSC_s[-1] )/self.R2On_R1On_PSC_tauR
                R2On_R1On_PSC_x_k1 = -R2On_R1On_PSC_x[-1]/self.R2On_R1On_PSC_tauD
                R2On_R1On_PSC_F_k1 = (1 - R2On_R1On_PSC_F[-1])/self.R2On_R1On_PSC_tauF
                R2On_R1On_PSC_P_k1 = (1 - R2On_R1On_PSC_P[-1])/self.R2On_R1On_PSC_tauP
                R2On_R1On_PSC_q_k1 = 0
                S2OnOff_R1On_PSC_s_k1 = ( self.S2OnOff_R1On_PSC_scale * S2OnOff_R1On_PSC_x[-1] - S2OnOff_R1On_PSC_s[-1] )/self.S2OnOff_R1On_PSC_tauR
                S2OnOff_R1On_PSC_x_k1 = -S2OnOff_R1On_PSC_x[-1]/self.S2OnOff_R1On_PSC_tauD
                S2OnOff_R1On_PSC_F_k1 = (1 - S2OnOff_R1On_PSC_F[-1])/self.S2OnOff_R1On_PSC_tauF
                S2OnOff_R1On_PSC_P_k1 = (1 - S2OnOff_R1On_PSC_P[-1])/self.S2OnOff_R1On_PSC_tauP
                S2OnOff_R1On_PSC_q_k1 = 0
                R2On_S2OnOff_PSC_s_k1 = ( self.R2On_S2OnOff_PSC_scale * R2On_S2OnOff_PSC_x[-1] - R2On_S2OnOff_PSC_s[-1] )/self.R2On_S2OnOff_PSC_tauR
                R2On_S2OnOff_PSC_x_k1 = -R2On_S2OnOff_PSC_x[-1]/self.R2On_S2OnOff_PSC_tauD
                R2On_S2OnOff_PSC_F_k1 = (1 - R2On_S2OnOff_PSC_F[-1])/self.R2On_S2OnOff_PSC_tauF
                R2On_S2OnOff_PSC_P_k1 = (1 - R2On_S2OnOff_PSC_P[-1])/self.R2On_S2OnOff_PSC_tauP
                R2On_S2OnOff_PSC_q_k1 = 0
                R2Off_S2OnOff_PSC_s_k1 = ( self.R2Off_S2OnOff_PSC_scale * R2Off_S2OnOff_PSC_x[-1] - R2Off_S2OnOff_PSC_s[-1] )/self.R2Off_S2OnOff_PSC_tauR
                R2Off_S2OnOff_PSC_x_k1 = -R2Off_S2OnOff_PSC_x[-1]/self.R2Off_S2OnOff_PSC_tauD
                R2Off_S2OnOff_PSC_F_k1 = (1 - R2Off_S2OnOff_PSC_F[-1])/self.R2Off_S2OnOff_PSC_tauF
                R2Off_S2OnOff_PSC_P_k1 = (1 - R2Off_S2OnOff_PSC_P[-1])/self.R2Off_S2OnOff_PSC_tauP
                R2Off_S2OnOff_PSC_q_k1 = 0
                R2Off_R1Off_PSC_s_k1 = ( self.R2Off_R1Off_PSC_scale * R2Off_R1Off_PSC_x[-1] - R2Off_R1Off_PSC_s[-1] )/self.R2Off_R1Off_PSC_tauR
                R2Off_R1Off_PSC_x_k1 = -R2Off_R1Off_PSC_x[-1]/self.R2Off_R1Off_PSC_tauD
                R2Off_R1Off_PSC_F_k1 = (1 - R2Off_R1Off_PSC_F[-1])/self.R2Off_R1Off_PSC_tauF
                R2Off_R1Off_PSC_P_k1 = (1 - R2Off_R1Off_PSC_P[-1])/self.R2Off_R1Off_PSC_tauP
                R2Off_R1Off_PSC_q_k1 = 0
                S2OnOff_R1Off_PSC_s_k1 = ( self.S2OnOff_R1Off_PSC_scale * S2OnOff_R1Off_PSC_x[-1] - S2OnOff_R1Off_PSC_s[-1] )/self.S2OnOff_R1Off_PSC_tauR
                S2OnOff_R1Off_PSC_x_k1 = -S2OnOff_R1Off_PSC_x[-1]/self.S2OnOff_R1Off_PSC_tauD
                S2OnOff_R1Off_PSC_F_k1 = (1 - S2OnOff_R1Off_PSC_F[-1])/self.S2OnOff_R1Off_PSC_tauF
                S2OnOff_R1Off_PSC_P_k1 = (1 - S2OnOff_R1Off_PSC_P[-1])/self.S2OnOff_R1Off_PSC_tauP
                S2OnOff_R1Off_PSC_q_k1 = 0
                R2On_R2On_iNoise_V3_sn_k1 = ( self.R2On_R2On_iNoise_V3_scale * R2On_R2On_iNoise_V3_xn[-1] - R2On_R2On_iNoise_V3_sn[-1] )/self.R2On_R2On_iNoise_V3_tauR_N
                R2On_R2On_iNoise_V3_xn_k1 = -R2On_R2On_iNoise_V3_xn[-1]/self.R2On_R2On_iNoise_V3_tauD_N + self.R2On_R2On_iNoise_V3_token[t]/self.R2On_R2On_iNoise_V3_dt


                #Update Eulers
                On_V[-2] = On_V[-1]
                On_V[-1] = (On_V[-1]+self.dt*On_V_k1).view(())
                On_g_ad[-2] = On_g_ad[-1]
                On_g_ad[-1] = (On_g_ad[-1]+self.dt*On_g_ad_k1).view(())
                Off_V[-2] = Off_V[-1]
                Off_V[-1] = (Off_V[-1]+self.dt*Off_V_k1).view(())
                Off_g_ad[-2] = Off_g_ad[-1]
                Off_g_ad[-1] = (Off_g_ad[-1]+self.dt*Off_g_ad_k1).view(())
                R1On_V[-2] = R1On_V[-1]
                R1On_V[-1] = (R1On_V[-1]+self.dt*R1On_V_k1).view(())
                R1On_g_ad[-2] = R1On_g_ad[-1]
                R1On_g_ad[-1] = (R1On_g_ad[-1]+self.dt*R1On_g_ad_k1).view(())
                R1Off_V[-2] = R1Off_V[-1]
                R1Off_V[-1] = (R1Off_V[-1]+self.dt*R1Off_V_k1).view(())
                R1Off_g_ad[-2] = R1Off_g_ad[-1]
                R1Off_g_ad[-1] = (R1Off_g_ad[-1]+self.dt*R1Off_g_ad_k1).view(())
                S1OnOff_V[-2] = S1OnOff_V[-1]
                S1OnOff_V[-1] = (S1OnOff_V[-1]+self.dt*S1OnOff_V_k1).view(())
                S1OnOff_g_ad[-2] = S1OnOff_g_ad[-1]
                S1OnOff_g_ad[-1] = (S1OnOff_g_ad[-1]+self.dt*S1OnOff_g_ad_k1).view(())
                R2On_V[-2] = R2On_V[-1]
                R2On_V[-1] = (R2On_V[-1]+self.dt*R2On_V_k1).view(())
                R2On_g_ad[-2] = R2On_g_ad[-1]
                R2On_g_ad[-1] = (R2On_g_ad[-1]+self.dt*R2On_g_ad_k1).view(())
                R2Off_V[-2] = R2Off_V[-1]
                R2Off_V[-1] = (R2Off_V[-1]+self.dt*R2Off_V_k1).view(())
                R2Off_g_ad[-2] = R2Off_g_ad[-1]
                R2Off_g_ad[-1] = (R2Off_g_ad[-1]+self.dt*R2Off_g_ad_k1).view(())
                S2OnOff_V[-2] = S2OnOff_V[-1]
                S2OnOff_V[-1] = (S2OnOff_V[-1]+self.dt*S2OnOff_V_k1).view(())
                S2OnOff_g_ad[-2] = S2OnOff_g_ad[-1]
                S2OnOff_g_ad[-1] = (S2OnOff_g_ad[-1]+self.dt*S2OnOff_g_ad_k1).view(())
                R1On_On_PSC_s[-2] = R1On_On_PSC_s[-1]
                R1On_On_PSC_s[-1] = (R1On_On_PSC_s[-1]+self.dt*R1On_On_PSC_s_k1).view(())
                R1On_On_PSC_x[-2] = R1On_On_PSC_x[-1]
                R1On_On_PSC_x[-1] = (R1On_On_PSC_x[-1]+self.dt*R1On_On_PSC_x_k1).view(())
                R1On_On_PSC_F[-2] = R1On_On_PSC_F[-1]
                R1On_On_PSC_F[-1] = (R1On_On_PSC_F[-1]+self.dt*R1On_On_PSC_F_k1).view(())
                R1On_On_PSC_P[-2] = R1On_On_PSC_P[-1]
                R1On_On_PSC_P[-1] = (R1On_On_PSC_P[-1]+self.dt*R1On_On_PSC_P_k1).view(())
                R1On_On_PSC_q[-2] = R1On_On_PSC_q[-1]
                R1On_On_PSC_q[-1] = (R1On_On_PSC_q[-1]+self.dt*R1On_On_PSC_q_k1).view(())
                S1OnOff_On_PSC_s[-2] = S1OnOff_On_PSC_s[-1]
                S1OnOff_On_PSC_s[-1] = (S1OnOff_On_PSC_s[-1]+self.dt*S1OnOff_On_PSC_s_k1).view(())
                S1OnOff_On_PSC_x[-2] = S1OnOff_On_PSC_x[-1]
                S1OnOff_On_PSC_x[-1] = (S1OnOff_On_PSC_x[-1]+self.dt*S1OnOff_On_PSC_x_k1).view(())
                S1OnOff_On_PSC_F[-2] = S1OnOff_On_PSC_F[-1]
                S1OnOff_On_PSC_F[-1] = (S1OnOff_On_PSC_F[-1]+self.dt*S1OnOff_On_PSC_F_k1).view(())
                S1OnOff_On_PSC_P[-2] = S1OnOff_On_PSC_P[-1]
                S1OnOff_On_PSC_P[-1] = (S1OnOff_On_PSC_P[-1]+self.dt*S1OnOff_On_PSC_P_k1).view(())
                S1OnOff_On_PSC_q[-2] = S1OnOff_On_PSC_q[-1]
                S1OnOff_On_PSC_q[-1] = (S1OnOff_On_PSC_q[-1]+self.dt*S1OnOff_On_PSC_q_k1).view(())
                R1On_S1OnOff_PSC_s[-2] = R1On_S1OnOff_PSC_s[-1]
                R1On_S1OnOff_PSC_s[-1] = (R1On_S1OnOff_PSC_s[-1]+self.dt*R1On_S1OnOff_PSC_s_k1).view(())
                R1On_S1OnOff_PSC_x[-2] = R1On_S1OnOff_PSC_x[-1]
                R1On_S1OnOff_PSC_x[-1] = (R1On_S1OnOff_PSC_x[-1]+self.dt*R1On_S1OnOff_PSC_x_k1).view(())
                R1On_S1OnOff_PSC_F[-2] = R1On_S1OnOff_PSC_F[-1]
                R1On_S1OnOff_PSC_F[-1] = (R1On_S1OnOff_PSC_F[-1]+self.dt*R1On_S1OnOff_PSC_F_k1).view(())
                R1On_S1OnOff_PSC_P[-2] = R1On_S1OnOff_PSC_P[-1]
                R1On_S1OnOff_PSC_P[-1] = (R1On_S1OnOff_PSC_P[-1]+self.dt*R1On_S1OnOff_PSC_P_k1).view(())
                R1On_S1OnOff_PSC_q[-2] = R1On_S1OnOff_PSC_q[-1]
                R1On_S1OnOff_PSC_q[-1] = (R1On_S1OnOff_PSC_q[-1]+self.dt*R1On_S1OnOff_PSC_q_k1).view(())
                R1Off_S1OnOff_PSC_s[-2] = R1Off_S1OnOff_PSC_s[-1]
                R1Off_S1OnOff_PSC_s[-1] = (R1Off_S1OnOff_PSC_s[-1]+self.dt*R1Off_S1OnOff_PSC_s_k1).view(())
                R1Off_S1OnOff_PSC_x[-2] = R1Off_S1OnOff_PSC_x[-1]
                R1Off_S1OnOff_PSC_x[-1] = (R1Off_S1OnOff_PSC_x[-1]+self.dt*R1Off_S1OnOff_PSC_x_k1).view(())
                R1Off_S1OnOff_PSC_F[-2] = R1Off_S1OnOff_PSC_F[-1]
                R1Off_S1OnOff_PSC_F[-1] = (R1Off_S1OnOff_PSC_F[-1]+self.dt*R1Off_S1OnOff_PSC_F_k1).view(())
                R1Off_S1OnOff_PSC_P[-2] = R1Off_S1OnOff_PSC_P[-1]
                R1Off_S1OnOff_PSC_P[-1] = (R1Off_S1OnOff_PSC_P[-1]+self.dt*R1Off_S1OnOff_PSC_P_k1).view(())
                R1Off_S1OnOff_PSC_q[-2] = R1Off_S1OnOff_PSC_q[-1]
                R1Off_S1OnOff_PSC_q[-1] = (R1Off_S1OnOff_PSC_q[-1]+self.dt*R1Off_S1OnOff_PSC_q_k1).view(())
                R1Off_Off_PSC_s[-2] = R1Off_Off_PSC_s[-1]
                R1Off_Off_PSC_s[-1] = (R1Off_Off_PSC_s[-1]+self.dt*R1Off_Off_PSC_s_k1).view(())
                R1Off_Off_PSC_x[-2] = R1Off_Off_PSC_x[-1]
                R1Off_Off_PSC_x[-1] = (R1Off_Off_PSC_x[-1]+self.dt*R1Off_Off_PSC_x_k1).view(())
                R1Off_Off_PSC_F[-2] = R1Off_Off_PSC_F[-1]
                R1Off_Off_PSC_F[-1] = (R1Off_Off_PSC_F[-1]+self.dt*R1Off_Off_PSC_F_k1).view(())
                R1Off_Off_PSC_P[-2] = R1Off_Off_PSC_P[-1]
                R1Off_Off_PSC_P[-1] = (R1Off_Off_PSC_P[-1]+self.dt*R1Off_Off_PSC_P_k1).view(())
                R1Off_Off_PSC_q[-2] = R1Off_Off_PSC_q[-1]
                R1Off_Off_PSC_q[-1] = (R1Off_Off_PSC_q[-1]+self.dt*R1Off_Off_PSC_q_k1).view(())
                S1OnOff_Off_PSC_s[-2] = S1OnOff_Off_PSC_s[-1]
                S1OnOff_Off_PSC_s[-1] = (S1OnOff_Off_PSC_s[-1]+self.dt*S1OnOff_Off_PSC_s_k1).view(())
                S1OnOff_Off_PSC_x[-2] = S1OnOff_Off_PSC_x[-1]
                S1OnOff_Off_PSC_x[-1] = (S1OnOff_Off_PSC_x[-1]+self.dt*S1OnOff_Off_PSC_x_k1).view(())
                S1OnOff_Off_PSC_F[-2] = S1OnOff_Off_PSC_F[-1]
                S1OnOff_Off_PSC_F[-1] = (S1OnOff_Off_PSC_F[-1]+self.dt*S1OnOff_Off_PSC_F_k1).view(())
                S1OnOff_Off_PSC_P[-2] = S1OnOff_Off_PSC_P[-1]
                S1OnOff_Off_PSC_P[-1] = (S1OnOff_Off_PSC_P[-1]+self.dt*S1OnOff_Off_PSC_P_k1).view(())
                S1OnOff_Off_PSC_q[-2] = S1OnOff_Off_PSC_q[-1]
                S1OnOff_Off_PSC_q[-1] = (S1OnOff_Off_PSC_q[-1]+self.dt*S1OnOff_Off_PSC_q_k1).view(())
                R2On_R1On_PSC_s[-2] = R2On_R1On_PSC_s[-1]
                R2On_R1On_PSC_s[-1] = (R2On_R1On_PSC_s[-1]+self.dt*R2On_R1On_PSC_s_k1).view(())
                R2On_R1On_PSC_x[-2] = R2On_R1On_PSC_x[-1]
                R2On_R1On_PSC_x[-1] = (R2On_R1On_PSC_x[-1]+self.dt*R2On_R1On_PSC_x_k1).view(())
                R2On_R1On_PSC_F[-2] = R2On_R1On_PSC_F[-1]
                R2On_R1On_PSC_F[-1] = (R2On_R1On_PSC_F[-1]+self.dt*R2On_R1On_PSC_F_k1).view(())
                R2On_R1On_PSC_P[-2] = R2On_R1On_PSC_P[-1]
                R2On_R1On_PSC_P[-1] = (R2On_R1On_PSC_P[-1]+self.dt*R2On_R1On_PSC_P_k1).view(())
                R2On_R1On_PSC_q[-2] = R2On_R1On_PSC_q[-1]
                R2On_R1On_PSC_q[-1] = (R2On_R1On_PSC_q[-1]+self.dt*R2On_R1On_PSC_q_k1).view(())
                S2OnOff_R1On_PSC_s[-2] = S2OnOff_R1On_PSC_s[-1]
                S2OnOff_R1On_PSC_s[-1] = (S2OnOff_R1On_PSC_s[-1]+self.dt*S2OnOff_R1On_PSC_s_k1).view(())
                S2OnOff_R1On_PSC_x[-2] = S2OnOff_R1On_PSC_x[-1]
                S2OnOff_R1On_PSC_x[-1] = (S2OnOff_R1On_PSC_x[-1]+self.dt*S2OnOff_R1On_PSC_x_k1).view(())
                S2OnOff_R1On_PSC_F[-2] = S2OnOff_R1On_PSC_F[-1]
                S2OnOff_R1On_PSC_F[-1] = (S2OnOff_R1On_PSC_F[-1]+self.dt*S2OnOff_R1On_PSC_F_k1).view(())
                S2OnOff_R1On_PSC_P[-2] = S2OnOff_R1On_PSC_P[-1]
                S2OnOff_R1On_PSC_P[-1] = (S2OnOff_R1On_PSC_P[-1]+self.dt*S2OnOff_R1On_PSC_P_k1).view(())
                S2OnOff_R1On_PSC_q[-2] = S2OnOff_R1On_PSC_q[-1]
                S2OnOff_R1On_PSC_q[-1] = (S2OnOff_R1On_PSC_q[-1]+self.dt*S2OnOff_R1On_PSC_q_k1).view(())
                R2On_S2OnOff_PSC_s[-2] = R2On_S2OnOff_PSC_s[-1]
                R2On_S2OnOff_PSC_s[-1] = (R2On_S2OnOff_PSC_s[-1]+self.dt*R2On_S2OnOff_PSC_s_k1).view(())
                R2On_S2OnOff_PSC_x[-2] = R2On_S2OnOff_PSC_x[-1]
                R2On_S2OnOff_PSC_x[-1] = (R2On_S2OnOff_PSC_x[-1]+self.dt*R2On_S2OnOff_PSC_x_k1).view(())
                R2On_S2OnOff_PSC_F[-2] = R2On_S2OnOff_PSC_F[-1]
                R2On_S2OnOff_PSC_F[-1] = (R2On_S2OnOff_PSC_F[-1]+self.dt*R2On_S2OnOff_PSC_F_k1).view(())
                R2On_S2OnOff_PSC_P[-2] = R2On_S2OnOff_PSC_P[-1]
                R2On_S2OnOff_PSC_P[-1] = (R2On_S2OnOff_PSC_P[-1]+self.dt*R2On_S2OnOff_PSC_P_k1).view(())
                R2On_S2OnOff_PSC_q[-2] = R2On_S2OnOff_PSC_q[-1]
                R2On_S2OnOff_PSC_q[-1] = (R2On_S2OnOff_PSC_q[-1]+self.dt*R2On_S2OnOff_PSC_q_k1).view(())
                R2Off_S2OnOff_PSC_s[-2] = R2Off_S2OnOff_PSC_s[-1]
                R2Off_S2OnOff_PSC_s[-1] = (R2Off_S2OnOff_PSC_s[-1]+self.dt*R2Off_S2OnOff_PSC_s_k1).view(())
                R2Off_S2OnOff_PSC_x[-2] = R2Off_S2OnOff_PSC_x[-1]
                R2Off_S2OnOff_PSC_x[-1] = (R2Off_S2OnOff_PSC_x[-1]+self.dt*R2Off_S2OnOff_PSC_x_k1).view(())
                R2Off_S2OnOff_PSC_F[-2] = R2Off_S2OnOff_PSC_F[-1]
                R2Off_S2OnOff_PSC_F[-1] = (R2Off_S2OnOff_PSC_F[-1]+self.dt*R2Off_S2OnOff_PSC_F_k1).view(())
                R2Off_S2OnOff_PSC_P[-2] = R2Off_S2OnOff_PSC_P[-1]
                R2Off_S2OnOff_PSC_P[-1] = (R2Off_S2OnOff_PSC_P[-1]+self.dt*R2Off_S2OnOff_PSC_P_k1).view(())
                R2Off_S2OnOff_PSC_q[-2] = R2Off_S2OnOff_PSC_q[-1]
                R2Off_S2OnOff_PSC_q[-1] = (R2Off_S2OnOff_PSC_q[-1]+self.dt*R2Off_S2OnOff_PSC_q_k1).view(())
                R2Off_R1Off_PSC_s[-2] = R2Off_R1Off_PSC_s[-1]
                R2Off_R1Off_PSC_s[-1] = (R2Off_R1Off_PSC_s[-1]+self.dt*R2Off_R1Off_PSC_s_k1).view(())
                R2Off_R1Off_PSC_x[-2] = R2Off_R1Off_PSC_x[-1]
                R2Off_R1Off_PSC_x[-1] = (R2Off_R1Off_PSC_x[-1]+self.dt*R2Off_R1Off_PSC_x_k1).view(())
                R2Off_R1Off_PSC_F[-2] = R2Off_R1Off_PSC_F[-1]
                R2Off_R1Off_PSC_F[-1] = (R2Off_R1Off_PSC_F[-1]+self.dt*R2Off_R1Off_PSC_F_k1).view(())
                R2Off_R1Off_PSC_P[-2] = R2Off_R1Off_PSC_P[-1]
                R2Off_R1Off_PSC_P[-1] = (R2Off_R1Off_PSC_P[-1]+self.dt*R2Off_R1Off_PSC_P_k1).view(())
                R2Off_R1Off_PSC_q[-2] = R2Off_R1Off_PSC_q[-1]
                R2Off_R1Off_PSC_q[-1] = (R2Off_R1Off_PSC_q[-1]+self.dt*R2Off_R1Off_PSC_q_k1).view(())
                S2OnOff_R1Off_PSC_s[-2] = S2OnOff_R1Off_PSC_s[-1]
                S2OnOff_R1Off_PSC_s[-1] = (S2OnOff_R1Off_PSC_s[-1]+self.dt*S2OnOff_R1Off_PSC_s_k1).view(())
                S2OnOff_R1Off_PSC_x[-2] = S2OnOff_R1Off_PSC_x[-1]
                S2OnOff_R1Off_PSC_x[-1] = (S2OnOff_R1Off_PSC_x[-1]+self.dt*S2OnOff_R1Off_PSC_x_k1).view(())
                S2OnOff_R1Off_PSC_F[-2] = S2OnOff_R1Off_PSC_F[-1]
                S2OnOff_R1Off_PSC_F[-1] = (S2OnOff_R1Off_PSC_F[-1]+self.dt*S2OnOff_R1Off_PSC_F_k1).view(())
                S2OnOff_R1Off_PSC_P[-2] = S2OnOff_R1Off_PSC_P[-1]
                S2OnOff_R1Off_PSC_P[-1] = (S2OnOff_R1Off_PSC_P[-1]+self.dt*S2OnOff_R1Off_PSC_P_k1).view(())
                S2OnOff_R1Off_PSC_q[-2] = S2OnOff_R1Off_PSC_q[-1]
                S2OnOff_R1Off_PSC_q[-1] = (S2OnOff_R1Off_PSC_q[-1]+self.dt*S2OnOff_R1Off_PSC_q_k1).view(())
                R2On_R2On_iNoise_V3_sn[-2] = R2On_R2On_iNoise_V3_sn[-1]
                R2On_R2On_iNoise_V3_sn[-1] = (R2On_R2On_iNoise_V3_sn[-1]+self.dt*R2On_R2On_iNoise_V3_sn_k1).view(())
                R2On_R2On_iNoise_V3_xn[-2] = R2On_R2On_iNoise_V3_xn[-1]
                R2On_R2On_iNoise_V3_xn[-1] = (R2On_R2On_iNoise_V3_xn[-1]+self.dt*R2On_R2On_iNoise_V3_xn_k1).view(())


                #Spiking and conditional actions

                On_V_spikes_holder.append(SurrogateSpike.apply(On_V[-1], On_V[-2], self.On_V_thresh))
                if On_V_spikes_holder[-1]:
                    On_tspike[int(On_buffer_index)-1] = helper[t]
                    On_buffer_index = (On_buffer_index % 5) + 1
                Off_V_spikes_holder.append(SurrogateSpike.apply(Off_V[-1], Off_V[-2], self.Off_V_thresh))
                if Off_V_spikes_holder[-1]:
                    Off_tspike[int(Off_buffer_index)-1] = helper[t]
                    Off_buffer_index = (Off_buffer_index % 5) + 1
                R1On_V_spikes_holder.append(SurrogateSpike.apply(R1On_V[-1], R1On_V[-2], self.R1On_V_thresh))
                if R1On_V_spikes_holder[-1]:
                    R1On_tspike[int(R1On_buffer_index)-1] = helper[t]
                    R1On_buffer_index = (R1On_buffer_index % 5) + 1
                R1Off_V_spikes_holder.append(SurrogateSpike.apply(R1Off_V[-1], R1Off_V[-2], self.R1Off_V_thresh))
                if R1Off_V_spikes_holder[-1]:
                    R1Off_tspike[int(R1Off_buffer_index)-1] = helper[t]
                    R1Off_buffer_index = (R1Off_buffer_index % 5) + 1
                S1OnOff_V_spikes_holder.append(SurrogateSpike.apply(S1OnOff_V[-1], S1OnOff_V[-2], self.S1OnOff_V_thresh))
                if S1OnOff_V_spikes_holder[-1]:
                    S1OnOff_tspike[int(S1OnOff_buffer_index)-1] = helper[t]
                    S1OnOff_buffer_index = (S1OnOff_buffer_index % 5) + 1
                R2On_V_spikes_holder.append(SurrogateSpike.apply(R2On_V[-1], R2On_V[-2], self.R2On_V_thresh))
                if R2On_V_spikes_holder[-1]:
                    R2On_tspike[int(R2On_buffer_index)-1] = helper[t]
                    R2On_buffer_index = (R2On_buffer_index % 5) + 1
                R2Off_V_spikes_holder.append(SurrogateSpike.apply(R2Off_V[-1], R2Off_V[-2], self.R2Off_V_thresh))
                if R2Off_V_spikes_holder[-1]:
                    R2Off_tspike[int(R2Off_buffer_index)-1] = helper[t]
                    R2Off_buffer_index = (R2Off_buffer_index % 5) + 1
                S2OnOff_V_spikes_holder.append(SurrogateSpike.apply(S2OnOff_V[-1], S2OnOff_V[-2], self.S2OnOff_V_thresh))
                if S2OnOff_V_spikes_holder[-1]:
                    S2OnOff_tspike[int(S2OnOff_buffer_index)-1] = helper[t]
                    S2OnOff_buffer_index = (S2OnOff_buffer_index % 5) + 1


                On_V_test2a = On_V[-1] > self.On_V_thresh
                if On_V_test2a:
                    On_V[-2] = On_V[-1] 
                    On_V[-1] = self.On_V_reset 
                    On_g_ad[-2] = On_g_ad[-1]
                    On_g_ad[-1] = On_g_ad[-1] + self.On_g_inc
                On_V_test2b = torch.any(helper[t] <= On_tspike + self.On_t_ref)
                if On_V_test2b:
                    On_V[-2] = On_V[-1]
                    On_V[-1] = self.On_V_reset
                Off_V_test2a = Off_V[-1] > self.Off_V_thresh
                if Off_V_test2a:
                    Off_V[-2] = Off_V[-1] 
                    Off_V[-1] = self.Off_V_reset 
                    Off_g_ad[-2] = Off_g_ad[-1]
                    Off_g_ad[-1] = Off_g_ad[-1] + self.Off_g_inc
                Off_V_test2b = torch.any(helper[t] <= Off_tspike + self.Off_t_ref)
                if Off_V_test2b:
                    Off_V[-2] = Off_V[-1]
                    Off_V[-1] = self.Off_V_reset
                R1On_V_test2a = R1On_V[-1] > self.R1On_V_thresh
                if R1On_V_test2a:
                    R1On_V[-2] = R1On_V[-1] 
                    R1On_V[-1] = self.R1On_V_reset 
                    R1On_g_ad[-2] = R1On_g_ad[-1]
                    R1On_g_ad[-1] = R1On_g_ad[-1] + self.R1On_g_inc
                R1On_V_test2b = torch.any(helper[t] <= R1On_tspike + self.R1On_t_ref)
                if R1On_V_test2b:
                    R1On_V[-2] = R1On_V[-1]
                    R1On_V[-1] = self.R1On_V_reset
                R1Off_V_test2a = R1Off_V[-1] > self.R1Off_V_thresh
                if R1Off_V_test2a:
                    R1Off_V[-2] = R1Off_V[-1] 
                    R1Off_V[-1] = self.R1Off_V_reset 
                    R1Off_g_ad[-2] = R1Off_g_ad[-1]
                    R1Off_g_ad[-1] = R1Off_g_ad[-1] + self.R1Off_g_inc
                R1Off_V_test2b = torch.any(helper[t] <= R1Off_tspike + self.R1Off_t_ref)
                if R1Off_V_test2b:
                    R1Off_V[-2] = R1Off_V[-1]
                    R1Off_V[-1] = self.R1Off_V_reset
                S1OnOff_V_test2a = S1OnOff_V[-1] > self.S1OnOff_V_thresh
                if S1OnOff_V_test2a:
                    S1OnOff_V[-2] = S1OnOff_V[-1] 
                    S1OnOff_V[-1] = self.S1OnOff_V_reset 
                    S1OnOff_g_ad[-2] = S1OnOff_g_ad[-1]
                    S1OnOff_g_ad[-1] = S1OnOff_g_ad[-1] + self.S1OnOff_g_inc
                S1OnOff_V_test2b = torch.any(helper[t] <= S1OnOff_tspike + self.S1OnOff_t_ref)
                if S1OnOff_V_test2b:
                    S1OnOff_V[-2] = S1OnOff_V[-1]
                    S1OnOff_V[-1] = self.S1OnOff_V_reset
                R2On_V_test2a = R2On_V[-1] > self.R2On_V_thresh
                if R2On_V_test2a:
                    R2On_V[-2] = R2On_V[-1] 
                    R2On_V[-1] = self.R2On_V_reset 
                    R2On_g_ad[-2] = R2On_g_ad[-1]
                    R2On_g_ad[-1] = R2On_g_ad[-1] + self.R2On_g_inc
                R2On_V_test2b = torch.any(helper[t] <= R2On_tspike + self.R2On_t_ref)
                if R2On_V_test2b:
                    R2On_V[-2] = R2On_V[-1]
                    R2On_V[-1] = self.R2On_V_reset
                R2Off_V_test2a = R2Off_V[-1] > self.R2Off_V_thresh
                if R2Off_V_test2a:
                    R2Off_V[-2] = R2Off_V[-1] 
                    R2Off_V[-1] = self.R2Off_V_reset 
                    R2Off_g_ad[-2] = R2Off_g_ad[-1]
                    R2Off_g_ad[-1] = R2Off_g_ad[-1] + self.R2Off_g_inc
                R2Off_V_test2b = torch.any(helper[t] <= R2Off_tspike + self.R2Off_t_ref)
                if R2Off_V_test2b:
                    R2Off_V[-2] = R2Off_V[-1]
                    R2Off_V[-1] = self.R2Off_V_reset
                S2OnOff_V_test2a = S2OnOff_V[-1] > self.S2OnOff_V_thresh
                if S2OnOff_V_test2a:
                    S2OnOff_V[-2] = S2OnOff_V[-1] 
                    S2OnOff_V[-1] = self.S2OnOff_V_reset 
                    S2OnOff_g_ad[-2] = S2OnOff_g_ad[-1]
                    S2OnOff_g_ad[-1] = S2OnOff_g_ad[-1] + self.S2OnOff_g_inc
                S2OnOff_V_test2b = torch.any(helper[t] <= S2OnOff_tspike + self.S2OnOff_t_ref)
                if S2OnOff_V_test2b:
                    S2OnOff_V[-2] = S2OnOff_V[-1]
                    S2OnOff_V[-1] = self.S2OnOff_V_reset


                S2OnOff_V_test3 = torch.any(helper[t] == On_tspike + self.R1On_On_PSC_delay)
                if S2OnOff_V_test3:
                    R1On_On_PSC_x[-2] = R1On_On_PSC_x[-1]
                    R1On_On_PSC_q[-2] = R1On_On_PSC_F[-1]
                    R1On_On_PSC_F[-2] = R1On_On_PSC_F[-1]
                    R1On_On_PSC_P[-2] = R1On_On_PSC_P[-1]
                    R1On_On_PSC_x[-1] = R1On_On_PSC_x[-1] + R1On_On_PSC_q[-1]
                    R1On_On_PSC_q[-1] = R1On_On_PSC_F[-1] * R1On_On_PSC_P[-1]
                    R1On_On_PSC_F[-1] = R1On_On_PSC_F[-1] + self.R1On_On_PSC_fF*(self.R1On_On_PSC_maxF-R1On_On_PSC_F[-1])
                    R1On_On_PSC_P[-1] = R1On_On_PSC_P[-1] * (1 - self.R1On_On_PSC_fP)
                R1On_On_PSC_de_test3 = torch.any(helper[t] == On_tspike + self.S1OnOff_On_PSC_delay)
                if R1On_On_PSC_de_test3:
                    S1OnOff_On_PSC_x[-2] = S1OnOff_On_PSC_x[-1]
                    S1OnOff_On_PSC_q[-2] = S1OnOff_On_PSC_F[-1]
                    S1OnOff_On_PSC_F[-2] = S1OnOff_On_PSC_F[-1]
                    S1OnOff_On_PSC_P[-2] = S1OnOff_On_PSC_P[-1]
                    S1OnOff_On_PSC_x[-1] = S1OnOff_On_PSC_x[-1] + S1OnOff_On_PSC_q[-1]
                    S1OnOff_On_PSC_q[-1] = S1OnOff_On_PSC_F[-1] * S1OnOff_On_PSC_P[-1]
                    S1OnOff_On_PSC_F[-1] = S1OnOff_On_PSC_F[-1] + self.S1OnOff_On_PSC_fF*(self.S1OnOff_On_PSC_maxF-S1OnOff_On_PSC_F[-1])
                    S1OnOff_On_PSC_P[-1] = S1OnOff_On_PSC_P[-1] * (1 - self.S1OnOff_On_PSC_fP)
                S1OnOff_On_PSC_de_test3 = torch.any(helper[t] == S1OnOff_tspike + self.R1On_S1OnOff_PSC_delay)
                if S1OnOff_On_PSC_de_test3:
                    R1On_S1OnOff_PSC_x[-2] = R1On_S1OnOff_PSC_x[-1]
                    R1On_S1OnOff_PSC_q[-2] = R1On_S1OnOff_PSC_F[-1]
                    R1On_S1OnOff_PSC_F[-2] = R1On_S1OnOff_PSC_F[-1]
                    R1On_S1OnOff_PSC_P[-2] = R1On_S1OnOff_PSC_P[-1]
                    R1On_S1OnOff_PSC_x[-1] = R1On_S1OnOff_PSC_x[-1] + R1On_S1OnOff_PSC_q[-1]
                    R1On_S1OnOff_PSC_q[-1] = R1On_S1OnOff_PSC_F[-1] * R1On_S1OnOff_PSC_P[-1]
                    R1On_S1OnOff_PSC_F[-1] = R1On_S1OnOff_PSC_F[-1] + self.R1On_S1OnOff_PSC_fF*(self.R1On_S1OnOff_PSC_maxF-R1On_S1OnOff_PSC_F[-1])
                    R1On_S1OnOff_PSC_P[-1] = R1On_S1OnOff_PSC_P[-1] * (1 - self.R1On_S1OnOff_PSC_fP)
                R1On_S1OnOff_PSC_de_test3 = torch.any(helper[t] == S1OnOff_tspike + self.R1Off_S1OnOff_PSC_delay)
                if R1On_S1OnOff_PSC_de_test3:
                    R1Off_S1OnOff_PSC_x[-2] = R1Off_S1OnOff_PSC_x[-1]
                    R1Off_S1OnOff_PSC_q[-2] = R1Off_S1OnOff_PSC_F[-1]
                    R1Off_S1OnOff_PSC_F[-2] = R1Off_S1OnOff_PSC_F[-1]
                    R1Off_S1OnOff_PSC_P[-2] = R1Off_S1OnOff_PSC_P[-1]
                    R1Off_S1OnOff_PSC_x[-1] = R1Off_S1OnOff_PSC_x[-1] + R1Off_S1OnOff_PSC_q[-1]
                    R1Off_S1OnOff_PSC_q[-1] = R1Off_S1OnOff_PSC_F[-1] * R1Off_S1OnOff_PSC_P[-1]
                    R1Off_S1OnOff_PSC_F[-1] = R1Off_S1OnOff_PSC_F[-1] + self.R1Off_S1OnOff_PSC_fF*(self.R1Off_S1OnOff_PSC_maxF-R1Off_S1OnOff_PSC_F[-1])
                    R1Off_S1OnOff_PSC_P[-1] = R1Off_S1OnOff_PSC_P[-1] * (1 - self.R1Off_S1OnOff_PSC_fP)
                R1Off_S1OnOff_PSC_de_test3 = torch.any(helper[t] == Off_tspike + self.R1Off_Off_PSC_delay)
                if R1Off_S1OnOff_PSC_de_test3:
                    R1Off_Off_PSC_x[-2] = R1Off_Off_PSC_x[-1]
                    R1Off_Off_PSC_q[-2] = R1Off_Off_PSC_F[-1]
                    R1Off_Off_PSC_F[-2] = R1Off_Off_PSC_F[-1]
                    R1Off_Off_PSC_P[-2] = R1Off_Off_PSC_P[-1]
                    R1Off_Off_PSC_x[-1] = R1Off_Off_PSC_x[-1] + R1Off_Off_PSC_q[-1]
                    R1Off_Off_PSC_q[-1] = R1Off_Off_PSC_F[-1] * R1Off_Off_PSC_P[-1]
                    R1Off_Off_PSC_F[-1] = R1Off_Off_PSC_F[-1] + self.R1Off_Off_PSC_fF*(self.R1Off_Off_PSC_maxF-R1Off_Off_PSC_F[-1])
                    R1Off_Off_PSC_P[-1] = R1Off_Off_PSC_P[-1] * (1 - self.R1Off_Off_PSC_fP)
                R1Off_Off_PSC_de_test3 = torch.any(helper[t] == Off_tspike + self.S1OnOff_Off_PSC_delay)
                if R1Off_Off_PSC_de_test3:
                    S1OnOff_Off_PSC_x[-2] = S1OnOff_Off_PSC_x[-1]
                    S1OnOff_Off_PSC_q[-2] = S1OnOff_Off_PSC_F[-1]
                    S1OnOff_Off_PSC_F[-2] = S1OnOff_Off_PSC_F[-1]
                    S1OnOff_Off_PSC_P[-2] = S1OnOff_Off_PSC_P[-1]
                    S1OnOff_Off_PSC_x[-1] = S1OnOff_Off_PSC_x[-1] + S1OnOff_Off_PSC_q[-1]
                    S1OnOff_Off_PSC_q[-1] = S1OnOff_Off_PSC_F[-1] * S1OnOff_Off_PSC_P[-1]
                    S1OnOff_Off_PSC_F[-1] = S1OnOff_Off_PSC_F[-1] + self.S1OnOff_Off_PSC_fF*(self.S1OnOff_Off_PSC_maxF-S1OnOff_Off_PSC_F[-1])
                    S1OnOff_Off_PSC_P[-1] = S1OnOff_Off_PSC_P[-1] * (1 - self.S1OnOff_Off_PSC_fP)
                S1OnOff_Off_PSC_de_test3 = torch.any(helper[t] == R1On_tspike + self.R2On_R1On_PSC_delay)
                if S1OnOff_Off_PSC_de_test3:
                    R2On_R1On_PSC_x[-2] = R2On_R1On_PSC_x[-1]
                    R2On_R1On_PSC_q[-2] = R2On_R1On_PSC_F[-1]
                    R2On_R1On_PSC_F[-2] = R2On_R1On_PSC_F[-1]
                    R2On_R1On_PSC_P[-2] = R2On_R1On_PSC_P[-1]
                    R2On_R1On_PSC_x[-1] = R2On_R1On_PSC_x[-1] + R2On_R1On_PSC_q[-1]
                    R2On_R1On_PSC_q[-1] = R2On_R1On_PSC_F[-1] * R2On_R1On_PSC_P[-1]
                    R2On_R1On_PSC_F[-1] = R2On_R1On_PSC_F[-1] + self.R2On_R1On_PSC_fF*(self.R2On_R1On_PSC_maxF-R2On_R1On_PSC_F[-1])
                    R2On_R1On_PSC_P[-1] = R2On_R1On_PSC_P[-1] * (1 - self.R2On_R1On_PSC_fP)
                R2On_R1On_PSC_de_test3 = torch.any(helper[t] == R1On_tspike + self.S2OnOff_R1On_PSC_delay)
                if R2On_R1On_PSC_de_test3:
                    S2OnOff_R1On_PSC_x[-2] = S2OnOff_R1On_PSC_x[-1]
                    S2OnOff_R1On_PSC_q[-2] = S2OnOff_R1On_PSC_F[-1]
                    S2OnOff_R1On_PSC_F[-2] = S2OnOff_R1On_PSC_F[-1]
                    S2OnOff_R1On_PSC_P[-2] = S2OnOff_R1On_PSC_P[-1]
                    S2OnOff_R1On_PSC_x[-1] = S2OnOff_R1On_PSC_x[-1] + S2OnOff_R1On_PSC_q[-1]
                    S2OnOff_R1On_PSC_q[-1] = S2OnOff_R1On_PSC_F[-1] * S2OnOff_R1On_PSC_P[-1]
                    S2OnOff_R1On_PSC_F[-1] = S2OnOff_R1On_PSC_F[-1] + self.S2OnOff_R1On_PSC_fF*(self.S2OnOff_R1On_PSC_maxF-S2OnOff_R1On_PSC_F[-1])
                    S2OnOff_R1On_PSC_P[-1] = S2OnOff_R1On_PSC_P[-1] * (1 - self.S2OnOff_R1On_PSC_fP)
                S2OnOff_R1On_PSC_de_test3 = torch.any(helper[t] == S2OnOff_tspike + self.R2On_S2OnOff_PSC_delay)
                if S2OnOff_R1On_PSC_de_test3:
                    R2On_S2OnOff_PSC_x[-2] = R2On_S2OnOff_PSC_x[-1]
                    R2On_S2OnOff_PSC_q[-2] = R2On_S2OnOff_PSC_F[-1]
                    R2On_S2OnOff_PSC_F[-2] = R2On_S2OnOff_PSC_F[-1]
                    R2On_S2OnOff_PSC_P[-2] = R2On_S2OnOff_PSC_P[-1]
                    R2On_S2OnOff_PSC_x[-1] = R2On_S2OnOff_PSC_x[-1] + R2On_S2OnOff_PSC_q[-1]
                    R2On_S2OnOff_PSC_q[-1] = R2On_S2OnOff_PSC_F[-1] * R2On_S2OnOff_PSC_P[-1]
                    R2On_S2OnOff_PSC_F[-1] = R2On_S2OnOff_PSC_F[-1] + self.R2On_S2OnOff_PSC_fF*(self.R2On_S2OnOff_PSC_maxF-R2On_S2OnOff_PSC_F[-1])
                    R2On_S2OnOff_PSC_P[-1] = R2On_S2OnOff_PSC_P[-1] * (1 - self.R2On_S2OnOff_PSC_fP)
                R2On_S2OnOff_PSC_de_test3 = torch.any(helper[t] == S2OnOff_tspike + self.R2Off_S2OnOff_PSC_delay)
                if R2On_S2OnOff_PSC_de_test3:
                    R2Off_S2OnOff_PSC_x[-2] = R2Off_S2OnOff_PSC_x[-1]
                    R2Off_S2OnOff_PSC_q[-2] = R2Off_S2OnOff_PSC_F[-1]
                    R2Off_S2OnOff_PSC_F[-2] = R2Off_S2OnOff_PSC_F[-1]
                    R2Off_S2OnOff_PSC_P[-2] = R2Off_S2OnOff_PSC_P[-1]
                    R2Off_S2OnOff_PSC_x[-1] = R2Off_S2OnOff_PSC_x[-1] + R2Off_S2OnOff_PSC_q[-1]
                    R2Off_S2OnOff_PSC_q[-1] = R2Off_S2OnOff_PSC_F[-1] * R2Off_S2OnOff_PSC_P[-1]
                    R2Off_S2OnOff_PSC_F[-1] = R2Off_S2OnOff_PSC_F[-1] + self.R2Off_S2OnOff_PSC_fF*(self.R2Off_S2OnOff_PSC_maxF-R2Off_S2OnOff_PSC_F[-1])
                    R2Off_S2OnOff_PSC_P[-1] = R2Off_S2OnOff_PSC_P[-1] * (1 - self.R2Off_S2OnOff_PSC_fP)
                R2Off_S2OnOff_PSC_de_test3 = torch.any(helper[t] == R1Off_tspike + self.R2Off_R1Off_PSC_delay)
                if R2Off_S2OnOff_PSC_de_test3:
                    R2Off_R1Off_PSC_x[-2] = R2Off_R1Off_PSC_x[-1]
                    R2Off_R1Off_PSC_q[-2] = R2Off_R1Off_PSC_F[-1]
                    R2Off_R1Off_PSC_F[-2] = R2Off_R1Off_PSC_F[-1]
                    R2Off_R1Off_PSC_P[-2] = R2Off_R1Off_PSC_P[-1]
                    R2Off_R1Off_PSC_x[-1] = R2Off_R1Off_PSC_x[-1] + R2Off_R1Off_PSC_q[-1]
                    R2Off_R1Off_PSC_q[-1] = R2Off_R1Off_PSC_F[-1] * R2Off_R1Off_PSC_P[-1]
                    R2Off_R1Off_PSC_F[-1] = R2Off_R1Off_PSC_F[-1] + self.R2Off_R1Off_PSC_fF*(self.R2Off_R1Off_PSC_maxF-R2Off_R1Off_PSC_F[-1])
                    R2Off_R1Off_PSC_P[-1] = R2Off_R1Off_PSC_P[-1] * (1 - self.R2Off_R1Off_PSC_fP)
                R2Off_R1Off_PSC_de_test3 = torch.any(helper[t] == R1Off_tspike + self.S2OnOff_R1Off_PSC_delay)
                if R2Off_R1Off_PSC_de_test3:
                    S2OnOff_R1Off_PSC_x[-2] = S2OnOff_R1Off_PSC_x[-1]
                    S2OnOff_R1Off_PSC_q[-2] = S2OnOff_R1Off_PSC_F[-1]
                    S2OnOff_R1Off_PSC_F[-2] = S2OnOff_R1Off_PSC_F[-1]
                    S2OnOff_R1Off_PSC_P[-2] = S2OnOff_R1Off_PSC_P[-1]
                    S2OnOff_R1Off_PSC_x[-1] = S2OnOff_R1Off_PSC_x[-1] + S2OnOff_R1Off_PSC_q[-1]
                    S2OnOff_R1Off_PSC_q[-1] = S2OnOff_R1Off_PSC_F[-1] * S2OnOff_R1Off_PSC_P[-1]
                    S2OnOff_R1Off_PSC_F[-1] = S2OnOff_R1Off_PSC_F[-1] + self.S2OnOff_R1Off_PSC_fF*(self.S2OnOff_R1Off_PSC_maxF-S2OnOff_R1Off_PSC_F[-1])
                    S2OnOff_R1Off_PSC_P[-1] = S2OnOff_R1Off_PSC_P[-1] * (1 - self.S2OnOff_R1Off_PSC_fP)
            R2On_V_spikes = torch.stack(R2On_V_spikes_holder, dim=0)
            print(len(R2On_V_spikes))

    

            #print(len(spike_holder))   
    

            #print(len(On_V_spikes))  
    

            #spike_holder = torch.cat((spike_holder, On_V_spikes), dim=0)
    

            #spike_holder = spike_holder.view(-1)
    

            #R2On_V_spikes = R2On_V_spikes.view(-1)
    

            #spike_holderOn = torch.cat((spike_holderOn, On_V_spikes), dim=0)
    

            #spike_holderOff = torch.cat((spike_holderOff, Off_V_spikes), dim=0)
    

            #spike_holderR1On = torch.cat((spike_holderR1On, R1On_V_spikes), dim=0)
    

            spike_holderR2On = torch.cat((spike_holderR2On, R2On_V_spikes), dim=0)
    

            #spike_holderS2OnOff = torch.cat((spike_holderS2OnOff, S2OnOff_V_spikes), dim=0)
    

            #spike_holderS1OnOff = torch.cat((spike_holderS1OnOff, S1OnOff_V_spikes), dim=0)
    

            #spike_holderR2Off = torch.cat((spike_holderR2Off, R2Off_V_spikes), dim=0)
    

            #spike_holderR1Off = torch.cat((spike_holderR1Off, R1Off_V_spikes), dim=0)
    

            #print('made it here 5')
    

        #print(max(self.On_On_IC_input))
    

        #print(max(self.Off_Off_IC_input))
    

        #return [On_V_spikes,Off_V_spikes,R1On_V_spikes,R1Off_V_spikes,R2On_V_spikes,S1OnOff_V_spikes,S2OnOff_V_spikes]
    

        #return R2On_V_spk_sum
    

        return spike_holderR2On
    

        #return [spike_holderOn,spike_holderOff,spike_holderR1On,spike_holderR2On,spike_holderS2OnOff,spike_holderS1OnOff,spike_holderR2Off,spike_holderR1Off]



def main():
    
    model = LIF_ODE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.0, 0.999))
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.0)
    num_epochs = 1

    #model = torch.compile(model, backend="inductor") 
    
    target_spikes = torch.tensor(50.0, dtype=torch.float32) #100/s
    


    for epoch in range(num_epochs):

        optimizer.zero_grad()

        #with autocast():
        #    output = model()  # forward pass


        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("/logdir"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for _ in range(5):
                output = model()  # your forward pass

        prof.export_chrome_trace("trace.json")  # view in Chrome

  
        #print("Forward pass ran successfully. Num Spikes")
        print('Avg Firing Rate')
        print(output.sum()/10/3)

        fr = output.sum()/10/3  #total spikes/num_trials/num_seconds
        #fr = output/10/3

        loss = (fr - target_spikes)**2

        print(type(output))                   # Tensor? List?
        print(output.requires_grad)
        print(output.grad_fn)

            
        optimizer.zero_grad()

        #loss.backward() 

        #optimizer.step()
        gc.collect()

        print(f"Epoch {epoch}: Loss = {loss.item()}",flush=True) 
        

    #return outputs, losses
    #return losses

if __name__ == "__main__":
    main()
