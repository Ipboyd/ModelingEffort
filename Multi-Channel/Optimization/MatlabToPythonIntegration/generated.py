
import torch
import torch.nn as nn
import numpy as np
import genPoissonTimes
import genPoissonInputs
import matplotlib.pyplot as plt


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (1.0 / (1.0 + torch.abs(input)) ** 2)
        return grad_input, None



class LIF_ODE(nn.Module):
    def __init__(self):
        super().__init__()
        
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
        self.tspan = torch.tensor(array('d', [0.1, 3500.0]), dtype=torch.float32)
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
        self.On_Npop = torch.tensor(1.0, dtype=torch.float32)
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
        self.Off_Npop = torch.tensor(1.0, dtype=torch.float32)
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
        self.R1On_Npop = torch.tensor(1.0, dtype=torch.float32)
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
        self.R1Off_Npop = torch.tensor(1.0, dtype=torch.float32)
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
        self.S1OnOff_Npop = torch.tensor(1.0, dtype=torch.float32)
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
        self.R2On_Npop = torch.tensor(1.0, dtype=torch.float32)
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
        self.R2Off_Npop = torch.tensor(1.0, dtype=torch.float32)
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
        self.S2OnOff_Npop = torch.tensor(1.0, dtype=torch.float32)
        self.On_On_IC_trial = torch.tensor(1.0, dtype=torch.float32)
        self.On_On_IC_locNum = torch.tensor(15.0, dtype=torch.float32)
        self.On_On_IC_label = 'on'
        self.On_On_IC_t_ref = torch.tensor(1.0, dtype=torch.float32)
        self.On_On_IC_t_ref_rel = torch.tensor(1.0, dtype=torch.float32)
        self.On_On_IC_rec = torch.tensor(2.0, dtype=torch.float32)
        self.On_On_IC_g_postIC = torch.tensor(0.17, dtype=torch.float32)
        self.On_On_IC_E_exc = torch.tensor(0.0, dtype=torch.float32)
        self.Off_Off_IC_trial = torch.tensor(1.0, dtype=torch.float32)
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

        # Fixed Params
        self.On_R = 1/self.On_g_L
        self.On_tau = self.On_C*self.On_R
        self.On_Imask = torch.ones(T, self.On_Npop)
        self.Off_R = 1/self.Off_g_L
        self.Off_tau = self.Off_C*self.Off_R
        self.Off_Imask = torch.ones(T, self.Off_Npop)
        self.R1On_R = 1/self.R1On_g_L
        self.R1On_tau = self.R1On_C*self.R1On_R
        self.R1On_Imask = torch.ones(T, self.R1On_Npop)
        self.R1Off_R = 1/self.R1Off_g_L
        self.R1Off_tau = self.R1Off_C*self.R1Off_R
        self.R1Off_Imask = torch.ones(T, self.R1Off_Npop)
        self.S1OnOff_R = 1/self.S1OnOff_g_L
        self.S1OnOff_tau = self.S1OnOff_C*self.S1OnOff_R
        self.S1OnOff_Imask = torch.ones(T, self.S1OnOff_Npop)
        self.R2On_R = 1/self.R2On_g_L
        self.R2On_tau = self.R2On_C*self.R2On_R
        self.R2On_Imask = torch.ones(T, self.R2On_Npop)
        self.R2Off_R = 1/self.R2Off_g_L
        self.R2Off_tau = self.R2Off_C*self.R2Off_R
        self.R2Off_Imask = torch.ones(T, self.R2Off_Npop)
        self.S2OnOff_R = 1/self.S2OnOff_g_L
        self.S2OnOff_tau = self.S2OnOff_C*self.S2OnOff_R
        self.S2OnOff_Imask = torch.ones(T, self.S2OnOff_Npop)
        self.On_On_IC_netcon = +1.000000000000000e+00
        self.On_On_IC_input = genPoissonInputs.gen_poisson_inputs(self.On_On_IC_trial,self.On_On_IC_locNum,self.On_On_IC_label,self.On_On_IC_t_ref,self.On_On_IC_t_ref_rel,self.On_On_IC_rec)
        self.Off_Off_IC_netcon = +1.000000000000000e+00
        self.Off_Off_IC_input = genPoissonInputs.gen_poisson_inputs(self.Off_Off_IC_trial,self.Off_Off_IC_locNum,self.Off_Off_IC_label,self.Off_Off_IC_t_ref,self.Off_Off_IC_t_ref_rel,self.Off_Off_IC_rec)
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


    def forward(self,t,state):
        
        #State Variables
        
        
        On_V = On_E_L*torch.ones(T, On_Npop)
        On_g_ad = torch.zeros(T, On_Npop)
        Off_V = Off_E_L*torch.ones(T, Off_Npop)
        Off_g_ad = torch.zeros(T, Off_Npop)
        R1On_V = R1On_E_L*torch.ones(T, R1On_Npop)
        R1On_g_ad = torch.zeros(T, R1On_Npop)
        R1Off_V = R1Off_E_L*torch.ones(T, R1Off_Npop)
        R1Off_g_ad = torch.zeros(T, R1Off_Npop)
        S1OnOff_V = S1OnOff_E_L*torch.ones(T, S1OnOff_Npop)
        S1OnOff_g_ad = torch.zeros(T, S1OnOff_Npop)
        R2On_V = R2On_E_L*torch.ones(T, R2On_Npop)
        R2On_g_ad = torch.zeros(T, R2On_Npop)
        R2Off_V = R2Off_E_L*torch.ones(T, R2Off_Npop)
        R2Off_g_ad = torch.zeros(T, R2Off_Npop)
        S2OnOff_V = S2OnOff_E_L*torch.ones(T, S2OnOff_Npop)
        S2OnOff_g_ad = torch.zeros(T, S2OnOff_Npop)
        R1On_On_PSC_s = torch.zeros(T, On_Npop)
        R1On_On_PSC_x = torch.zeros(T, On_Npop)
        R1On_On_PSC_F = torch.ones(T, On_Npop)
        R1On_On_PSC_P = torch.ones(T, On_Npop)
        R1On_On_PSC_q = torch.ones(T, On_Npop)
        S1OnOff_On_PSC_s = torch.zeros(T, On_Npop)
        S1OnOff_On_PSC_x = torch.zeros(T, On_Npop)
        S1OnOff_On_PSC_F = torch.ones(T, On_Npop)
        S1OnOff_On_PSC_P = torch.ones(T, On_Npop)
        S1OnOff_On_PSC_q = torch.ones(T, On_Npop)
        R1On_S1OnOff_PSC_s = torch.zeros(T, S1OnOff_Npop)
        R1On_S1OnOff_PSC_x = torch.zeros(T, S1OnOff_Npop)
        R1On_S1OnOff_PSC_F = torch.ones(T, S1OnOff_Npop)
        R1On_S1OnOff_PSC_P = torch.ones(T, S1OnOff_Npop)
        R1On_S1OnOff_PSC_q = torch.ones(T, S1OnOff_Npop)
        R1Off_S1OnOff_PSC_s = torch.zeros(T, S1OnOff_Npop)
        R1Off_S1OnOff_PSC_x = torch.zeros(T, S1OnOff_Npop)
        R1Off_S1OnOff_PSC_F = torch.ones(T, S1OnOff_Npop)
        R1Off_S1OnOff_PSC_P = torch.ones(T, S1OnOff_Npop)
        R1Off_S1OnOff_PSC_q = torch.ones(T, S1OnOff_Npop)
        R1Off_Off_PSC_s = torch.zeros(T, Off_Npop)
        R1Off_Off_PSC_x = torch.zeros(T, Off_Npop)
        R1Off_Off_PSC_F = torch.ones(T, Off_Npop)
        R1Off_Off_PSC_P = torch.ones(T, Off_Npop)
        R1Off_Off_PSC_q = torch.ones(T, Off_Npop)
        S1OnOff_Off_PSC_s = torch.zeros(T, Off_Npop)
        S1OnOff_Off_PSC_x = torch.zeros(T, Off_Npop)
        S1OnOff_Off_PSC_F = torch.ones(T, Off_Npop)
        S1OnOff_Off_PSC_P = torch.ones(T, Off_Npop)
        S1OnOff_Off_PSC_q = torch.ones(T, Off_Npop)
        R2On_R1On_PSC_s = torch.zeros(T, R1On_Npop)
        R2On_R1On_PSC_x = torch.zeros(T, R1On_Npop)
        R2On_R1On_PSC_F = torch.ones(T, R1On_Npop)
        R2On_R1On_PSC_P = torch.ones(T, R1On_Npop)
        R2On_R1On_PSC_q = torch.ones(T, R1On_Npop)
        S2OnOff_R1On_PSC_s = torch.zeros(T, R1On_Npop)
        S2OnOff_R1On_PSC_x = torch.zeros(T, R1On_Npop)
        S2OnOff_R1On_PSC_F = torch.ones(T, R1On_Npop)
        S2OnOff_R1On_PSC_P = torch.ones(T, R1On_Npop)
        S2OnOff_R1On_PSC_q = torch.ones(T, R1On_Npop)
        R2On_S2OnOff_PSC_s = torch.zeros(T, S2OnOff_Npop)
        R2On_S2OnOff_PSC_x = torch.zeros(T, S2OnOff_Npop)
        R2On_S2OnOff_PSC_F = torch.ones(T, S2OnOff_Npop)
        R2On_S2OnOff_PSC_P = torch.ones(T, S2OnOff_Npop)
        R2On_S2OnOff_PSC_q = torch.ones(T, S2OnOff_Npop)
        R2Off_S2OnOff_PSC_s = torch.zeros(T, S2OnOff_Npop)
        R2Off_S2OnOff_PSC_x = torch.zeros(T, S2OnOff_Npop)
        R2Off_S2OnOff_PSC_F = torch.ones(T, S2OnOff_Npop)
        R2Off_S2OnOff_PSC_P = torch.ones(T, S2OnOff_Npop)
        R2Off_S2OnOff_PSC_q = torch.ones(T, S2OnOff_Npop)
        R2Off_R1Off_PSC_s = torch.zeros(T, R1Off_Npop)
        R2Off_R1Off_PSC_x = torch.zeros(T, R1Off_Npop)
        R2Off_R1Off_PSC_F = torch.ones(T, R1Off_Npop)
        R2Off_R1Off_PSC_P = torch.ones(T, R1Off_Npop)
        R2Off_R1Off_PSC_q = torch.ones(T, R1Off_Npop)
        S2OnOff_R1Off_PSC_s = torch.zeros(T, R1Off_Npop)
        S2OnOff_R1Off_PSC_x = torch.zeros(T, R1Off_Npop)
        S2OnOff_R1Off_PSC_F = torch.ones(T, R1Off_Npop)
        S2OnOff_R1Off_PSC_P = torch.ones(T, R1Off_Npop)
        S2OnOff_R1Off_PSC_q = torch.ones(T, R1Off_Npop)
        R2On_R2On_iNoise_V3_sn = 0 * torch.ones(T, R2On_Npop)
        R2On_R2On_iNoise_V3_xn = 0 * torch.ones(T, R2On_Npop)


        #ODEs

        T = len(np.arange(self.tspan[0],self.dt,self.tspan[1])

        for t in range(1,T):
            On_V_k1 = ( (self.On_E_L-On_V[t-1]) - self.On_R*On_g_ad[t-1]*(On_V[t-1]-self.On_E_k) - self.On_R*((((self.On_On_IC_g_postIC*(self.On_On_IC_input[t]*self.On_On_IC_netcon)*(On_V[t-1]-self.On_On_IC_E_exc))))) + self.On_R*self.On_Itonic*self.On_Imask  ) / self.On_tau
            On_g_ad_k1 = -On_g_ad[t-1] / self.On_tau_ad
            Off_V_k1 = ( (self.Off_E_L-Off_V[t-1]) - self.Off_R*Off_g_ad[t-1]*(Off_V[t-1]-self.Off_E_k) - self.Off_R*((((self.Off_Off_IC_g_postIC*(self.Off_Off_IC_input[t]*self.Off_Off_IC_netcon)*(Off_V[t-1]-self.Off_Off_IC_E_exc))))) + self.Off_R*self.Off_Itonic*self.Off_Imask  ) / self.Off_tau
            Off_g_ad_k1 = -Off_g_ad[t-1] / self.Off_tau_ad
            R1On_V_k1 = ( (self.R1On_E_L-R1On_V[t-1]) - self.R1On_R*R1On_g_ad[t-1]*(R1On_V[t-1]-self.R1On_E_k) - self.R1On_R*((((self.R1On_On_PSC_gSYN*(R1On_On_PSC_s[t-1]*self.R1On_On_PSC_netcon)*(R1On_V[t-1]-self.R1On_On_PSC_ESYN))))+((((self.R1On_S1OnOff_PSC_gSYN*(R1On_S1OnOff_PSC_s[t-1]*self.R1On_S1OnOff_PSC_netcon)*(R1On_V[t-1]-self.R1On_S1OnOff_PSC_ESYN)))))) + self.R1On_R*self.R1On_Itonic*self.R1On_Imask  ) / self.R1On_tau
            R1On_g_ad_k1 = -R1On_g_ad[t-1] / self.R1On_tau_ad
            R1Off_V_k1 = ( (self.R1Off_E_L-R1Off_V[t-1]) - self.R1Off_R*R1Off_g_ad[t-1]*(R1Off_V[t-1]-self.R1Off_E_k) - self.R1Off_R*((((self.R1Off_S1OnOff_PSC_gSYN*(R1Off_S1OnOff_PSC_s[t-1]*self.R1Off_S1OnOff_PSC_netcon)*(R1Off_V[t-1]-self.R1Off_S1OnOff_PSC_ESYN))))+((((self.R1Off_Off_PSC_gSYN*(R1Off_Off_PSC_s[t-1]*self.R1Off_Off_PSC_netcon)*(R1Off_V[t-1]-self.R1Off_Off_PSC_ESYN)))))) + self.R1Off_R*self.R1Off_Itonic*self.R1Off_Imask  ) / self.R1Off_tau
            R1Off_g_ad_k1 = -R1Off_g_ad[t-1] / self.R1Off_tau_ad
            S1OnOff_V_k1 = ( (self.S1OnOff_E_L-S1OnOff_V[t-1]) - self.S1OnOff_R*S1OnOff_g_ad[t-1]*(S1OnOff_V[t-1]-self.S1OnOff_E_k) - self.S1OnOff_R*((((self.S1OnOff_On_PSC_gSYN*(S1OnOff_On_PSC_s[t-1]*self.S1OnOff_On_PSC_netcon)*(S1OnOff_V[t-1]-self.S1OnOff_On_PSC_ESYN))))+((((self.S1OnOff_Off_PSC_gSYN*(S1OnOff_Off_PSC_s[t-1]*self.S1OnOff_Off_PSC_netcon)*(S1OnOff_V[t-1]-self.S1OnOff_Off_PSC_ESYN)))))) + self.S1OnOff_R*self.S1OnOff_Itonic*self.S1OnOff_Imask  ) / self.S1OnOff_tau
            S1OnOff_g_ad_k1 = -S1OnOff_g_ad[t-1] / self.S1OnOff_tau_ad
            R2On_V_k1 = ( (self.R2On_E_L-R2On_V[t-1]) - self.R2On_R*R2On_g_ad[t-1]*(R2On_V[t-1]-self.R2On_E_k) - self.R2On_R*((((self.R2On_R1On_PSC_gSYN*(R2On_R1On_PSC_s[t-1]*self.R2On_R1On_PSC_netcon)*(R2On_V[t-1]-self.R2On_R1On_PSC_ESYN))))+((((self.R2On_S2OnOff_PSC_gSYN*(R2On_S2OnOff_PSC_s[t-1]*self.R2On_S2OnOff_PSC_netcon)*(R2On_V[t-1]-self.R2On_S2OnOff_PSC_ESYN))))+((((self.R2On_R2On_iNoise_V3_nSYN*(R2On_R2On_iNoise_V3_sn[t-1]*self.R2On_R2On_iNoise_V3_netcon)*(R2On_V[t-1]-self.R2On_R2On_iNoise_V3_E_exc))))))) + self.R2On_R*self.R2On_Itonic*self.R2On_Imask  ) / self.R2On_tau
            R2On_g_ad_k1 = -R2On_g_ad[t-1] / self.R2On_tau_ad
            R2Off_V_k1 = ( (self.R2Off_E_L-R2Off_V[t-1]) - self.R2Off_R*R2Off_g_ad[t-1]*(R2Off_V[t-1]-self.R2Off_E_k) - self.R2Off_R*((((self.R2Off_S2OnOff_PSC_gSYN*(R2Off_S2OnOff_PSC_s[t-1]*self.R2Off_S2OnOff_PSC_netcon)*(R2Off_V[t-1]-self.R2Off_S2OnOff_PSC_ESYN))))+((((self.R2Off_R1Off_PSC_gSYN*(R2Off_R1Off_PSC_s[t-1]*self.R2Off_R1Off_PSC_netcon)*(R2Off_V[t-1]-self.R2Off_R1Off_PSC_ESYN)))))) + self.R2Off_R*self.R2Off_Itonic*self.R2Off_Imask  ) / self.R2Off_tau
            R2Off_g_ad_k1 = -R2Off_g_ad[t-1] / self.R2Off_tau_ad
            S2OnOff_V_k1 = ( (self.S2OnOff_E_L-S2OnOff_V[t-1]) - self.S2OnOff_R*S2OnOff_g_ad[t-1]*(S2OnOff_V[t-1]-self.S2OnOff_E_k) - self.S2OnOff_R*((((self.S2OnOff_R1On_PSC_gSYN*(S2OnOff_R1On_PSC_s[t-1]*self.S2OnOff_R1On_PSC_netcon)*(S2OnOff_V[t-1]-self.S2OnOff_R1On_PSC_ESYN))))+((((self.S2OnOff_R1Off_PSC_gSYN*(S2OnOff_R1Off_PSC_s[t-1]*self.S2OnOff_R1Off_PSC_netcon)*(S2OnOff_V[t-1]-self.S2OnOff_R1Off_PSC_ESYN)))))) + self.S2OnOff_R*self.S2OnOff_Itonic*self.S2OnOff_Imask  ) / self.S2OnOff_tau
            S2OnOff_g_ad_k1 = -S2OnOff_g_ad[t-1] / self.S2OnOff_tau_ad
            R1On_On_PSC_s_k1 = ( self.R1On_On_PSC_scale * R1On_On_PSC_x[t-1] - R1On_On_PSC_s[t-1] )/self.R1On_On_PSC_tauR
            R1On_On_PSC_x_k1 = -R1On_On_PSC_x[t-1]/self.R1On_On_PSC_tauD
            R1On_On_PSC_F_k1 = (1 - R1On_On_PSC_F[t-1])/self.R1On_On_PSC_tauF
            R1On_On_PSC_P_k1 = (1 - R1On_On_PSC_P[t-1])/self.R1On_On_PSC_tauP
            R1On_On_PSC_q_k1 = 0
            S1OnOff_On_PSC_s_k1 = ( self.S1OnOff_On_PSC_scale * S1OnOff_On_PSC_x[t-1] - S1OnOff_On_PSC_s[t-1] )/self.S1OnOff_On_PSC_tauR
            S1OnOff_On_PSC_x_k1 = -S1OnOff_On_PSC_x[t-1]/self.S1OnOff_On_PSC_tauD
            S1OnOff_On_PSC_F_k1 = (1 - S1OnOff_On_PSC_F[t-1])/self.S1OnOff_On_PSC_tauF
            S1OnOff_On_PSC_P_k1 = (1 - S1OnOff_On_PSC_P[t-1])/self.S1OnOff_On_PSC_tauP
            S1OnOff_On_PSC_q_k1 = 0
            R1On_S1OnOff_PSC_s_k1 = ( self.R1On_S1OnOff_PSC_scale * R1On_S1OnOff_PSC_x[t-1] - R1On_S1OnOff_PSC_s[t-1] )/self.R1On_S1OnOff_PSC_tauR
            R1On_S1OnOff_PSC_x_k1 = -R1On_S1OnOff_PSC_x[t-1]/self.R1On_S1OnOff_PSC_tauD
            R1On_S1OnOff_PSC_F_k1 = (1 - R1On_S1OnOff_PSC_F[t-1])/self.R1On_S1OnOff_PSC_tauF
            R1On_S1OnOff_PSC_P_k1 = (1 - R1On_S1OnOff_PSC_P[t-1])/self.R1On_S1OnOff_PSC_tauP
            R1On_S1OnOff_PSC_q_k1 = 0
            R1Off_S1OnOff_PSC_s_k1 = ( self.R1Off_S1OnOff_PSC_scale * R1Off_S1OnOff_PSC_x[t-1] - R1Off_S1OnOff_PSC_s[t-1] )/self.R1Off_S1OnOff_PSC_tauR
            R1Off_S1OnOff_PSC_x_k1 = -R1Off_S1OnOff_PSC_x[t-1]/self.R1Off_S1OnOff_PSC_tauD
            R1Off_S1OnOff_PSC_F_k1 = (1 - R1Off_S1OnOff_PSC_F[t-1])/self.R1Off_S1OnOff_PSC_tauF
            R1Off_S1OnOff_PSC_P_k1 = (1 - R1Off_S1OnOff_PSC_P[t-1])/self.R1Off_S1OnOff_PSC_tauP
            R1Off_S1OnOff_PSC_q_k1 = 0
            R1Off_Off_PSC_s_k1 = ( self.R1Off_Off_PSC_scale * R1Off_Off_PSC_x[t-1] - R1Off_Off_PSC_s[t-1] )/self.R1Off_Off_PSC_tauR
            R1Off_Off_PSC_x_k1 = -R1Off_Off_PSC_x[t-1]/self.R1Off_Off_PSC_tauD
            R1Off_Off_PSC_F_k1 = (1 - R1Off_Off_PSC_F[t-1])/self.R1Off_Off_PSC_tauF
            R1Off_Off_PSC_P_k1 = (1 - R1Off_Off_PSC_P[t-1])/self.R1Off_Off_PSC_tauP
            R1Off_Off_PSC_q_k1 = 0
            S1OnOff_Off_PSC_s_k1 = ( self.S1OnOff_Off_PSC_scale * S1OnOff_Off_PSC_x[t-1] - S1OnOff_Off_PSC_s[t-1] )/self.S1OnOff_Off_PSC_tauR
            S1OnOff_Off_PSC_x_k1 = -S1OnOff_Off_PSC_x[t-1]/self.S1OnOff_Off_PSC_tauD
            S1OnOff_Off_PSC_F_k1 = (1 - S1OnOff_Off_PSC_F[t-1])/self.S1OnOff_Off_PSC_tauF
            S1OnOff_Off_PSC_P_k1 = (1 - S1OnOff_Off_PSC_P[t-1])/self.S1OnOff_Off_PSC_tauP
            S1OnOff_Off_PSC_q_k1 = 0
            R2On_R1On_PSC_s_k1 = ( self.R2On_R1On_PSC_scale * R2On_R1On_PSC_x[t-1] - R2On_R1On_PSC_s[t-1] )/self.R2On_R1On_PSC_tauR
            R2On_R1On_PSC_x_k1 = -R2On_R1On_PSC_x[t-1]/self.R2On_R1On_PSC_tauD
            R2On_R1On_PSC_F_k1 = (1 - R2On_R1On_PSC_F[t-1])/self.R2On_R1On_PSC_tauF
            R2On_R1On_PSC_P_k1 = (1 - R2On_R1On_PSC_P[t-1])/self.R2On_R1On_PSC_tauP
            R2On_R1On_PSC_q_k1 = 0
            S2OnOff_R1On_PSC_s_k1 = ( self.S2OnOff_R1On_PSC_scale * S2OnOff_R1On_PSC_x[t-1] - S2OnOff_R1On_PSC_s[t-1] )/self.S2OnOff_R1On_PSC_tauR
            S2OnOff_R1On_PSC_x_k1 = -S2OnOff_R1On_PSC_x[t-1]/self.S2OnOff_R1On_PSC_tauD
            S2OnOff_R1On_PSC_F_k1 = (1 - S2OnOff_R1On_PSC_F[t-1])/self.S2OnOff_R1On_PSC_tauF
            S2OnOff_R1On_PSC_P_k1 = (1 - S2OnOff_R1On_PSC_P[t-1])/self.S2OnOff_R1On_PSC_tauP
            S2OnOff_R1On_PSC_q_k1 = 0
            R2On_S2OnOff_PSC_s_k1 = ( self.R2On_S2OnOff_PSC_scale * R2On_S2OnOff_PSC_x[t-1] - R2On_S2OnOff_PSC_s[t-1] )/self.R2On_S2OnOff_PSC_tauR
            R2On_S2OnOff_PSC_x_k1 = -R2On_S2OnOff_PSC_x[t-1]/self.R2On_S2OnOff_PSC_tauD
            R2On_S2OnOff_PSC_F_k1 = (1 - R2On_S2OnOff_PSC_F[t-1])/self.R2On_S2OnOff_PSC_tauF
            R2On_S2OnOff_PSC_P_k1 = (1 - R2On_S2OnOff_PSC_P[t-1])/self.R2On_S2OnOff_PSC_tauP
            R2On_S2OnOff_PSC_q_k1 = 0
            R2Off_S2OnOff_PSC_s_k1 = ( self.R2Off_S2OnOff_PSC_scale * R2Off_S2OnOff_PSC_x[t-1] - R2Off_S2OnOff_PSC_s[t-1] )/self.R2Off_S2OnOff_PSC_tauR
            R2Off_S2OnOff_PSC_x_k1 = -R2Off_S2OnOff_PSC_x[t-1]/self.R2Off_S2OnOff_PSC_tauD
            R2Off_S2OnOff_PSC_F_k1 = (1 - R2Off_S2OnOff_PSC_F[t-1])/self.R2Off_S2OnOff_PSC_tauF
            R2Off_S2OnOff_PSC_P_k1 = (1 - R2Off_S2OnOff_PSC_P[t-1])/self.R2Off_S2OnOff_PSC_tauP
            R2Off_S2OnOff_PSC_q_k1 = 0
            R2Off_R1Off_PSC_s_k1 = ( self.R2Off_R1Off_PSC_scale * R2Off_R1Off_PSC_x[t-1] - R2Off_R1Off_PSC_s[t-1] )/self.R2Off_R1Off_PSC_tauR
            R2Off_R1Off_PSC_x_k1 = -R2Off_R1Off_PSC_x[t-1]/self.R2Off_R1Off_PSC_tauD
            R2Off_R1Off_PSC_F_k1 = (1 - R2Off_R1Off_PSC_F[t-1])/self.R2Off_R1Off_PSC_tauF
            R2Off_R1Off_PSC_P_k1 = (1 - R2Off_R1Off_PSC_P[t-1])/self.R2Off_R1Off_PSC_tauP
            R2Off_R1Off_PSC_q_k1 = 0
            S2OnOff_R1Off_PSC_s_k1 = ( self.S2OnOff_R1Off_PSC_scale * S2OnOff_R1Off_PSC_x[t-1] - S2OnOff_R1Off_PSC_s[t-1] )/self.S2OnOff_R1Off_PSC_tauR
            S2OnOff_R1Off_PSC_x_k1 = -S2OnOff_R1Off_PSC_x[t-1]/self.S2OnOff_R1Off_PSC_tauD
            S2OnOff_R1Off_PSC_F_k1 = (1 - S2OnOff_R1Off_PSC_F[t-1])/self.S2OnOff_R1Off_PSC_tauF
            S2OnOff_R1Off_PSC_P_k1 = (1 - S2OnOff_R1Off_PSC_P[t-1])/self.S2OnOff_R1Off_PSC_tauP
            S2OnOff_R1Off_PSC_q_k1 = 0
            R2On_R2On_iNoise_V3_sn_k1 = ( self.R2On_R2On_iNoise_V3_scale * R2On_R2On_iNoise_V3_xn[t-1] - R2On_R2On_iNoise_V3_sn[t-1] )/self.R2On_R2On_iNoise_V3_tauR_N
            R2On_R2On_iNoise_V3_xn_k1 = -R2On_R2On_iNoise_V3_xn[t-1]/self.R2On_R2On_iNoise_V3_tauD_N + self.R2On_R2On_iNoise_V3_token[t]/self.R2On_R2On_iNoise_V3_dt


def main():
      model = LIF_ODE()
      init_cond = (    model.On_E_L*torch.ones(T, model.On_Npop),
    torch.zeros(T, model.On_Npop),
    model.Off_E_L*torch.ones(T, model.Off_Npop),
    torch.zeros(T, model.Off_Npop),
    model.R1On_E_L*torch.ones(T, model.R1On_Npop),
    torch.zeros(T, model.R1On_Npop),
    model.R1Off_E_L*torch.ones(T, model.R1Off_Npop),
    torch.zeros(T, model.R1Off_Npop),
    model.S1OnOff_E_L*torch.ones(T, model.S1OnOff_Npop),
    torch.zeros(T, model.S1OnOff_Npop),
    model.R2On_E_L*torch.ones(T, model.R2On_Npop),
    torch.zeros(T, model.R2On_Npop),
    model.R2Off_E_L*torch.ones(T, model.R2Off_Npop),
    torch.zeros(T, model.R2Off_Npop),
    model.S2OnOff_E_L*torch.ones(T, model.S2OnOff_Npop),
    torch.zeros(T, model.S2OnOff_Npop),
    torch.zeros(T, model.On_Npop),
    torch.zeros(T, model.On_Npop),
    torch.ones(T, model.On_Npop),
    torch.ones(T, model.On_Npop),
    torch.ones(T, model.On_Npop),
    torch.zeros(T, model.On_Npop),
    torch.zeros(T, model.On_Npop),
    torch.ones(T, model.On_Npop),
    torch.ones(T, model.On_Npop),
    torch.ones(T, model.On_Npop),
    torch.zeros(T, model.S1OnOff_Npop),
    torch.zeros(T, model.S1OnOff_Npop),
    torch.ones(T, model.S1OnOff_Npop),
    torch.ones(T, model.S1OnOff_Npop),
    torch.ones(T, model.S1OnOff_Npop),
    torch.zeros(T, model.S1OnOff_Npop),
    torch.zeros(T, model.S1OnOff_Npop),
    torch.ones(T, model.S1OnOff_Npop),
    torch.ones(T, model.S1OnOff_Npop),
    torch.ones(T, model.S1OnOff_Npop),
    torch.zeros(T, model.Off_Npop),
    torch.zeros(T, model.Off_Npop),
    torch.ones(T, model.Off_Npop),
    torch.ones(T, model.Off_Npop),
    torch.ones(T, model.Off_Npop),
    torch.zeros(T, model.Off_Npop),
    torch.zeros(T, model.Off_Npop),
    torch.ones(T, model.Off_Npop),
    torch.ones(T, model.Off_Npop),
    torch.ones(T, model.Off_Npop),
    torch.zeros(T, model.R1On_Npop),
    torch.zeros(T, model.R1On_Npop),
    torch.ones(T, model.R1On_Npop),
    torch.ones(T, model.R1On_Npop),
    torch.ones(T, model.R1On_Npop),
    torch.zeros(T, model.R1On_Npop),
    torch.zeros(T, model.R1On_Npop),
    torch.ones(T, model.R1On_Npop),
    torch.ones(T, model.R1On_Npop),
    torch.ones(T, model.R1On_Npop),
    torch.zeros(T, model.S2OnOff_Npop),
    torch.zeros(T, model.S2OnOff_Npop),
    torch.ones(T, model.S2OnOff_Npop),
    torch.ones(T, model.S2OnOff_Npop),
    torch.ones(T, model.S2OnOff_Npop),
    torch.zeros(T, model.S2OnOff_Npop),
    torch.zeros(T, model.S2OnOff_Npop),
    torch.ones(T, model.S2OnOff_Npop),
    torch.ones(T, model.S2OnOff_Npop),
    torch.ones(T, model.S2OnOff_Npop),
    torch.zeros(T, model.R1Off_Npop),
    torch.zeros(T, model.R1Off_Npop),
    torch.ones(T, model.R1Off_Npop),
    torch.ones(T, model.R1Off_Npop),
    torch.ones(T, model.R1Off_Npop),
    torch.zeros(T, model.R1Off_Npop),
    torch.zeros(T, model.R1Off_Npop),
    torch.ones(T, model.R1Off_Npop),
    torch.ones(T, model.R1Off_Npop),
    torch.ones(T, model.R1Off_Npop),
    0 * torch.ones(T, model.R2On_Npop),
    0 * torch.ones(T, model.R2On_Npop))

    t = torch.linspace(0, 5, 100)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


    # Training loop
    for step in range(500):
        optimizer.zero_grad()
    
        pred_z = odeint(model, ninit_cond, t)  # pred_z shape: (T, state_dim)

        # Suppose R2On_V is stored as part of the full state vector at index `i_R2On_V`
        R2On_V_trace = pred_z[:, 10]  # shape: (T, N_R2On)

        # Compute average firing rate over time
        avg_fr = compute_avg_firing_rate(R2On_V_trace)

        print(avg_fr)

        # Define a target firing rate (e.g., 10 Hz)
        target_fr = torch.tensor(0.01)  # if time units are seconds and you're simulating 1000 steps

        loss = (avg_fr - target_fr).pow(2)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 499:
            print(f"Step {step} | Loss: {loss.item():.6f} | Avg FR: {avg_fr.item():.6f}")

if __name__ == "__main__":
    main()



