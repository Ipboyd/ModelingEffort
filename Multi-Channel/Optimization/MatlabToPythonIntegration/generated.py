
import torch
import torch.nn as nn


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
        self.downsample_factor = torch.tensor(1.0, dtype=torch.float32)
        self.random_seed = torch.tensor(shuffle, dtype=torch.float32)
        self.solver = torch.tensor(euler, dtype=torch.float32)
        self.disk_flag = torch.tensor(0.0, dtype=torch.float32)
        self.dt = torch.tensor(0.1, dtype=torch.float32)
        self.datafile = torch.tensor(data.csv, dtype=torch.float32)
        self.mex_flag = torch.tensor(0.0, dtype=torch.float32)
        self.verbose_flag = torch.tensor(0.0, dtype=torch.float32)
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
        self.On_On_IC_trial = torch.tensor(20.0, dtype=torch.float32)
        self.On_On_IC_locNum = torch.tensor(15.0, dtype=torch.float32)
        self.On_On_IC_label = torch.tensor('on', dtype=torch.float32)
        self.On_On_IC_t_ref = torch.tensor(1.0, dtype=torch.float32)
        self.On_On_IC_t_ref_rel = torch.tensor(1.0, dtype=torch.float32)
        self.On_On_IC_rec = torch.tensor(2.0, dtype=torch.float32)
        self.On_On_IC_g_postIC = torch.tensor(0.17, dtype=torch.float32)
        self.On_On_IC_E_exc = torch.tensor(0.0, dtype=torch.float32)
        self.Off_Off_IC_trial = torch.tensor(20.0, dtype=torch.float32)
        self.Off_Off_IC_locNum = torch.tensor(15.0, dtype=torch.float32)
        self.Off_Off_IC_label = torch.tensor('off', dtype=torch.float32)
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


    def forward(self,t,state):
        
        #State Variables
        
        
        On_V, On_g_ad, Off_V, Off_g_ad, R1On_V, R1On_g_ad, R1Off_V, R1Off_g_ad, S1OnOff_V, S1OnOff_g_ad, R2On_V, R2On_g_ad, R2Off_V, R2Off_g_ad, S2OnOff_V, S2OnOff_g_ad, R1On_On_PSC_s, R1On_On_PSC_x, R1On_On_PSC_F, R1On_On_PSC_P, R1On_On_PSC_q, S1OnOff_On_PSC_s, S1OnOff_On_PSC_x, S1OnOff_On_PSC_F, S1OnOff_On_PSC_P, S1OnOff_On_PSC_q, R1On_S1OnOff_PSC_s, R1On_S1OnOff_PSC_x, R1On_S1OnOff_PSC_F, R1On_S1OnOff_PSC_P, R1On_S1OnOff_PSC_q, R1Off_S1OnOff_PSC_s, R1Off_S1OnOff_PSC_x, R1Off_S1OnOff_PSC_F, R1Off_S1OnOff_PSC_P, R1Off_S1OnOff_PSC_q, R1Off_Off_PSC_s, R1Off_Off_PSC_x, R1Off_Off_PSC_F, R1Off_Off_PSC_P, R1Off_Off_PSC_q, S1OnOff_Off_PSC_s, S1OnOff_Off_PSC_x, S1OnOff_Off_PSC_F, S1OnOff_Off_PSC_P, S1OnOff_Off_PSC_q, R2On_R1On_PSC_s, R2On_R1On_PSC_x, R2On_R1On_PSC_F, R2On_R1On_PSC_P, R2On_R1On_PSC_q, S2OnOff_R1On_PSC_s, S2OnOff_R1On_PSC_x, S2OnOff_R1On_PSC_F, S2OnOff_R1On_PSC_P, S2OnOff_R1On_PSC_q, R2On_S2OnOff_PSC_s, R2On_S2OnOff_PSC_x, R2On_S2OnOff_PSC_F, R2On_S2OnOff_PSC_P, R2On_S2OnOff_PSC_q, R2Off_S2OnOff_PSC_s, R2Off_S2OnOff_PSC_x, R2Off_S2OnOff_PSC_F, R2Off_S2OnOff_PSC_P, R2Off_S2OnOff_PSC_q, R2Off_R1Off_PSC_s, R2Off_R1Off_PSC_x, R2Off_R1Off_PSC_F, R2Off_R1Off_PSC_P, R2Off_R1Off_PSC_q, S2OnOff_R1Off_PSC_s, S2OnOff_R1Off_PSC_x, S2OnOff_R1Off_PSC_F, S2OnOff_R1Off_PSC_P, S2OnOff_R1Off_PSC_q, R2On_R2On_iNoise_V3_sn, R2On_R2On_iNoise_V3_xn = state

        #ODEs
        On_V_k1 = ( (On_E_L-On_V(n-1,:)) - On_R*On_g_ad(n-1,:).*(On_V(n-1,:)-On_E_k) - On_R*((((On_On_IC_g_postIC*(On_On_IC_input(k,:)*On_On_IC_netcon).*(On_V(n-1,:)-On_On_IC_E_exc))))) + On_R*On_Itonic.*On_Imask + On_R*On_noise.*randn(1,On_Npop) ) / On_tau
        On_g_ad_k1 = -On_g_ad(n-1,:) / On_tau_ad
        Off_V_k1 = ( (Off_E_L-Off_V(n-1,:)) - Off_R*Off_g_ad(n-1,:).*(Off_V(n-1,:)-Off_E_k) - Off_R*((((Off_Off_IC_g_postIC*(Off_Off_IC_input(k,:)*Off_Off_IC_netcon).*(Off_V(n-1,:)-Off_Off_IC_E_exc))))) + Off_R*Off_Itonic.*Off_Imask + Off_R*Off_noise.*randn(1,Off_Npop) ) / Off_tau
        Off_g_ad_k1 = -Off_g_ad(n-1,:) / Off_tau_ad
        R1On_V_k1 = ( (R1On_E_L-R1On_V(n-1,:)) - R1On_R*R1On_g_ad(n-1,:).*(R1On_V(n-1,:)-R1On_E_k) - R1On_R*((((R1On_On_PSC_gSYN.*(R1On_On_PSC_s(n-1,:)*R1On_On_PSC_netcon).*(R1On_V(n-1,:)-R1On_On_PSC_ESYN))))+((((R1On_S1OnOff_PSC_gSYN.*(R1On_S1OnOff_PSC_s(n-1,:)*R1On_S1OnOff_PSC_netcon).*(R1On_V(n-1,:)-R1On_S1OnOff_PSC_ESYN)))))) + R1On_R*R1On_Itonic.*R1On_Imask + R1On_R*R1On_noise.*randn(1,R1On_Npop) ) / R1On_tau
        R1On_g_ad_k1 = -R1On_g_ad(n-1,:) / R1On_tau_ad
        R1Off_V_k1 = ( (R1Off_E_L-R1Off_V(n-1,:)) - R1Off_R*R1Off_g_ad(n-1,:).*(R1Off_V(n-1,:)-R1Off_E_k) - R1Off_R*((((R1Off_S1OnOff_PSC_gSYN.*(R1Off_S1OnOff_PSC_s(n-1,:)*R1Off_S1OnOff_PSC_netcon).*(R1Off_V(n-1,:)-R1Off_S1OnOff_PSC_ESYN))))+((((R1Off_Off_PSC_gSYN.*(R1Off_Off_PSC_s(n-1,:)*R1Off_Off_PSC_netcon).*(R1Off_V(n-1,:)-R1Off_Off_PSC_ESYN)))))) + R1Off_R*R1Off_Itonic.*R1Off_Imask + R1Off_R*R1Off_noise.*randn(1,R1Off_Npop) ) / R1Off_tau
        R1Off_g_ad_k1 = -R1Off_g_ad(n-1,:) / R1Off_tau_ad
        S1OnOff_V_k1 = ( (S1OnOff_E_L-S1OnOff_V(n-1,:)) - S1OnOff_R*S1OnOff_g_ad(n-1,:).*(S1OnOff_V(n-1,:)-S1OnOff_E_k) - S1OnOff_R*((((S1OnOff_On_PSC_gSYN.*(S1OnOff_On_PSC_s(n-1,:)*S1OnOff_On_PSC_netcon).*(S1OnOff_V(n-1,:)-S1OnOff_On_PSC_ESYN))))+((((S1OnOff_Off_PSC_gSYN.*(S1OnOff_Off_PSC_s(n-1,:)*S1OnOff_Off_PSC_netcon).*(S1OnOff_V(n-1,:)-S1OnOff_Off_PSC_ESYN)))))) + S1OnOff_R*S1OnOff_Itonic.*S1OnOff_Imask + S1OnOff_R*S1OnOff_noise.*randn(1,S1OnOff_Npop) ) / S1OnOff_tau
        S1OnOff_g_ad_k1 = -S1OnOff_g_ad(n-1,:) / S1OnOff_tau_ad
        R2On_V_k1 = ( (R2On_E_L-R2On_V(n-1,:)) - R2On_R*R2On_g_ad(n-1,:).*(R2On_V(n-1,:)-R2On_E_k) - R2On_R*((((R2On_R1On_PSC_gSYN.*(R2On_R1On_PSC_s(n-1,:)*R2On_R1On_PSC_netcon).*(R2On_V(n-1,:)-R2On_R1On_PSC_ESYN))))+((((R2On_S2OnOff_PSC_gSYN.*(R2On_S2OnOff_PSC_s(n-1,:)*R2On_S2OnOff_PSC_netcon).*(R2On_V(n-1,:)-R2On_S2OnOff_PSC_ESYN))))+((((R2On_R2On_iNoise_V3_nSYN.*(R2On_R2On_iNoise_V3_sn(n-1,:)*R2On_R2On_iNoise_V3_netcon).*(R2On_V(n-1,:)-R2On_R2On_iNoise_V3_E_exc))))))) + R2On_R*R2On_Itonic.*R2On_Imask + R2On_R*R2On_noise.*randn(1,R2On_Npop) ) / R2On_tau
        R2On_g_ad_k1 = -R2On_g_ad(n-1,:) / R2On_tau_ad
        R2Off_V_k1 = ( (R2Off_E_L-R2Off_V(n-1,:)) - R2Off_R*R2Off_g_ad(n-1,:).*(R2Off_V(n-1,:)-R2Off_E_k) - R2Off_R*((((R2Off_S2OnOff_PSC_gSYN.*(R2Off_S2OnOff_PSC_s(n-1,:)*R2Off_S2OnOff_PSC_netcon).*(R2Off_V(n-1,:)-R2Off_S2OnOff_PSC_ESYN))))+((((R2Off_R1Off_PSC_gSYN.*(R2Off_R1Off_PSC_s(n-1,:)*R2Off_R1Off_PSC_netcon).*(R2Off_V(n-1,:)-R2Off_R1Off_PSC_ESYN)))))) + R2Off_R*R2Off_Itonic.*R2Off_Imask + R2Off_R*R2Off_noise.*randn(1,R2Off_Npop) ) / R2Off_tau
        R2Off_g_ad_k1 = -R2Off_g_ad(n-1,:) / R2Off_tau_ad
        S2OnOff_V_k1 = ( (S2OnOff_E_L-S2OnOff_V(n-1,:)) - S2OnOff_R*S2OnOff_g_ad(n-1,:).*(S2OnOff_V(n-1,:)-S2OnOff_E_k) - S2OnOff_R*((((S2OnOff_R1On_PSC_gSYN.*(S2OnOff_R1On_PSC_s(n-1,:)*S2OnOff_R1On_PSC_netcon).*(S2OnOff_V(n-1,:)-S2OnOff_R1On_PSC_ESYN))))+((((S2OnOff_R1Off_PSC_gSYN.*(S2OnOff_R1Off_PSC_s(n-1,:)*S2OnOff_R1Off_PSC_netcon).*(S2OnOff_V(n-1,:)-S2OnOff_R1Off_PSC_ESYN)))))) + S2OnOff_R*S2OnOff_Itonic.*S2OnOff_Imask + S2OnOff_R*S2OnOff_noise.*randn(1,S2OnOff_Npop) ) / S2OnOff_tau
        S2OnOff_g_ad_k1 = -S2OnOff_g_ad(n-1,:) / S2OnOff_tau_ad
        R1On_On_PSC_s_k1 = ( R1On_On_PSC_scale * R1On_On_PSC_x(n-1,:) - R1On_On_PSC_s(n-1,:) )/R1On_On_PSC_tauR
        R1On_On_PSC_x_k1 = -R1On_On_PSC_x(n-1,:)/R1On_On_PSC_tauD
        R1On_On_PSC_F_k1 = (1 - R1On_On_PSC_F(n-1,:))/R1On_On_PSC_tauF
        R1On_On_PSC_P_k1 = (1 - R1On_On_PSC_P(n-1,:))/R1On_On_PSC_tauP
        R1On_On_PSC_q_k1 = 0
        S1OnOff_On_PSC_s_k1 = ( S1OnOff_On_PSC_scale * S1OnOff_On_PSC_x(n-1,:) - S1OnOff_On_PSC_s(n-1,:) )/S1OnOff_On_PSC_tauR
        S1OnOff_On_PSC_x_k1 = -S1OnOff_On_PSC_x(n-1,:)/S1OnOff_On_PSC_tauD
        S1OnOff_On_PSC_F_k1 = (1 - S1OnOff_On_PSC_F(n-1,:))/S1OnOff_On_PSC_tauF
        S1OnOff_On_PSC_P_k1 = (1 - S1OnOff_On_PSC_P(n-1,:))/S1OnOff_On_PSC_tauP
        S1OnOff_On_PSC_q_k1 = 0
        R1On_S1OnOff_PSC_s_k1 = ( R1On_S1OnOff_PSC_scale * R1On_S1OnOff_PSC_x(n-1,:) - R1On_S1OnOff_PSC_s(n-1,:) )/R1On_S1OnOff_PSC_tauR
        R1On_S1OnOff_PSC_x_k1 = -R1On_S1OnOff_PSC_x(n-1,:)/R1On_S1OnOff_PSC_tauD
        R1On_S1OnOff_PSC_F_k1 = (1 - R1On_S1OnOff_PSC_F(n-1,:))/R1On_S1OnOff_PSC_tauF
        R1On_S1OnOff_PSC_P_k1 = (1 - R1On_S1OnOff_PSC_P(n-1,:))/R1On_S1OnOff_PSC_tauP
        R1On_S1OnOff_PSC_q_k1 = 0
        R1Off_S1OnOff_PSC_s_k1 = ( R1Off_S1OnOff_PSC_scale * R1Off_S1OnOff_PSC_x(n-1,:) - R1Off_S1OnOff_PSC_s(n-1,:) )/R1Off_S1OnOff_PSC_tauR
        R1Off_S1OnOff_PSC_x_k1 = -R1Off_S1OnOff_PSC_x(n-1,:)/R1Off_S1OnOff_PSC_tauD
        R1Off_S1OnOff_PSC_F_k1 = (1 - R1Off_S1OnOff_PSC_F(n-1,:))/R1Off_S1OnOff_PSC_tauF
        R1Off_S1OnOff_PSC_P_k1 = (1 - R1Off_S1OnOff_PSC_P(n-1,:))/R1Off_S1OnOff_PSC_tauP
        R1Off_S1OnOff_PSC_q_k1 = 0
        R1Off_Off_PSC_s_k1 = ( R1Off_Off_PSC_scale * R1Off_Off_PSC_x(n-1,:) - R1Off_Off_PSC_s(n-1,:) )/R1Off_Off_PSC_tauR
        R1Off_Off_PSC_x_k1 = -R1Off_Off_PSC_x(n-1,:)/R1Off_Off_PSC_tauD
        R1Off_Off_PSC_F_k1 = (1 - R1Off_Off_PSC_F(n-1,:))/R1Off_Off_PSC_tauF
        R1Off_Off_PSC_P_k1 = (1 - R1Off_Off_PSC_P(n-1,:))/R1Off_Off_PSC_tauP
        R1Off_Off_PSC_q_k1 = 0
        S1OnOff_Off_PSC_s_k1 = ( S1OnOff_Off_PSC_scale * S1OnOff_Off_PSC_x(n-1,:) - S1OnOff_Off_PSC_s(n-1,:) )/S1OnOff_Off_PSC_tauR
        S1OnOff_Off_PSC_x_k1 = -S1OnOff_Off_PSC_x(n-1,:)/S1OnOff_Off_PSC_tauD
        S1OnOff_Off_PSC_F_k1 = (1 - S1OnOff_Off_PSC_F(n-1,:))/S1OnOff_Off_PSC_tauF
        S1OnOff_Off_PSC_P_k1 = (1 - S1OnOff_Off_PSC_P(n-1,:))/S1OnOff_Off_PSC_tauP
        S1OnOff_Off_PSC_q_k1 = 0
        R2On_R1On_PSC_s_k1 = ( R2On_R1On_PSC_scale * R2On_R1On_PSC_x(n-1,:) - R2On_R1On_PSC_s(n-1,:) )/R2On_R1On_PSC_tauR
        R2On_R1On_PSC_x_k1 = -R2On_R1On_PSC_x(n-1,:)/R2On_R1On_PSC_tauD
        R2On_R1On_PSC_F_k1 = (1 - R2On_R1On_PSC_F(n-1,:))/R2On_R1On_PSC_tauF
        R2On_R1On_PSC_P_k1 = (1 - R2On_R1On_PSC_P(n-1,:))/R2On_R1On_PSC_tauP
        R2On_R1On_PSC_q_k1 = 0
        S2OnOff_R1On_PSC_s_k1 = ( S2OnOff_R1On_PSC_scale * S2OnOff_R1On_PSC_x(n-1,:) - S2OnOff_R1On_PSC_s(n-1,:) )/S2OnOff_R1On_PSC_tauR
        S2OnOff_R1On_PSC_x_k1 = -S2OnOff_R1On_PSC_x(n-1,:)/S2OnOff_R1On_PSC_tauD
        S2OnOff_R1On_PSC_F_k1 = (1 - S2OnOff_R1On_PSC_F(n-1,:))/S2OnOff_R1On_PSC_tauF
        S2OnOff_R1On_PSC_P_k1 = (1 - S2OnOff_R1On_PSC_P(n-1,:))/S2OnOff_R1On_PSC_tauP
        S2OnOff_R1On_PSC_q_k1 = 0
        R2On_S2OnOff_PSC_s_k1 = ( R2On_S2OnOff_PSC_scale * R2On_S2OnOff_PSC_x(n-1,:) - R2On_S2OnOff_PSC_s(n-1,:) )/R2On_S2OnOff_PSC_tauR
        R2On_S2OnOff_PSC_x_k1 = -R2On_S2OnOff_PSC_x(n-1,:)/R2On_S2OnOff_PSC_tauD
        R2On_S2OnOff_PSC_F_k1 = (1 - R2On_S2OnOff_PSC_F(n-1,:))/R2On_S2OnOff_PSC_tauF
        R2On_S2OnOff_PSC_P_k1 = (1 - R2On_S2OnOff_PSC_P(n-1,:))/R2On_S2OnOff_PSC_tauP
        R2On_S2OnOff_PSC_q_k1 = 0
        R2Off_S2OnOff_PSC_s_k1 = ( R2Off_S2OnOff_PSC_scale * R2Off_S2OnOff_PSC_x(n-1,:) - R2Off_S2OnOff_PSC_s(n-1,:) )/R2Off_S2OnOff_PSC_tauR
        R2Off_S2OnOff_PSC_x_k1 = -R2Off_S2OnOff_PSC_x(n-1,:)/R2Off_S2OnOff_PSC_tauD
        R2Off_S2OnOff_PSC_F_k1 = (1 - R2Off_S2OnOff_PSC_F(n-1,:))/R2Off_S2OnOff_PSC_tauF
        R2Off_S2OnOff_PSC_P_k1 = (1 - R2Off_S2OnOff_PSC_P(n-1,:))/R2Off_S2OnOff_PSC_tauP
        R2Off_S2OnOff_PSC_q_k1 = 0
        R2Off_R1Off_PSC_s_k1 = ( R2Off_R1Off_PSC_scale * R2Off_R1Off_PSC_x(n-1,:) - R2Off_R1Off_PSC_s(n-1,:) )/R2Off_R1Off_PSC_tauR
        R2Off_R1Off_PSC_x_k1 = -R2Off_R1Off_PSC_x(n-1,:)/R2Off_R1Off_PSC_tauD
        R2Off_R1Off_PSC_F_k1 = (1 - R2Off_R1Off_PSC_F(n-1,:))/R2Off_R1Off_PSC_tauF
        R2Off_R1Off_PSC_P_k1 = (1 - R2Off_R1Off_PSC_P(n-1,:))/R2Off_R1Off_PSC_tauP
        R2Off_R1Off_PSC_q_k1 = 0
        S2OnOff_R1Off_PSC_s_k1 = ( S2OnOff_R1Off_PSC_scale * S2OnOff_R1Off_PSC_x(n-1,:) - S2OnOff_R1Off_PSC_s(n-1,:) )/S2OnOff_R1Off_PSC_tauR
        S2OnOff_R1Off_PSC_x_k1 = -S2OnOff_R1Off_PSC_x(n-1,:)/S2OnOff_R1Off_PSC_tauD
        S2OnOff_R1Off_PSC_F_k1 = (1 - S2OnOff_R1Off_PSC_F(n-1,:))/S2OnOff_R1Off_PSC_tauF
        S2OnOff_R1Off_PSC_P_k1 = (1 - S2OnOff_R1Off_PSC_P(n-1,:))/S2OnOff_R1Off_PSC_tauP
        S2OnOff_R1Off_PSC_q_k1 = 0
        R2On_R2On_iNoise_V3_sn_k1 = ( R2On_R2On_iNoise_V3_scale * R2On_R2On_iNoise_V3_xn(n-1,:) - R2On_R2On_iNoise_V3_sn(n-1,:) )/R2On_R2On_iNoise_V3_tauR_N
        R2On_R2On_iNoise_V3_xn_k1 = -R2On_R2On_iNoise_V3_xn(n-1,:)/R2On_R2On_iNoise_V3_tauD_N + R2On_R2On_iNoise_V3_token(k,:)/R2On_R2On_iNoise_V3_dt
