
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
import scipy.io
import numpy as np

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
        self.R2On_R1On_PSC_gSYN = nn.Parameter(torch.tensor(0.01192328, dtype=torch.float32))
        self.S2OnOff_R1On_PSC_gSYN = nn.Parameter(torch.tensor(0.085, dtype=torch.float32))
        self.R2On_S2OnOff_PSC_gSYN = nn.Parameter(torch.tensor(0.0322174, dtype=torch.float32))
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



        for num_trials_count in range(10):


       
            
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

        return spike_holderR2On
    


def main():
    

if __name__ == "__main__":
    main()
