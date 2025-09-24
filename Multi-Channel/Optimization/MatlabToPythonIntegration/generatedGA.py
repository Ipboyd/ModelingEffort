import genPoissonTimes
import genPoissonInputs
import matplotlib.pyplot as plt
import gc
import scipy.io
import numpy as np
import Calc_output_grad
import Update_params
def main(trial_number,ps,scale_factor):

    #Params
    tspan = np.array([0.1, 2980.1*scale_factor])
    downsample_factor = 1


    disk_flag = 0
    dt = 0.1

    mex_flag = 0
    verbose_flag = 0
    On_C = 0.1
    On_g_L = 0.005
    On_E_L = -65
    On_noise = 0
    On_t_ref = 1
    On_E_k = -80
    On_tau_ad = 5
    On_g_inc = 0
    On_Itonic = 0
    On_V_thresh = -47
    On_V_reset = -54
    On_Npop = 1
    Off_C = 0.1
    Off_g_L = 0.005
    Off_E_L = -65
    Off_noise = 0
    Off_t_ref = 1
    Off_E_k = -80
    Off_tau_ad = 5
    Off_g_inc = 0
    Off_Itonic = 0
    Off_V_thresh = -47
    Off_V_reset = -54
    Off_Npop = 1
    R1On_C = 0.1
    R1On_g_L = 0.005
    R1On_E_L = -65
    R1On_noise = 0
    R1On_t_ref = 1
    R1On_E_k = -80
    R1On_tau_ad = 100
    R1On_g_inc = 0.0003
    R1On_Itonic = 0
    R1On_V_thresh = -47
    R1On_V_reset = -54
    R1On_Npop = 1
    R1Off_C = 0.1
    R1Off_g_L = 0.005
    R1Off_E_L = -65
    R1Off_noise = 0
    R1Off_t_ref = 1
    R1Off_E_k = -80
    R1Off_tau_ad = 100
    R1Off_g_inc = 0.0003
    R1Off_Itonic = 0
    R1Off_V_thresh = -47
    R1Off_V_reset = -54
    R1Off_Npop = 1
    S1OnOff_C = 0.1
    S1OnOff_g_L = 0.01
    S1OnOff_E_L = -57
    S1OnOff_noise = 0
    S1OnOff_t_ref = 0.5
    S1OnOff_E_k = -80
    S1OnOff_tau_ad = 5
    S1OnOff_g_inc = 0
    S1OnOff_Itonic = 0
    S1OnOff_V_thresh = -47
    S1OnOff_V_reset = -52
    S1OnOff_Npop = 1
    R2On_C = 0.1
    R2On_g_L = 0.005
    R2On_E_L = -65
    R2On_noise = 0
    R2On_t_ref = 1
    R2On_E_k = -80
    R2On_tau_ad = 100
    R2On_g_inc = 0.0003
    R2On_Itonic = 0
    R2On_V_thresh = -47
    R2On_V_reset = -54
    R2On_Npop = 1
    R2Off_C = 0.1
    R2Off_g_L = 0.005
    R2Off_E_L = -65
    R2Off_noise = 0
    R2Off_t_ref = 1
    R2Off_E_k = -80
    R2Off_tau_ad = 100
    R2Off_g_inc = 0.0003
    R2Off_Itonic = 0
    R2Off_V_thresh = -47
    R2Off_V_reset = -54
    R2Off_Npop = 1
    S2OnOff_C = 0.1
    S2OnOff_g_L = 0.01
    S2OnOff_E_L = -57
    S2OnOff_noise = 0
    S2OnOff_t_ref = 0.5
    S2OnOff_E_k = -80
    S2OnOff_tau_ad = 5
    S2OnOff_g_inc = 0
    S2OnOff_Itonic = 0
    S2OnOff_V_thresh = -47
    S2OnOff_V_reset = -52
    S2OnOff_Npop = 1
    On_On_IC_trial = 20
    On_On_IC_locNum = 15
    On_On_IC_label = 'on'
    On_On_IC_t_ref = 1
    On_On_IC_t_ref_rel = 1
    On_On_IC_rec = 2
    On_On_IC_g_postIC = 0.17
    On_On_IC_E_exc = 0
    Off_Off_IC_trial = 20
    Off_Off_IC_locNum = 15
    Off_Off_IC_label = 'off'
    Off_Off_IC_t_ref = 1
    Off_Off_IC_t_ref_rel = 1
    Off_Off_IC_rec = 2
    Off_Off_IC_g_postIC = 0.17
    Off_Off_IC_E_exc = 0
    R1On_On_PSC_ESYN = 0
    R1On_On_PSC_tauD = 1.5
    R1On_On_PSC_tauR = 0.7
    R1On_On_PSC_delay = 0
    R1On_On_PSC_gSYN = ps[0]
    R1On_On_PSC_fF = 0
    R1On_On_PSC_fP = 0.1
    R1On_On_PSC_tauF = 180
    R1On_On_PSC_tauP = 30
    R1On_On_PSC_maxF = 4
    S1OnOff_On_PSC_ESYN = 0
    S1OnOff_On_PSC_tauD = 1
    S1OnOff_On_PSC_tauR = 0.1
    S1OnOff_On_PSC_delay = 0
    S1OnOff_On_PSC_gSYN = ps[1]
    S1OnOff_On_PSC_fF = 0
    S1OnOff_On_PSC_fP = 0.2
    S1OnOff_On_PSC_tauF = 180
    S1OnOff_On_PSC_tauP = 80
    S1OnOff_On_PSC_maxF = 4
    R1On_S1OnOff_PSC_ESYN = -80
    R1On_S1OnOff_PSC_tauD = 4.5
    R1On_S1OnOff_PSC_tauR = 1
    R1On_S1OnOff_PSC_delay = 0
    R1On_S1OnOff_PSC_gSYN = ps[2]
    R1On_S1OnOff_PSC_fF = 0
    R1On_S1OnOff_PSC_fP = 0.5
    R1On_S1OnOff_PSC_tauF = 180
    R1On_S1OnOff_PSC_tauP = 120
    R1On_S1OnOff_PSC_maxF = 4
    R1Off_S1OnOff_PSC_ESYN = -80
    R1Off_S1OnOff_PSC_tauD = 4.5
    R1Off_S1OnOff_PSC_tauR = 1
    R1Off_S1OnOff_PSC_delay = 0
    R1Off_S1OnOff_PSC_gSYN = ps[3]
    R1Off_S1OnOff_PSC_fF = 0
    R1Off_S1OnOff_PSC_fP = 0.5
    R1Off_S1OnOff_PSC_tauF = 180
    R1Off_S1OnOff_PSC_tauP = 120
    R1Off_S1OnOff_PSC_maxF = 4
    R1Off_Off_PSC_ESYN = 0
    R1Off_Off_PSC_tauD = 1.5
    R1Off_Off_PSC_tauR = 0.7
    R1Off_Off_PSC_delay = 0
    R1Off_Off_PSC_gSYN = ps[4]
    R1Off_Off_PSC_fF = 0
    R1Off_Off_PSC_fP = 0.1
    R1Off_Off_PSC_tauF = 180
    R1Off_Off_PSC_tauP = 30
    R1Off_Off_PSC_maxF = 4
    S1OnOff_Off_PSC_ESYN = 0
    S1OnOff_Off_PSC_tauD = 1
    S1OnOff_Off_PSC_tauR = 0.1
    S1OnOff_Off_PSC_delay = 0
    S1OnOff_Off_PSC_gSYN = ps[5]
    S1OnOff_Off_PSC_fF = 0
    S1OnOff_Off_PSC_fP = 0
    S1OnOff_Off_PSC_tauF = 180
    S1OnOff_Off_PSC_tauP = 80
    S1OnOff_Off_PSC_maxF = 4
    R2On_R1On_PSC_ESYN = 0
    R2On_R1On_PSC_tauD = 1.5
    R2On_R1On_PSC_tauR = 0.7
    R2On_R1On_PSC_delay = 0
    R2On_R1On_PSC_gSYN = ps[6]
    R2On_R1On_PSC_fF = 0
    R2On_R1On_PSC_fP = 0.1
    R2On_R1On_PSC_tauF = 180
    R2On_R1On_PSC_tauP = 30
    R2On_R1On_PSC_maxF = 4
    S2OnOff_R1On_PSC_ESYN = 0
    S2OnOff_R1On_PSC_tauD = 1
    S2OnOff_R1On_PSC_tauR = 0.1
    S2OnOff_R1On_PSC_delay = 0
    S2OnOff_R1On_PSC_gSYN = ps[7]
    S2OnOff_R1On_PSC_fF = 0
    S2OnOff_R1On_PSC_fP = 0.2
    S2OnOff_R1On_PSC_tauF = 180
    S2OnOff_R1On_PSC_tauP = 80
    S2OnOff_R1On_PSC_maxF = 4
    R2On_S2OnOff_PSC_ESYN = -80
    R2On_S2OnOff_PSC_tauD = 4.5
    R2On_S2OnOff_PSC_tauR = 1
    R2On_S2OnOff_PSC_delay = 0
    R2On_S2OnOff_PSC_gSYN = ps[8]
    R2On_S2OnOff_PSC_fF = 0
    R2On_S2OnOff_PSC_fP = 0.5
    R2On_S2OnOff_PSC_tauF = 180
    R2On_S2OnOff_PSC_tauP = 120
    R2On_S2OnOff_PSC_maxF = 4
    R2Off_S2OnOff_PSC_ESYN = -80
    R2Off_S2OnOff_PSC_tauD = 4.5
    R2Off_S2OnOff_PSC_tauR = 1
    R2Off_S2OnOff_PSC_delay = 0
    R2Off_S2OnOff_PSC_gSYN = 0.025
    R2Off_S2OnOff_PSC_fF = 0
    R2Off_S2OnOff_PSC_fP = 0.5
    R2Off_S2OnOff_PSC_tauF = 180
    R2Off_S2OnOff_PSC_tauP = 120
    R2Off_S2OnOff_PSC_maxF = 4
    R2Off_R1Off_PSC_ESYN = 0
    R2Off_R1Off_PSC_tauD = 1.5
    R2Off_R1Off_PSC_tauR = 0.7
    R2Off_R1Off_PSC_delay = 0
    R2Off_R1Off_PSC_gSYN = 0.02
    R2Off_R1Off_PSC_fF = 0
    R2Off_R1Off_PSC_fP = 0.1
    R2Off_R1Off_PSC_tauF = 180
    R2Off_R1Off_PSC_tauP = 30
    R2Off_R1Off_PSC_maxF = 4
    S2OnOff_R1Off_PSC_ESYN = 0
    S2OnOff_R1Off_PSC_tauD = 1
    S2OnOff_R1Off_PSC_tauR = 0.1
    S2OnOff_R1Off_PSC_delay = 0
    S2OnOff_R1Off_PSC_gSYN = ps[9]
    S2OnOff_R1Off_PSC_fF = 0
    S2OnOff_R1Off_PSC_fP = 0
    S2OnOff_R1Off_PSC_tauF = 180
    S2OnOff_R1Off_PSC_tauP = 80
    S2OnOff_R1Off_PSC_maxF = 4
    R2On_R2On_iNoise_V3_FR = 8
    R2On_R2On_iNoise_V3_sigma = 0
    R2On_R2On_iNoise_V3_dt = 0.1
    R2On_R2On_iNoise_V3_nSYN = 0.015
    R2On_R2On_iNoise_V3_simlen = tspan[1]*10
    R2On_R2On_iNoise_V3_tauD_N = 1.5
    R2On_R2On_iNoise_V3_tauR_N = 0.7
    R2On_R2On_iNoise_V3_E_exc = 0
    ROn_X_PSC3_netcon = 1
    ROn_SOnOff_PSC3_netcon = 1
    C_ROn_PSC3_netcon = 1
    dGSYNR1On_On = np.zeros((200))
    dGSYNS1OnOff_On = np.zeros((200))
    dGSYNR1On_S1OnOff = np.zeros((200))
    dGSYNR1Off_S1OnOff = np.zeros((200))
    dGSYNR1Off_Off = np.zeros((200))
    dGSYNS1OnOff_Off = np.zeros((200))
    dGSYNR2On_R1On = np.zeros((200))
    dGSYNS2OnOff_R1On = np.zeros((200))
    dGSYNR2On_S2OnOff = np.zeros((200))
    dGSYNS2OnOff_R1Off = np.zeros((200))

    #Fixed Param Declaration
    On_R = 1/On_g_L
    On_tau = On_C*On_R
    On_Imask = np.ones((1,On_Npop))
    Off_R = 1/Off_g_L
    Off_tau = Off_C*Off_R
    Off_Imask = np.ones((1,Off_Npop))
    R1On_R = 1/R1On_g_L
    R1On_tau = R1On_C*R1On_R
    R1On_Imask = np.ones((1,R1On_Npop))
    R1Off_R = 1/R1Off_g_L
    R1Off_tau = R1Off_C*R1Off_R
    R1Off_Imask = np.ones((1,R1Off_Npop))
    S1OnOff_R = 1/S1OnOff_g_L
    S1OnOff_tau = S1OnOff_C*S1OnOff_R
    S1OnOff_Imask = np.ones((1,S1OnOff_Npop))
    R2On_R = 1/R2On_g_L
    R2On_tau = R2On_C*R2On_R
    R2On_Imask = np.ones((1,R2On_Npop))
    R2Off_R = 1/R2Off_g_L
    R2Off_tau = R2Off_C*R2Off_R
    R2Off_Imask = np.ones((1,R2Off_Npop))
    S2OnOff_R = 1/S2OnOff_g_L
    S2OnOff_tau = S2OnOff_C*S2OnOff_R
    S2OnOff_Imask = np.ones((1,S2OnOff_Npop))
    On_On_IC_netcon = +1.000000000000000e+00
    Off_Off_IC_netcon = +1.000000000000000e+00
    R1On_On_PSC_netcon = np.eye(On_Npop, R1On_Npop)
    R1On_On_PSC_scale = (R1On_On_PSC_tauD/R1On_On_PSC_tauR)**(R1On_On_PSC_tauR/(R1On_On_PSC_tauD-R1On_On_PSC_tauR))
    S1OnOff_On_PSC_netcon = np.eye(On_Npop, S1OnOff_Npop)
    S1OnOff_On_PSC_scale = (S1OnOff_On_PSC_tauD/S1OnOff_On_PSC_tauR)**(S1OnOff_On_PSC_tauR/(S1OnOff_On_PSC_tauD-S1OnOff_On_PSC_tauR))
    R1On_S1OnOff_PSC_netcon = np.eye(S1OnOff_Npop, R1On_Npop)
    R1On_S1OnOff_PSC_scale = (R1On_S1OnOff_PSC_tauD/R1On_S1OnOff_PSC_tauR)**(R1On_S1OnOff_PSC_tauR/(R1On_S1OnOff_PSC_tauD-R1On_S1OnOff_PSC_tauR))
    R1Off_S1OnOff_PSC_netcon = np.eye(S1OnOff_Npop, R1Off_Npop)
    R1Off_S1OnOff_PSC_scale = (R1Off_S1OnOff_PSC_tauD/R1Off_S1OnOff_PSC_tauR)**(R1Off_S1OnOff_PSC_tauR/(R1Off_S1OnOff_PSC_tauD-R1Off_S1OnOff_PSC_tauR))
    R1Off_Off_PSC_netcon = np.eye(Off_Npop, R1Off_Npop)
    R1Off_Off_PSC_scale = (R1Off_Off_PSC_tauD/R1Off_Off_PSC_tauR)**(R1Off_Off_PSC_tauR/(R1Off_Off_PSC_tauD-R1Off_Off_PSC_tauR))
    S1OnOff_Off_PSC_netcon = np.eye(Off_Npop, S1OnOff_Npop)
    S1OnOff_Off_PSC_scale = (S1OnOff_Off_PSC_tauD/S1OnOff_Off_PSC_tauR)**(S1OnOff_Off_PSC_tauR/(S1OnOff_Off_PSC_tauD-S1OnOff_Off_PSC_tauR))
    R2On_R1On_PSC_netcon = np.eye(R1On_Npop, R2On_Npop)
    R2On_R1On_PSC_scale = (R2On_R1On_PSC_tauD/R2On_R1On_PSC_tauR)**(R2On_R1On_PSC_tauR/(R2On_R1On_PSC_tauD-R2On_R1On_PSC_tauR))
    S2OnOff_R1On_PSC_netcon = np.eye(R1On_Npop, S2OnOff_Npop)
    S2OnOff_R1On_PSC_scale = (S2OnOff_R1On_PSC_tauD/S2OnOff_R1On_PSC_tauR)**(S2OnOff_R1On_PSC_tauR/(S2OnOff_R1On_PSC_tauD-S2OnOff_R1On_PSC_tauR))
    R2On_S2OnOff_PSC_netcon = np.eye(S2OnOff_Npop, R2On_Npop)
    R2On_S2OnOff_PSC_scale = (R2On_S2OnOff_PSC_tauD/R2On_S2OnOff_PSC_tauR)**(R2On_S2OnOff_PSC_tauR/(R2On_S2OnOff_PSC_tauD-R2On_S2OnOff_PSC_tauR))
    R2Off_S2OnOff_PSC_netcon = np.eye(S2OnOff_Npop, R2Off_Npop)
    R2Off_S2OnOff_PSC_scale = (R2Off_S2OnOff_PSC_tauD/R2Off_S2OnOff_PSC_tauR)**(R2Off_S2OnOff_PSC_tauR/(R2Off_S2OnOff_PSC_tauD-R2Off_S2OnOff_PSC_tauR))
    R2Off_R1Off_PSC_netcon = np.eye(R1Off_Npop, R2Off_Npop)
    R2Off_R1Off_PSC_scale = (R2Off_R1Off_PSC_tauD/R2Off_R1Off_PSC_tauR)**(R2Off_R1Off_PSC_tauR/(R2Off_R1Off_PSC_tauD-R2Off_R1Off_PSC_tauR))
    S2OnOff_R1Off_PSC_netcon = np.eye(R1Off_Npop, S2OnOff_Npop)
    S2OnOff_R1Off_PSC_scale = (S2OnOff_R1Off_PSC_tauD/S2OnOff_R1Off_PSC_tauR)**(S2OnOff_R1Off_PSC_tauR/(S2OnOff_R1Off_PSC_tauD-S2OnOff_R1Off_PSC_tauR))
    R2On_R2On_iNoise_V3_netcon = np.eye(R2On_Npop, R2On_Npop)
    R2On_R2On_iNoise_V3_token = genPoissonTimes.gen_poisson_times(R2On_Npop,R2On_R2On_iNoise_V3_dt,R2On_R2On_iNoise_V3_FR,R2On_R2On_iNoise_V3_sigma,R2On_R2On_iNoise_V3_simlen)
    R2On_R2On_iNoise_V3_scale = (R2On_R2On_iNoise_V3_tauD_N/R2On_R2On_iNoise_V3_tauR_N)**(R2On_R2On_iNoise_V3_tauR_N/(R2On_R2On_iNoise_V3_tauD_N-R2On_R2On_iNoise_V3_tauR_N))

    T = len(np.arange(tspan[0],tspan[1]+(dt),dt))
    helper = np.arange(tspan[0],tspan[1]+(dt),dt)

    #State Variable Declaration
    On_V = np.ones((200,2)) * [On_E_L, On_E_L]
    On_g_ad = np.ones((200,2)) * [0,0]
    Off_V = np.ones((200,2)) * [Off_E_L, Off_E_L]
    Off_g_ad = np.ones((200,2)) * [0,0]
    R1On_V = np.ones((200,2)) * [R1On_E_L, R1On_E_L]
    R1On_g_ad = np.ones((200,2)) * [0,0]
    R1Off_V = np.ones((200,2)) * [R1Off_E_L, R1Off_E_L]
    R1Off_g_ad = np.ones((200,2)) * [0,0]
    S1OnOff_V = np.ones((200,2)) * [S1OnOff_E_L, S1OnOff_E_L]
    S1OnOff_g_ad = np.ones((200,2)) * [0,0]
    R2On_V = np.ones((200,2)) * [R2On_E_L, R2On_E_L]
    R2On_g_ad = np.ones((200,2)) * [0,0]
    R2Off_V = np.ones((200,2)) * [R2Off_E_L, R2Off_E_L]
    R2Off_g_ad = np.ones((200,2)) * [0,0]
    S2OnOff_V = np.ones((200,2)) * [S2OnOff_E_L, S2OnOff_E_L]
    S2OnOff_g_ad = np.ones((200,2)) * [0,0]
    R1On_On_PSC_s = np.ones((200,2)) * [0,0]
    R1On_On_PSC_x = np.ones((200,2)) * [0,0]
    R1On_On_PSC_F = np.ones((200,2)) * [1,1]
    R1On_On_PSC_P = np.ones((200,2)) * [1,1]
    R1On_On_PSC_q = np.ones((200,2)) * [1,1]
    S1OnOff_On_PSC_s = np.ones((200,2)) * [0,0]
    S1OnOff_On_PSC_x = np.ones((200,2)) * [0,0]
    S1OnOff_On_PSC_F = np.ones((200,2)) * [1,1]
    S1OnOff_On_PSC_P = np.ones((200,2)) * [1,1]
    S1OnOff_On_PSC_q = np.ones((200,2)) * [1,1]
    R1On_S1OnOff_PSC_s = np.ones((200,2)) * [0,0]
    R1On_S1OnOff_PSC_x = np.ones((200,2)) * [0,0]
    R1On_S1OnOff_PSC_F = np.ones((200,2)) * [1,1]
    R1On_S1OnOff_PSC_P = np.ones((200,2)) * [1,1]
    R1On_S1OnOff_PSC_q = np.ones((200,2)) * [1,1]
    R1Off_S1OnOff_PSC_s = np.ones((200,2)) * [0,0]
    R1Off_S1OnOff_PSC_x = np.ones((200,2)) * [0,0]
    R1Off_S1OnOff_PSC_F = np.ones((200,2)) * [1,1]
    R1Off_S1OnOff_PSC_P = np.ones((200,2)) * [1,1]
    R1Off_S1OnOff_PSC_q = np.ones((200,2)) * [1,1]
    R1Off_Off_PSC_s = np.ones((200,2)) * [0,0]
    R1Off_Off_PSC_x = np.ones((200,2)) * [0,0]
    R1Off_Off_PSC_F = np.ones((200,2)) * [1,1]
    R1Off_Off_PSC_P = np.ones((200,2)) * [1,1]
    R1Off_Off_PSC_q = np.ones((200,2)) * [1,1]
    S1OnOff_Off_PSC_s = np.ones((200,2)) * [0,0]
    S1OnOff_Off_PSC_x = np.ones((200,2)) * [0,0]
    S1OnOff_Off_PSC_F = np.ones((200,2)) * [1,1]
    S1OnOff_Off_PSC_P = np.ones((200,2)) * [1,1]
    S1OnOff_Off_PSC_q = np.ones((200,2)) * [1,1]
    R2On_R1On_PSC_s = np.ones((200,2)) * [0,0]
    R2On_R1On_PSC_x = np.ones((200,2)) * [0,0]
    R2On_R1On_PSC_F = np.ones((200,2)) * [1,1]
    R2On_R1On_PSC_P = np.ones((200,2)) * [1,1]
    R2On_R1On_PSC_q = np.ones((200,2)) * [1,1]
    S2OnOff_R1On_PSC_s = np.ones((200,2)) * [0,0]
    S2OnOff_R1On_PSC_x = np.ones((200,2)) * [0,0]
    S2OnOff_R1On_PSC_F = np.ones((200,2)) * [1,1]
    S2OnOff_R1On_PSC_P = np.ones((200,2)) * [1,1]
    S2OnOff_R1On_PSC_q = np.ones((200,2)) * [1,1]
    R2On_S2OnOff_PSC_s = np.ones((200,2)) * [0,0]
    R2On_S2OnOff_PSC_x = np.ones((200,2)) * [0,0]
    R2On_S2OnOff_PSC_F = np.ones((200,2)) * [1,1]
    R2On_S2OnOff_PSC_P = np.ones((200,2)) * [1,1]
    R2On_S2OnOff_PSC_q = np.ones((200,2)) * [1,1]
    R2Off_S2OnOff_PSC_s = np.ones((200,2)) * [0,0]
    R2Off_S2OnOff_PSC_x = np.ones((200,2)) * [0,0]
    R2Off_S2OnOff_PSC_F = np.ones((200,2)) * [1,1]
    R2Off_S2OnOff_PSC_P = np.ones((200,2)) * [1,1]
    R2Off_S2OnOff_PSC_q = np.ones((200,2)) * [1,1]
    R2Off_R1Off_PSC_s = np.ones((200,2)) * [0,0]
    R2Off_R1Off_PSC_x = np.ones((200,2)) * [0,0]
    R2Off_R1Off_PSC_F = np.ones((200,2)) * [1,1]
    R2Off_R1Off_PSC_P = np.ones((200,2)) * [1,1]
    R2Off_R1Off_PSC_q = np.ones((200,2)) * [1,1]
    S2OnOff_R1Off_PSC_s = np.ones((200,2)) * [0,0]
    S2OnOff_R1Off_PSC_x = np.ones((200,2)) * [0,0]
    S2OnOff_R1Off_PSC_F = np.ones((200,2)) * [1,1]
    S2OnOff_R1Off_PSC_P = np.ones((200,2)) * [1,1]
    S2OnOff_R1Off_PSC_q = np.ones((200,2)) * [1,1]
    R2On_R2On_iNoise_V3_sn = np.ones((200,2)) * [0, 0]
    R2On_R2On_iNoise_V3_xn = np.ones((200,2)) * [0, 0]

    #Monitor Declaration
    On_tspike = -1e32*np.ones((200, 5, On_Npop))
    On_buffer_index = np.ones((200))
    On_V_spikes_holder = []
    Off_tspike = -1e32*np.ones((200, 5, Off_Npop))
    Off_buffer_index = np.ones((200))
    Off_V_spikes_holder = []
    R1On_tspike = -1e32*np.ones((200, 5, R1On_Npop))
    R1On_buffer_index = np.ones((200))
    R1On_V_spikes_holder = []
    R1Off_tspike = -1e32*np.ones((200, 5, R1Off_Npop))
    R1Off_buffer_index = np.ones((200))
    R1Off_V_spikes_holder = []
    S1OnOff_tspike = -1e32*np.ones((200, 5, S1OnOff_Npop))
    S1OnOff_buffer_index = np.ones((200))
    S1OnOff_V_spikes_holder = []
    R2On_tspike = -1e32*np.ones((200, 5, R2On_Npop))
    R2On_buffer_index = np.ones((200))
    R2On_V_spikes_holder = []
    R2Off_tspike = -1e32*np.ones((200, 5, R2Off_Npop))
    R2Off_buffer_index = np.ones((200))
    R2Off_V_spikes_holder = []
    S2OnOff_tspike = -1e32*np.ones((200, 5, S2OnOff_Npop))
    S2OnOff_buffer_index = np.ones((200))
    S2OnOff_V_spikes_holder = []
    On_On_IC_iIC = 0
    Off_Off_IC_iIC = 0
    R1On_On_PSC_syn = 0
    S1OnOff_On_PSC_syn = 0
    R1On_S1OnOff_PSC_syn = 0
    R1Off_S1OnOff_PSC_syn = 0
    R1Off_Off_PSC_syn = 0
    S1OnOff_Off_PSC_syn = 0
    R2On_R1On_PSC_syn = 0
    S2OnOff_R1On_PSC_syn = 0
    R2On_S2OnOff_PSC_syn = 0
    R2Off_S2OnOff_PSC_syn = 0
    R2Off_R1Off_PSC_syn = 0
    S2OnOff_R1Off_PSC_syn = 0

    #Delcare Inputs
    On_On_IC_input = genPoissonInputs.gen_poisson_inputs(trial_number,On_On_IC_locNum,On_On_IC_label,On_On_IC_t_ref,On_On_IC_t_ref_rel,On_On_IC_rec,scale_factor)
    Off_Off_IC_input = genPoissonInputs.gen_poisson_inputs(trial_number,Off_Off_IC_locNum,Off_Off_IC_label,Off_Off_IC_t_ref,Off_Off_IC_t_ref_rel,Off_Off_IC_rec,scale_factor)

    for t in range(0,T):

        #ODEs
        On_V_k1 = ( (On_E_L-On_V[:,-1]) - On_R*On_g_ad[:,-1]*(On_V[:,-1]-On_E_k) - On_R*((((On_On_IC_g_postIC*(On_On_IC_input[t]*On_On_IC_netcon)*(On_V[:,-1]-On_On_IC_E_exc))))) + On_R*On_Itonic*On_Imask  ) / On_tau
        On_g_ad_k1 = -On_g_ad[:,-1] / On_tau_ad
        Off_V_k1 = ( (Off_E_L-Off_V[:,-1]) - Off_R*Off_g_ad[:,-1]*(Off_V[:,-1]-Off_E_k) - Off_R*((((Off_Off_IC_g_postIC*(Off_Off_IC_input[t]*Off_Off_IC_netcon)*(Off_V[:,-1]-Off_Off_IC_E_exc))))) + Off_R*Off_Itonic*Off_Imask  ) / Off_tau
        Off_g_ad_k1 = -Off_g_ad[:,-1] / Off_tau_ad
        R1On_V_k1 = ( (R1On_E_L-R1On_V[:,-1]) - R1On_R*R1On_g_ad[:,-1]*(R1On_V[:,-1]-R1On_E_k) - R1On_R*((((R1On_On_PSC_gSYN*(R1On_On_PSC_s[:,-1]*R1On_On_PSC_netcon)*(R1On_V[:,-1]-R1On_On_PSC_ESYN))))+((((R1On_S1OnOff_PSC_gSYN*(R1On_S1OnOff_PSC_s[:,-1]*R1On_S1OnOff_PSC_netcon)*(R1On_V[:,-1]-R1On_S1OnOff_PSC_ESYN)))))) + R1On_R*R1On_Itonic*R1On_Imask  ) / R1On_tau
        R1On_g_ad_k1 = -R1On_g_ad[:,-1] / R1On_tau_ad
        R1Off_V_k1 = ( (R1Off_E_L-R1Off_V[:,-1]) - R1Off_R*R1Off_g_ad[:,-1]*(R1Off_V[:,-1]-R1Off_E_k) - R1Off_R*((((R1Off_S1OnOff_PSC_gSYN*(R1Off_S1OnOff_PSC_s[:,-1]*R1Off_S1OnOff_PSC_netcon)*(R1Off_V[:,-1]-R1Off_S1OnOff_PSC_ESYN))))+((((R1Off_Off_PSC_gSYN*(R1Off_Off_PSC_s[:,-1]*R1Off_Off_PSC_netcon)*(R1Off_V[:,-1]-R1Off_Off_PSC_ESYN)))))) + R1Off_R*R1Off_Itonic*R1Off_Imask  ) / R1Off_tau
        R1Off_g_ad_k1 = -R1Off_g_ad[:,-1] / R1Off_tau_ad
        S1OnOff_V_k1 = ( (S1OnOff_E_L-S1OnOff_V[:,-1]) - S1OnOff_R*S1OnOff_g_ad[:,-1]*(S1OnOff_V[:,-1]-S1OnOff_E_k) - S1OnOff_R*((((S1OnOff_On_PSC_gSYN*(S1OnOff_On_PSC_s[:,-1]*S1OnOff_On_PSC_netcon)*(S1OnOff_V[:,-1]-S1OnOff_On_PSC_ESYN))))+((((S1OnOff_Off_PSC_gSYN*(S1OnOff_Off_PSC_s[:,-1]*S1OnOff_Off_PSC_netcon)*(S1OnOff_V[:,-1]-S1OnOff_Off_PSC_ESYN)))))) + S1OnOff_R*S1OnOff_Itonic*S1OnOff_Imask  ) / S1OnOff_tau
        S1OnOff_g_ad_k1 = -S1OnOff_g_ad[:,-1] / S1OnOff_tau_ad
        R2On_V_k1 = ( (R2On_E_L-R2On_V[:,-1]) - R2On_R*R2On_g_ad[:,-1]*(R2On_V[:,-1]-R2On_E_k) - R2On_R*((((R2On_R1On_PSC_gSYN*(R2On_R1On_PSC_s[:,-1]*R2On_R1On_PSC_netcon)*(R2On_V[:,-1]-R2On_R1On_PSC_ESYN))))+((((R2On_S2OnOff_PSC_gSYN*(R2On_S2OnOff_PSC_s[:,-1]*R2On_S2OnOff_PSC_netcon)*(R2On_V[:,-1]-R2On_S2OnOff_PSC_ESYN))))+((((R2On_R2On_iNoise_V3_nSYN*(R2On_R2On_iNoise_V3_sn[:,-1]*R2On_R2On_iNoise_V3_netcon)*(R2On_V[:,-1]-R2On_R2On_iNoise_V3_E_exc))))))) + R2On_R*R2On_Itonic*R2On_Imask  ) / R2On_tau
        R2On_g_ad_k1 = -R2On_g_ad[:,-1] / R2On_tau_ad
        R2Off_V_k1 = ( (R2Off_E_L-R2Off_V[:,-1]) - R2Off_R*R2Off_g_ad[:,-1]*(R2Off_V[:,-1]-R2Off_E_k) - R2Off_R*((((R2Off_S2OnOff_PSC_gSYN*(R2Off_S2OnOff_PSC_s[:,-1]*R2Off_S2OnOff_PSC_netcon)*(R2Off_V[:,-1]-R2Off_S2OnOff_PSC_ESYN))))+((((R2Off_R1Off_PSC_gSYN*(R2Off_R1Off_PSC_s[:,-1]*R2Off_R1Off_PSC_netcon)*(R2Off_V[:,-1]-R2Off_R1Off_PSC_ESYN)))))) + R2Off_R*R2Off_Itonic*R2Off_Imask  ) / R2Off_tau
        R2Off_g_ad_k1 = -R2Off_g_ad[:,-1] / R2Off_tau_ad
        S2OnOff_V_k1 = ( (S2OnOff_E_L-S2OnOff_V[:,-1]) - S2OnOff_R*S2OnOff_g_ad[:,-1]*(S2OnOff_V[:,-1]-S2OnOff_E_k) - S2OnOff_R*((((S2OnOff_R1On_PSC_gSYN*(S2OnOff_R1On_PSC_s[:,-1]*S2OnOff_R1On_PSC_netcon)*(S2OnOff_V[:,-1]-S2OnOff_R1On_PSC_ESYN))))+((((S2OnOff_R1Off_PSC_gSYN*(S2OnOff_R1Off_PSC_s[:,-1]*S2OnOff_R1Off_PSC_netcon)*(S2OnOff_V[:,-1]-S2OnOff_R1Off_PSC_ESYN)))))) + S2OnOff_R*S2OnOff_Itonic*S2OnOff_Imask  ) / S2OnOff_tau
        S2OnOff_g_ad_k1 = -S2OnOff_g_ad[:,-1] / S2OnOff_tau_ad
        R1On_On_PSC_s_k1 = ( R1On_On_PSC_scale * R1On_On_PSC_x[:,-1] - R1On_On_PSC_s[:,-1] )/R1On_On_PSC_tauR
        R1On_On_PSC_x_k1 = -R1On_On_PSC_x[:,-1]/R1On_On_PSC_tauD
        R1On_On_PSC_F_k1 = (1 - R1On_On_PSC_F[:,-1])/R1On_On_PSC_tauF
        R1On_On_PSC_P_k1 = (1 - R1On_On_PSC_P[:,-1])/R1On_On_PSC_tauP
        R1On_On_PSC_q_k1 = 0
        S1OnOff_On_PSC_s_k1 = ( S1OnOff_On_PSC_scale * S1OnOff_On_PSC_x[:,-1] - S1OnOff_On_PSC_s[:,-1] )/S1OnOff_On_PSC_tauR
        S1OnOff_On_PSC_x_k1 = -S1OnOff_On_PSC_x[:,-1]/S1OnOff_On_PSC_tauD
        S1OnOff_On_PSC_F_k1 = (1 - S1OnOff_On_PSC_F[:,-1])/S1OnOff_On_PSC_tauF
        S1OnOff_On_PSC_P_k1 = (1 - S1OnOff_On_PSC_P[:,-1])/S1OnOff_On_PSC_tauP
        S1OnOff_On_PSC_q_k1 = 0
        R1On_S1OnOff_PSC_s_k1 = ( R1On_S1OnOff_PSC_scale * R1On_S1OnOff_PSC_x[:,-1] - R1On_S1OnOff_PSC_s[:,-1] )/R1On_S1OnOff_PSC_tauR
        R1On_S1OnOff_PSC_x_k1 = -R1On_S1OnOff_PSC_x[:,-1]/R1On_S1OnOff_PSC_tauD
        R1On_S1OnOff_PSC_F_k1 = (1 - R1On_S1OnOff_PSC_F[:,-1])/R1On_S1OnOff_PSC_tauF
        R1On_S1OnOff_PSC_P_k1 = (1 - R1On_S1OnOff_PSC_P[:,-1])/R1On_S1OnOff_PSC_tauP
        R1On_S1OnOff_PSC_q_k1 = 0
        R1Off_S1OnOff_PSC_s_k1 = ( R1Off_S1OnOff_PSC_scale * R1Off_S1OnOff_PSC_x[:,-1] - R1Off_S1OnOff_PSC_s[:,-1] )/R1Off_S1OnOff_PSC_tauR
        R1Off_S1OnOff_PSC_x_k1 = -R1Off_S1OnOff_PSC_x[:,-1]/R1Off_S1OnOff_PSC_tauD
        R1Off_S1OnOff_PSC_F_k1 = (1 - R1Off_S1OnOff_PSC_F[:,-1])/R1Off_S1OnOff_PSC_tauF
        R1Off_S1OnOff_PSC_P_k1 = (1 - R1Off_S1OnOff_PSC_P[:,-1])/R1Off_S1OnOff_PSC_tauP
        R1Off_S1OnOff_PSC_q_k1 = 0
        R1Off_Off_PSC_s_k1 = ( R1Off_Off_PSC_scale * R1Off_Off_PSC_x[:,-1] - R1Off_Off_PSC_s[:,-1] )/R1Off_Off_PSC_tauR
        R1Off_Off_PSC_x_k1 = -R1Off_Off_PSC_x[:,-1]/R1Off_Off_PSC_tauD
        R1Off_Off_PSC_F_k1 = (1 - R1Off_Off_PSC_F[:,-1])/R1Off_Off_PSC_tauF
        R1Off_Off_PSC_P_k1 = (1 - R1Off_Off_PSC_P[:,-1])/R1Off_Off_PSC_tauP
        R1Off_Off_PSC_q_k1 = 0
        S1OnOff_Off_PSC_s_k1 = ( S1OnOff_Off_PSC_scale * S1OnOff_Off_PSC_x[:,-1] - S1OnOff_Off_PSC_s[:,-1] )/S1OnOff_Off_PSC_tauR
        S1OnOff_Off_PSC_x_k1 = -S1OnOff_Off_PSC_x[:,-1]/S1OnOff_Off_PSC_tauD
        S1OnOff_Off_PSC_F_k1 = (1 - S1OnOff_Off_PSC_F[:,-1])/S1OnOff_Off_PSC_tauF
        S1OnOff_Off_PSC_P_k1 = (1 - S1OnOff_Off_PSC_P[:,-1])/S1OnOff_Off_PSC_tauP
        S1OnOff_Off_PSC_q_k1 = 0
        R2On_R1On_PSC_s_k1 = ( R2On_R1On_PSC_scale * R2On_R1On_PSC_x[:,-1] - R2On_R1On_PSC_s[:,-1] )/R2On_R1On_PSC_tauR
        R2On_R1On_PSC_x_k1 = -R2On_R1On_PSC_x[:,-1]/R2On_R1On_PSC_tauD
        R2On_R1On_PSC_F_k1 = (1 - R2On_R1On_PSC_F[:,-1])/R2On_R1On_PSC_tauF
        R2On_R1On_PSC_P_k1 = (1 - R2On_R1On_PSC_P[:,-1])/R2On_R1On_PSC_tauP
        R2On_R1On_PSC_q_k1 = 0
        S2OnOff_R1On_PSC_s_k1 = ( S2OnOff_R1On_PSC_scale * S2OnOff_R1On_PSC_x[:,-1] - S2OnOff_R1On_PSC_s[:,-1] )/S2OnOff_R1On_PSC_tauR
        S2OnOff_R1On_PSC_x_k1 = -S2OnOff_R1On_PSC_x[:,-1]/S2OnOff_R1On_PSC_tauD
        S2OnOff_R1On_PSC_F_k1 = (1 - S2OnOff_R1On_PSC_F[:,-1])/S2OnOff_R1On_PSC_tauF
        S2OnOff_R1On_PSC_P_k1 = (1 - S2OnOff_R1On_PSC_P[:,-1])/S2OnOff_R1On_PSC_tauP
        S2OnOff_R1On_PSC_q_k1 = 0
        R2On_S2OnOff_PSC_s_k1 = ( R2On_S2OnOff_PSC_scale * R2On_S2OnOff_PSC_x[:,-1] - R2On_S2OnOff_PSC_s[:,-1] )/R2On_S2OnOff_PSC_tauR
        R2On_S2OnOff_PSC_x_k1 = -R2On_S2OnOff_PSC_x[:,-1]/R2On_S2OnOff_PSC_tauD
        R2On_S2OnOff_PSC_F_k1 = (1 - R2On_S2OnOff_PSC_F[:,-1])/R2On_S2OnOff_PSC_tauF
        R2On_S2OnOff_PSC_P_k1 = (1 - R2On_S2OnOff_PSC_P[:,-1])/R2On_S2OnOff_PSC_tauP
        R2On_S2OnOff_PSC_q_k1 = 0
        R2Off_S2OnOff_PSC_s_k1 = ( R2Off_S2OnOff_PSC_scale * R2Off_S2OnOff_PSC_x[:,-1] - R2Off_S2OnOff_PSC_s[:,-1] )/R2Off_S2OnOff_PSC_tauR
        R2Off_S2OnOff_PSC_x_k1 = -R2Off_S2OnOff_PSC_x[:,-1]/R2Off_S2OnOff_PSC_tauD
        R2Off_S2OnOff_PSC_F_k1 = (1 - R2Off_S2OnOff_PSC_F[:,-1])/R2Off_S2OnOff_PSC_tauF
        R2Off_S2OnOff_PSC_P_k1 = (1 - R2Off_S2OnOff_PSC_P[:,-1])/R2Off_S2OnOff_PSC_tauP
        R2Off_S2OnOff_PSC_q_k1 = 0
        R2Off_R1Off_PSC_s_k1 = ( R2Off_R1Off_PSC_scale * R2Off_R1Off_PSC_x[:,-1] - R2Off_R1Off_PSC_s[:,-1] )/R2Off_R1Off_PSC_tauR
        R2Off_R1Off_PSC_x_k1 = -R2Off_R1Off_PSC_x[:,-1]/R2Off_R1Off_PSC_tauD
        R2Off_R1Off_PSC_F_k1 = (1 - R2Off_R1Off_PSC_F[:,-1])/R2Off_R1Off_PSC_tauF
        R2Off_R1Off_PSC_P_k1 = (1 - R2Off_R1Off_PSC_P[:,-1])/R2Off_R1Off_PSC_tauP
        R2Off_R1Off_PSC_q_k1 = 0
        S2OnOff_R1Off_PSC_s_k1 = ( S2OnOff_R1Off_PSC_scale * S2OnOff_R1Off_PSC_x[:,-1] - S2OnOff_R1Off_PSC_s[:,-1] )/S2OnOff_R1Off_PSC_tauR
        S2OnOff_R1Off_PSC_x_k1 = -S2OnOff_R1Off_PSC_x[:,-1]/S2OnOff_R1Off_PSC_tauD
        S2OnOff_R1Off_PSC_F_k1 = (1 - S2OnOff_R1Off_PSC_F[:,-1])/S2OnOff_R1Off_PSC_tauF
        S2OnOff_R1Off_PSC_P_k1 = (1 - S2OnOff_R1Off_PSC_P[:,-1])/S2OnOff_R1Off_PSC_tauP
        S2OnOff_R1Off_PSC_q_k1 = 0
        R2On_R2On_iNoise_V3_sn_k1 = ( R2On_R2On_iNoise_V3_scale * R2On_R2On_iNoise_V3_xn[:,-1] - R2On_R2On_iNoise_V3_sn[:,-1] )/R2On_R2On_iNoise_V3_tauR_N
        R2On_R2On_iNoise_V3_xn_k1 = -R2On_R2On_iNoise_V3_xn[:,-1]/R2On_R2On_iNoise_V3_tauD_N + R2On_R2On_iNoise_V3_token[t]/R2On_R2On_iNoise_V3_dt

        #Update Eulers
        On_V[:,-2] = On_V[:,-1]
        On_V[:,-1] = On_V[:,-1]+dt*On_V_k1
        On_g_ad[:,-2] = On_g_ad[:,-1]
        On_g_ad[:,-1] = On_g_ad[:,-1]+dt*On_g_ad_k1
        Off_V[:,-2] = Off_V[:,-1]
        Off_V[:,-1] = Off_V[:,-1]+dt*Off_V_k1
        Off_g_ad[:,-2] = Off_g_ad[:,-1]
        Off_g_ad[:,-1] = Off_g_ad[:,-1]+dt*Off_g_ad_k1
        R1On_V[:,-2] = R1On_V[:,-1]
        R1On_V[:,-1] = R1On_V[:,-1]+dt*R1On_V_k1
        R1On_g_ad[:,-2] = R1On_g_ad[:,-1]
        R1On_g_ad[:,-1] = R1On_g_ad[:,-1]+dt*R1On_g_ad_k1
        R1Off_V[:,-2] = R1Off_V[:,-1]
        R1Off_V[:,-1] = R1Off_V[:,-1]+dt*R1Off_V_k1
        R1Off_g_ad[:,-2] = R1Off_g_ad[:,-1]
        R1Off_g_ad[:,-1] = R1Off_g_ad[:,-1]+dt*R1Off_g_ad_k1
        S1OnOff_V[:,-2] = S1OnOff_V[:,-1]
        S1OnOff_V[:,-1] = S1OnOff_V[:,-1]+dt*S1OnOff_V_k1
        S1OnOff_g_ad[:,-2] = S1OnOff_g_ad[:,-1]
        S1OnOff_g_ad[:,-1] = S1OnOff_g_ad[:,-1]+dt*S1OnOff_g_ad_k1
        R2On_V[:,-2] = R2On_V[:,-1]
        R2On_V[:,-1] = R2On_V[:,-1]+dt*R2On_V_k1
        R2On_g_ad[:,-2] = R2On_g_ad[:,-1]
        R2On_g_ad[:,-1] = R2On_g_ad[:,-1]+dt*R2On_g_ad_k1
        R2Off_V[:,-2] = R2Off_V[:,-1]
        R2Off_V[:,-1] = R2Off_V[:,-1]+dt*R2Off_V_k1
        R2Off_g_ad[:,-2] = R2Off_g_ad[:,-1]
        R2Off_g_ad[:,-1] = R2Off_g_ad[:,-1]+dt*R2Off_g_ad_k1
        S2OnOff_V[:,-2] = S2OnOff_V[:,-1]
        S2OnOff_V[:,-1] = S2OnOff_V[:,-1]+dt*S2OnOff_V_k1
        S2OnOff_g_ad[:,-2] = S2OnOff_g_ad[:,-1]
        S2OnOff_g_ad[:,-1] = S2OnOff_g_ad[:,-1]+dt*S2OnOff_g_ad_k1
        R1On_On_PSC_s[:,-2] = R1On_On_PSC_s[:,-1]
        R1On_On_PSC_s[:,-1] = R1On_On_PSC_s[:,-1]+dt*R1On_On_PSC_s_k1
        R1On_On_PSC_x[:,-2] = R1On_On_PSC_x[:,-1]
        R1On_On_PSC_x[:,-1] = R1On_On_PSC_x[:,-1]+dt*R1On_On_PSC_x_k1
        R1On_On_PSC_F[:,-2] = R1On_On_PSC_F[:,-1]
        R1On_On_PSC_F[:,-1] = R1On_On_PSC_F[:,-1]+dt*R1On_On_PSC_F_k1
        R1On_On_PSC_P[:,-2] = R1On_On_PSC_P[:,-1]
        R1On_On_PSC_P[:,-1] = R1On_On_PSC_P[:,-1]+dt*R1On_On_PSC_P_k1
        R1On_On_PSC_q[:,-2] = R1On_On_PSC_q[:,-1]
        R1On_On_PSC_q[:,-1] = R1On_On_PSC_q[:,-1]+dt*R1On_On_PSC_q_k1
        S1OnOff_On_PSC_s[:,-2] = S1OnOff_On_PSC_s[:,-1]
        S1OnOff_On_PSC_s[:,-1] = S1OnOff_On_PSC_s[:,-1]+dt*S1OnOff_On_PSC_s_k1
        S1OnOff_On_PSC_x[:,-2] = S1OnOff_On_PSC_x[:,-1]
        S1OnOff_On_PSC_x[:,-1] = S1OnOff_On_PSC_x[:,-1]+dt*S1OnOff_On_PSC_x_k1
        S1OnOff_On_PSC_F[:,-2] = S1OnOff_On_PSC_F[:,-1]
        S1OnOff_On_PSC_F[:,-1] = S1OnOff_On_PSC_F[:,-1]+dt*S1OnOff_On_PSC_F_k1
        S1OnOff_On_PSC_P[:,-2] = S1OnOff_On_PSC_P[:,-1]
        S1OnOff_On_PSC_P[:,-1] = S1OnOff_On_PSC_P[:,-1]+dt*S1OnOff_On_PSC_P_k1
        S1OnOff_On_PSC_q[:,-2] = S1OnOff_On_PSC_q[:,-1]
        S1OnOff_On_PSC_q[:,-1] = S1OnOff_On_PSC_q[:,-1]+dt*S1OnOff_On_PSC_q_k1
        R1On_S1OnOff_PSC_s[:,-2] = R1On_S1OnOff_PSC_s[:,-1]
        R1On_S1OnOff_PSC_s[:,-1] = R1On_S1OnOff_PSC_s[:,-1]+dt*R1On_S1OnOff_PSC_s_k1
        R1On_S1OnOff_PSC_x[:,-2] = R1On_S1OnOff_PSC_x[:,-1]
        R1On_S1OnOff_PSC_x[:,-1] = R1On_S1OnOff_PSC_x[:,-1]+dt*R1On_S1OnOff_PSC_x_k1
        R1On_S1OnOff_PSC_F[:,-2] = R1On_S1OnOff_PSC_F[:,-1]
        R1On_S1OnOff_PSC_F[:,-1] = R1On_S1OnOff_PSC_F[:,-1]+dt*R1On_S1OnOff_PSC_F_k1
        R1On_S1OnOff_PSC_P[:,-2] = R1On_S1OnOff_PSC_P[:,-1]
        R1On_S1OnOff_PSC_P[:,-1] = R1On_S1OnOff_PSC_P[:,-1]+dt*R1On_S1OnOff_PSC_P_k1
        R1On_S1OnOff_PSC_q[:,-2] = R1On_S1OnOff_PSC_q[:,-1]
        R1On_S1OnOff_PSC_q[:,-1] = R1On_S1OnOff_PSC_q[:,-1]+dt*R1On_S1OnOff_PSC_q_k1
        R1Off_S1OnOff_PSC_s[:,-2] = R1Off_S1OnOff_PSC_s[:,-1]
        R1Off_S1OnOff_PSC_s[:,-1] = R1Off_S1OnOff_PSC_s[:,-1]+dt*R1Off_S1OnOff_PSC_s_k1
        R1Off_S1OnOff_PSC_x[:,-2] = R1Off_S1OnOff_PSC_x[:,-1]
        R1Off_S1OnOff_PSC_x[:,-1] = R1Off_S1OnOff_PSC_x[:,-1]+dt*R1Off_S1OnOff_PSC_x_k1
        R1Off_S1OnOff_PSC_F[:,-2] = R1Off_S1OnOff_PSC_F[:,-1]
        R1Off_S1OnOff_PSC_F[:,-1] = R1Off_S1OnOff_PSC_F[:,-1]+dt*R1Off_S1OnOff_PSC_F_k1
        R1Off_S1OnOff_PSC_P[:,-2] = R1Off_S1OnOff_PSC_P[:,-1]
        R1Off_S1OnOff_PSC_P[:,-1] = R1Off_S1OnOff_PSC_P[:,-1]+dt*R1Off_S1OnOff_PSC_P_k1
        R1Off_S1OnOff_PSC_q[:,-2] = R1Off_S1OnOff_PSC_q[:,-1]
        R1Off_S1OnOff_PSC_q[:,-1] = R1Off_S1OnOff_PSC_q[:,-1]+dt*R1Off_S1OnOff_PSC_q_k1
        R1Off_Off_PSC_s[:,-2] = R1Off_Off_PSC_s[:,-1]
        R1Off_Off_PSC_s[:,-1] = R1Off_Off_PSC_s[:,-1]+dt*R1Off_Off_PSC_s_k1
        R1Off_Off_PSC_x[:,-2] = R1Off_Off_PSC_x[:,-1]
        R1Off_Off_PSC_x[:,-1] = R1Off_Off_PSC_x[:,-1]+dt*R1Off_Off_PSC_x_k1
        R1Off_Off_PSC_F[:,-2] = R1Off_Off_PSC_F[:,-1]
        R1Off_Off_PSC_F[:,-1] = R1Off_Off_PSC_F[:,-1]+dt*R1Off_Off_PSC_F_k1
        R1Off_Off_PSC_P[:,-2] = R1Off_Off_PSC_P[:,-1]
        R1Off_Off_PSC_P[:,-1] = R1Off_Off_PSC_P[:,-1]+dt*R1Off_Off_PSC_P_k1
        R1Off_Off_PSC_q[:,-2] = R1Off_Off_PSC_q[:,-1]
        R1Off_Off_PSC_q[:,-1] = R1Off_Off_PSC_q[:,-1]+dt*R1Off_Off_PSC_q_k1
        S1OnOff_Off_PSC_s[:,-2] = S1OnOff_Off_PSC_s[:,-1]
        S1OnOff_Off_PSC_s[:,-1] = S1OnOff_Off_PSC_s[:,-1]+dt*S1OnOff_Off_PSC_s_k1
        S1OnOff_Off_PSC_x[:,-2] = S1OnOff_Off_PSC_x[:,-1]
        S1OnOff_Off_PSC_x[:,-1] = S1OnOff_Off_PSC_x[:,-1]+dt*S1OnOff_Off_PSC_x_k1
        S1OnOff_Off_PSC_F[:,-2] = S1OnOff_Off_PSC_F[:,-1]
        S1OnOff_Off_PSC_F[:,-1] = S1OnOff_Off_PSC_F[:,-1]+dt*S1OnOff_Off_PSC_F_k1
        S1OnOff_Off_PSC_P[:,-2] = S1OnOff_Off_PSC_P[:,-1]
        S1OnOff_Off_PSC_P[:,-1] = S1OnOff_Off_PSC_P[:,-1]+dt*S1OnOff_Off_PSC_P_k1
        S1OnOff_Off_PSC_q[:,-2] = S1OnOff_Off_PSC_q[:,-1]
        S1OnOff_Off_PSC_q[:,-1] = S1OnOff_Off_PSC_q[:,-1]+dt*S1OnOff_Off_PSC_q_k1
        R2On_R1On_PSC_s[:,-2] = R2On_R1On_PSC_s[:,-1]
        R2On_R1On_PSC_s[:,-1] = R2On_R1On_PSC_s[:,-1]+dt*R2On_R1On_PSC_s_k1
        R2On_R1On_PSC_x[:,-2] = R2On_R1On_PSC_x[:,-1]
        R2On_R1On_PSC_x[:,-1] = R2On_R1On_PSC_x[:,-1]+dt*R2On_R1On_PSC_x_k1
        R2On_R1On_PSC_F[:,-2] = R2On_R1On_PSC_F[:,-1]
        R2On_R1On_PSC_F[:,-1] = R2On_R1On_PSC_F[:,-1]+dt*R2On_R1On_PSC_F_k1
        R2On_R1On_PSC_P[:,-2] = R2On_R1On_PSC_P[:,-1]
        R2On_R1On_PSC_P[:,-1] = R2On_R1On_PSC_P[:,-1]+dt*R2On_R1On_PSC_P_k1
        R2On_R1On_PSC_q[:,-2] = R2On_R1On_PSC_q[:,-1]
        R2On_R1On_PSC_q[:,-1] = R2On_R1On_PSC_q[:,-1]+dt*R2On_R1On_PSC_q_k1
        S2OnOff_R1On_PSC_s[:,-2] = S2OnOff_R1On_PSC_s[:,-1]
        S2OnOff_R1On_PSC_s[:,-1] = S2OnOff_R1On_PSC_s[:,-1]+dt*S2OnOff_R1On_PSC_s_k1
        S2OnOff_R1On_PSC_x[:,-2] = S2OnOff_R1On_PSC_x[:,-1]
        S2OnOff_R1On_PSC_x[:,-1] = S2OnOff_R1On_PSC_x[:,-1]+dt*S2OnOff_R1On_PSC_x_k1
        S2OnOff_R1On_PSC_F[:,-2] = S2OnOff_R1On_PSC_F[:,-1]
        S2OnOff_R1On_PSC_F[:,-1] = S2OnOff_R1On_PSC_F[:,-1]+dt*S2OnOff_R1On_PSC_F_k1
        S2OnOff_R1On_PSC_P[:,-2] = S2OnOff_R1On_PSC_P[:,-1]
        S2OnOff_R1On_PSC_P[:,-1] = S2OnOff_R1On_PSC_P[:,-1]+dt*S2OnOff_R1On_PSC_P_k1
        S2OnOff_R1On_PSC_q[:,-2] = S2OnOff_R1On_PSC_q[:,-1]
        S2OnOff_R1On_PSC_q[:,-1] = S2OnOff_R1On_PSC_q[:,-1]+dt*S2OnOff_R1On_PSC_q_k1
        R2On_S2OnOff_PSC_s[:,-2] = R2On_S2OnOff_PSC_s[:,-1]
        R2On_S2OnOff_PSC_s[:,-1] = R2On_S2OnOff_PSC_s[:,-1]+dt*R2On_S2OnOff_PSC_s_k1
        R2On_S2OnOff_PSC_x[:,-2] = R2On_S2OnOff_PSC_x[:,-1]
        R2On_S2OnOff_PSC_x[:,-1] = R2On_S2OnOff_PSC_x[:,-1]+dt*R2On_S2OnOff_PSC_x_k1
        R2On_S2OnOff_PSC_F[:,-2] = R2On_S2OnOff_PSC_F[:,-1]
        R2On_S2OnOff_PSC_F[:,-1] = R2On_S2OnOff_PSC_F[:,-1]+dt*R2On_S2OnOff_PSC_F_k1
        R2On_S2OnOff_PSC_P[:,-2] = R2On_S2OnOff_PSC_P[:,-1]
        R2On_S2OnOff_PSC_P[:,-1] = R2On_S2OnOff_PSC_P[:,-1]+dt*R2On_S2OnOff_PSC_P_k1
        R2On_S2OnOff_PSC_q[:,-2] = R2On_S2OnOff_PSC_q[:,-1]
        R2On_S2OnOff_PSC_q[:,-1] = R2On_S2OnOff_PSC_q[:,-1]+dt*R2On_S2OnOff_PSC_q_k1
        R2Off_S2OnOff_PSC_s[:,-2] = R2Off_S2OnOff_PSC_s[:,-1]
        R2Off_S2OnOff_PSC_s[:,-1] = R2Off_S2OnOff_PSC_s[:,-1]+dt*R2Off_S2OnOff_PSC_s_k1
        R2Off_S2OnOff_PSC_x[:,-2] = R2Off_S2OnOff_PSC_x[:,-1]
        R2Off_S2OnOff_PSC_x[:,-1] = R2Off_S2OnOff_PSC_x[:,-1]+dt*R2Off_S2OnOff_PSC_x_k1
        R2Off_S2OnOff_PSC_F[:,-2] = R2Off_S2OnOff_PSC_F[:,-1]
        R2Off_S2OnOff_PSC_F[:,-1] = R2Off_S2OnOff_PSC_F[:,-1]+dt*R2Off_S2OnOff_PSC_F_k1
        R2Off_S2OnOff_PSC_P[:,-2] = R2Off_S2OnOff_PSC_P[:,-1]
        R2Off_S2OnOff_PSC_P[:,-1] = R2Off_S2OnOff_PSC_P[:,-1]+dt*R2Off_S2OnOff_PSC_P_k1
        R2Off_S2OnOff_PSC_q[:,-2] = R2Off_S2OnOff_PSC_q[:,-1]
        R2Off_S2OnOff_PSC_q[:,-1] = R2Off_S2OnOff_PSC_q[:,-1]+dt*R2Off_S2OnOff_PSC_q_k1
        R2Off_R1Off_PSC_s[:,-2] = R2Off_R1Off_PSC_s[:,-1]
        R2Off_R1Off_PSC_s[:,-1] = R2Off_R1Off_PSC_s[:,-1]+dt*R2Off_R1Off_PSC_s_k1
        R2Off_R1Off_PSC_x[:,-2] = R2Off_R1Off_PSC_x[:,-1]
        R2Off_R1Off_PSC_x[:,-1] = R2Off_R1Off_PSC_x[:,-1]+dt*R2Off_R1Off_PSC_x_k1
        R2Off_R1Off_PSC_F[:,-2] = R2Off_R1Off_PSC_F[:,-1]
        R2Off_R1Off_PSC_F[:,-1] = R2Off_R1Off_PSC_F[:,-1]+dt*R2Off_R1Off_PSC_F_k1
        R2Off_R1Off_PSC_P[:,-2] = R2Off_R1Off_PSC_P[:,-1]
        R2Off_R1Off_PSC_P[:,-1] = R2Off_R1Off_PSC_P[:,-1]+dt*R2Off_R1Off_PSC_P_k1
        R2Off_R1Off_PSC_q[:,-2] = R2Off_R1Off_PSC_q[:,-1]
        R2Off_R1Off_PSC_q[:,-1] = R2Off_R1Off_PSC_q[:,-1]+dt*R2Off_R1Off_PSC_q_k1
        S2OnOff_R1Off_PSC_s[:,-2] = S2OnOff_R1Off_PSC_s[:,-1]
        S2OnOff_R1Off_PSC_s[:,-1] = S2OnOff_R1Off_PSC_s[:,-1]+dt*S2OnOff_R1Off_PSC_s_k1
        S2OnOff_R1Off_PSC_x[:,-2] = S2OnOff_R1Off_PSC_x[:,-1]
        S2OnOff_R1Off_PSC_x[:,-1] = S2OnOff_R1Off_PSC_x[:,-1]+dt*S2OnOff_R1Off_PSC_x_k1
        S2OnOff_R1Off_PSC_F[:,-2] = S2OnOff_R1Off_PSC_F[:,-1]
        S2OnOff_R1Off_PSC_F[:,-1] = S2OnOff_R1Off_PSC_F[:,-1]+dt*S2OnOff_R1Off_PSC_F_k1
        S2OnOff_R1Off_PSC_P[:,-2] = S2OnOff_R1Off_PSC_P[:,-1]
        S2OnOff_R1Off_PSC_P[:,-1] = S2OnOff_R1Off_PSC_P[:,-1]+dt*S2OnOff_R1Off_PSC_P_k1
        S2OnOff_R1Off_PSC_q[:,-2] = S2OnOff_R1Off_PSC_q[:,-1]
        S2OnOff_R1Off_PSC_q[:,-1] = S2OnOff_R1Off_PSC_q[:,-1]+dt*S2OnOff_R1Off_PSC_q_k1
        R2On_R2On_iNoise_V3_sn[:,-2] = R2On_R2On_iNoise_V3_sn[:,-1]
        R2On_R2On_iNoise_V3_sn[:,-1] = R2On_R2On_iNoise_V3_sn[:,-1]+dt*R2On_R2On_iNoise_V3_sn_k1
        R2On_R2On_iNoise_V3_xn[:,-2] = R2On_R2On_iNoise_V3_xn[:,-1]
        R2On_R2On_iNoise_V3_xn[:,-1] = R2On_R2On_iNoise_V3_xn[:,-1]+dt*R2On_R2On_iNoise_V3_xn_k1

        #Spiking and conditional actions
        mask = ((On_V[:,-1] >= On_V_thresh) & (On_V[:,-2] < On_V_thresh)).astype(np.int8).tolist()
        On_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers = np.flatnonzero(mask)
            On_tspike[spikers, On_buffer_index[spikers].astype(np.int8)-1] = helper[t]
            On_buffer_index[spikers] = (On_buffer_index[spikers] % 5) + 1
        mask = ((Off_V[:,-1] >= Off_V_thresh) & (Off_V[:,-2] < Off_V_thresh)).astype(np.int8).tolist()
        Off_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers = np.flatnonzero(mask)
            Off_tspike[spikers, Off_buffer_index[spikers].astype(np.int8)-1] = helper[t]
            Off_buffer_index[spikers] = (Off_buffer_index[spikers] % 5) + 1
        mask = ((R1On_V[:,-1] >= R1On_V_thresh) & (R1On_V[:,-2] < R1On_V_thresh)).astype(np.int8).tolist()
        R1On_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers = np.flatnonzero(mask)
            R1On_tspike[spikers, R1On_buffer_index[spikers].astype(np.int8)-1] = helper[t]
            R1On_buffer_index[spikers] = (R1On_buffer_index[spikers] % 5) + 1
        mask = ((R1Off_V[:,-1] >= R1Off_V_thresh) & (R1Off_V[:,-2] < R1Off_V_thresh)).astype(np.int8).tolist()
        R1Off_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers = np.flatnonzero(mask)
            R1Off_tspike[spikers, R1Off_buffer_index[spikers].astype(np.int8)-1] = helper[t]
            R1Off_buffer_index[spikers] = (R1Off_buffer_index[spikers] % 5) + 1
        mask = ((S1OnOff_V[:,-1] >= S1OnOff_V_thresh) & (S1OnOff_V[:,-2] < S1OnOff_V_thresh)).astype(np.int8).tolist()
        S1OnOff_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers = np.flatnonzero(mask)
            S1OnOff_tspike[spikers, S1OnOff_buffer_index[spikers].astype(np.int8)-1] = helper[t]
            S1OnOff_buffer_index[spikers] = (S1OnOff_buffer_index[spikers] % 5) + 1
        mask = ((R2On_V[:,-1] >= R2On_V_thresh) & (R2On_V[:,-2] < R2On_V_thresh)).astype(np.int8).tolist()
        R2On_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers = np.flatnonzero(mask)
            R2On_tspike[spikers, R2On_buffer_index[spikers].astype(np.int8)-1] = helper[t]
            R2On_buffer_index[spikers] = (R2On_buffer_index[spikers] % 5) + 1
        mask = ((R2Off_V[:,-1] >= R2Off_V_thresh) & (R2Off_V[:,-2] < R2Off_V_thresh)).astype(np.int8).tolist()
        R2Off_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers = np.flatnonzero(mask)
            R2Off_tspike[spikers, R2Off_buffer_index[spikers].astype(np.int8)-1] = helper[t]
            R2Off_buffer_index[spikers] = (R2Off_buffer_index[spikers] % 5) + 1
        mask = ((S2OnOff_V[:,-1] >= S2OnOff_V_thresh) & (S2OnOff_V[:,-2] < S2OnOff_V_thresh)).astype(np.int8).tolist()
        S2OnOff_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers = np.flatnonzero(mask)
            S2OnOff_tspike[spikers, S2OnOff_buffer_index[spikers].astype(np.int8)-1] = helper[t]
            S2OnOff_buffer_index[spikers] = (S2OnOff_buffer_index[spikers] % 5) + 1

            #Voltage reset and adaptation
        mask = (On_V[:,-1] > On_V_thresh)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            On_V[spikers,-2] = On_V[spikers,-1] 
            On_V[spikers,-1] = On_V_reset 
            On_g_ad[spikers,-2] = On_g_ad[spikers,-1]
            On_g_ad[spikers,-1] = On_g_ad[spikers,-1] + On_g_inc
        mask = np.any((helper[t] <= (On_tspike + On_t_ref)), axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            On_V[spikers,-2] = On_V[spikers,-1]
            On_V[spikers,-1] = On_V_reset
        mask = (Off_V[:,-1] > Off_V_thresh)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            Off_V[spikers,-2] = Off_V[spikers,-1] 
            Off_V[spikers,-1] = Off_V_reset 
            Off_g_ad[spikers,-2] = Off_g_ad[spikers,-1]
            Off_g_ad[spikers,-1] = Off_g_ad[spikers,-1] + Off_g_inc
        mask = np.any((helper[t] <= (Off_tspike + Off_t_ref)), axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            Off_V[spikers,-2] = Off_V[spikers,-1]
            Off_V[spikers,-1] = Off_V_reset
        mask = (R1On_V[:,-1] > R1On_V_thresh)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R1On_V[spikers,-2] = R1On_V[spikers,-1] 
            R1On_V[spikers,-1] = R1On_V_reset 
            R1On_g_ad[spikers,-2] = R1On_g_ad[spikers,-1]
            R1On_g_ad[spikers,-1] = R1On_g_ad[spikers,-1] + R1On_g_inc
        mask = np.any((helper[t] <= (R1On_tspike + R1On_t_ref)), axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R1On_V[spikers,-2] = R1On_V[spikers,-1]
            R1On_V[spikers,-1] = R1On_V_reset
        mask = (R1Off_V[:,-1] > R1Off_V_thresh)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R1Off_V[spikers,-2] = R1Off_V[spikers,-1] 
            R1Off_V[spikers,-1] = R1Off_V_reset 
            R1Off_g_ad[spikers,-2] = R1Off_g_ad[spikers,-1]
            R1Off_g_ad[spikers,-1] = R1Off_g_ad[spikers,-1] + R1Off_g_inc
        mask = np.any((helper[t] <= (R1Off_tspike + R1Off_t_ref)), axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R1Off_V[spikers,-2] = R1Off_V[spikers,-1]
            R1Off_V[spikers,-1] = R1Off_V_reset
        mask = (S1OnOff_V[:,-1] > S1OnOff_V_thresh)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            S1OnOff_V[spikers,-2] = S1OnOff_V[spikers,-1] 
            S1OnOff_V[spikers,-1] = S1OnOff_V_reset 
            S1OnOff_g_ad[spikers,-2] = S1OnOff_g_ad[spikers,-1]
            S1OnOff_g_ad[spikers,-1] = S1OnOff_g_ad[spikers,-1] + S1OnOff_g_inc
        mask = np.any((helper[t] <= (S1OnOff_tspike + S1OnOff_t_ref)), axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            S1OnOff_V[spikers,-2] = S1OnOff_V[spikers,-1]
            S1OnOff_V[spikers,-1] = S1OnOff_V_reset
        mask = (R2On_V[:,-1] > R2On_V_thresh)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R2On_V[spikers,-2] = R2On_V[spikers,-1] 
            R2On_V[spikers,-1] = R2On_V_reset 
            R2On_g_ad[spikers,-2] = R2On_g_ad[spikers,-1]
            R2On_g_ad[spikers,-1] = R2On_g_ad[spikers,-1] + R2On_g_inc
        mask = np.any((helper[t] <= (R2On_tspike + R2On_t_ref)), axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R2On_V[spikers,-2] = R2On_V[spikers,-1]
            R2On_V[spikers,-1] = R2On_V_reset
        mask = (R2Off_V[:,-1] > R2Off_V_thresh)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R2Off_V[spikers,-2] = R2Off_V[spikers,-1] 
            R2Off_V[spikers,-1] = R2Off_V_reset 
            R2Off_g_ad[spikers,-2] = R2Off_g_ad[spikers,-1]
            R2Off_g_ad[spikers,-1] = R2Off_g_ad[spikers,-1] + R2Off_g_inc
        mask = np.any((helper[t] <= (R2Off_tspike + R2Off_t_ref)), axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R2Off_V[spikers,-2] = R2Off_V[spikers,-1]
            R2Off_V[spikers,-1] = R2Off_V_reset
        mask = (S2OnOff_V[:,-1] > S2OnOff_V_thresh)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            S2OnOff_V[spikers,-2] = S2OnOff_V[spikers,-1] 
            S2OnOff_V[spikers,-1] = S2OnOff_V_reset 
            S2OnOff_g_ad[spikers,-2] = S2OnOff_g_ad[spikers,-1]
            S2OnOff_g_ad[spikers,-1] = S2OnOff_g_ad[spikers,-1] + S2OnOff_g_inc
        mask = np.any((helper[t] <= (S2OnOff_tspike + S2OnOff_t_ref)), axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            S2OnOff_V[spikers,-2] = S2OnOff_V[spikers,-1]
            S2OnOff_V[spikers,-1] = S2OnOff_V_reset

            #Update PSC vars
        mask = np.any((helper[t] == (On_tspike + R1On_On_PSC_delay)),axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R1On_On_PSC_x[spikers,-2] = R1On_On_PSC_x[spikers,-1]
            R1On_On_PSC_q[spikers,-2] = R1On_On_PSC_F[spikers,-1]
            R1On_On_PSC_F[spikers,-2] = R1On_On_PSC_F[spikers,-1]
            R1On_On_PSC_P[spikers,-2] = R1On_On_PSC_P[spikers,-1]
            R1On_On_PSC_x[spikers,-1] = R1On_On_PSC_x[spikers,-1] + R1On_On_PSC_q[spikers,-1]
            R1On_On_PSC_q[spikers,-1] = R1On_On_PSC_F[spikers,-1] * R1On_On_PSC_P[spikers,-1]
            R1On_On_PSC_F[spikers,-1] = R1On_On_PSC_F[spikers,-1] + R1On_On_PSC_fF*(R1On_On_PSC_maxF-R1On_On_PSC_F[spikers,-1])
            R1On_On_PSC_P[spikers,-1] = R1On_On_PSC_P[spikers,-1] * (1 - R1On_On_PSC_fP)
        mask = np.any((helper[t] == (On_tspike + S1OnOff_On_PSC_delay)),axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            S1OnOff_On_PSC_x[spikers,-2] = S1OnOff_On_PSC_x[spikers,-1]
            S1OnOff_On_PSC_q[spikers,-2] = S1OnOff_On_PSC_F[spikers,-1]
            S1OnOff_On_PSC_F[spikers,-2] = S1OnOff_On_PSC_F[spikers,-1]
            S1OnOff_On_PSC_P[spikers,-2] = S1OnOff_On_PSC_P[spikers,-1]
            S1OnOff_On_PSC_x[spikers,-1] = S1OnOff_On_PSC_x[spikers,-1] + S1OnOff_On_PSC_q[spikers,-1]
            S1OnOff_On_PSC_q[spikers,-1] = S1OnOff_On_PSC_F[spikers,-1] * S1OnOff_On_PSC_P[spikers,-1]
            S1OnOff_On_PSC_F[spikers,-1] = S1OnOff_On_PSC_F[spikers,-1] + S1OnOff_On_PSC_fF*(S1OnOff_On_PSC_maxF-S1OnOff_On_PSC_F[spikers,-1])
            S1OnOff_On_PSC_P[spikers,-1] = S1OnOff_On_PSC_P[spikers,-1] * (1 - S1OnOff_On_PSC_fP)
        mask = np.any((helper[t] == (S1OnOff_tspike + R1On_S1OnOff_PSC_delay)),axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R1On_S1OnOff_PSC_x[spikers,-2] = R1On_S1OnOff_PSC_x[spikers,-1]
            R1On_S1OnOff_PSC_q[spikers,-2] = R1On_S1OnOff_PSC_F[spikers,-1]
            R1On_S1OnOff_PSC_F[spikers,-2] = R1On_S1OnOff_PSC_F[spikers,-1]
            R1On_S1OnOff_PSC_P[spikers,-2] = R1On_S1OnOff_PSC_P[spikers,-1]
            R1On_S1OnOff_PSC_x[spikers,-1] = R1On_S1OnOff_PSC_x[spikers,-1] + R1On_S1OnOff_PSC_q[spikers,-1]
            R1On_S1OnOff_PSC_q[spikers,-1] = R1On_S1OnOff_PSC_F[spikers,-1] * R1On_S1OnOff_PSC_P[spikers,-1]
            R1On_S1OnOff_PSC_F[spikers,-1] = R1On_S1OnOff_PSC_F[spikers,-1] + R1On_S1OnOff_PSC_fF*(R1On_S1OnOff_PSC_maxF-R1On_S1OnOff_PSC_F[spikers,-1])
            R1On_S1OnOff_PSC_P[spikers,-1] = R1On_S1OnOff_PSC_P[spikers,-1] * (1 - R1On_S1OnOff_PSC_fP)
        mask = np.any((helper[t] == (S1OnOff_tspike + R1Off_S1OnOff_PSC_delay)),axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R1Off_S1OnOff_PSC_x[spikers,-2] = R1Off_S1OnOff_PSC_x[spikers,-1]
            R1Off_S1OnOff_PSC_q[spikers,-2] = R1Off_S1OnOff_PSC_F[spikers,-1]
            R1Off_S1OnOff_PSC_F[spikers,-2] = R1Off_S1OnOff_PSC_F[spikers,-1]
            R1Off_S1OnOff_PSC_P[spikers,-2] = R1Off_S1OnOff_PSC_P[spikers,-1]
            R1Off_S1OnOff_PSC_x[spikers,-1] = R1Off_S1OnOff_PSC_x[spikers,-1] + R1Off_S1OnOff_PSC_q[spikers,-1]
            R1Off_S1OnOff_PSC_q[spikers,-1] = R1Off_S1OnOff_PSC_F[spikers,-1] * R1Off_S1OnOff_PSC_P[spikers,-1]
            R1Off_S1OnOff_PSC_F[spikers,-1] = R1Off_S1OnOff_PSC_F[spikers,-1] + R1Off_S1OnOff_PSC_fF*(R1Off_S1OnOff_PSC_maxF-R1Off_S1OnOff_PSC_F[spikers,-1])
            R1Off_S1OnOff_PSC_P[spikers,-1] = R1Off_S1OnOff_PSC_P[spikers,-1] * (1 - R1Off_S1OnOff_PSC_fP)
        mask = np.any((helper[t] == (Off_tspike + R1Off_Off_PSC_delay)),axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R1Off_Off_PSC_x[spikers,-2] = R1Off_Off_PSC_x[spikers,-1]
            R1Off_Off_PSC_q[spikers,-2] = R1Off_Off_PSC_F[spikers,-1]
            R1Off_Off_PSC_F[spikers,-2] = R1Off_Off_PSC_F[spikers,-1]
            R1Off_Off_PSC_P[spikers,-2] = R1Off_Off_PSC_P[spikers,-1]
            R1Off_Off_PSC_x[spikers,-1] = R1Off_Off_PSC_x[spikers,-1] + R1Off_Off_PSC_q[spikers,-1]
            R1Off_Off_PSC_q[spikers,-1] = R1Off_Off_PSC_F[spikers,-1] * R1Off_Off_PSC_P[spikers,-1]
            R1Off_Off_PSC_F[spikers,-1] = R1Off_Off_PSC_F[spikers,-1] + R1Off_Off_PSC_fF*(R1Off_Off_PSC_maxF-R1Off_Off_PSC_F[spikers,-1])
            R1Off_Off_PSC_P[spikers,-1] = R1Off_Off_PSC_P[spikers,-1] * (1 - R1Off_Off_PSC_fP)
        mask = np.any((helper[t] == (Off_tspike + S1OnOff_Off_PSC_delay)),axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            S1OnOff_Off_PSC_x[spikers,-2] = S1OnOff_Off_PSC_x[spikers,-1]
            S1OnOff_Off_PSC_q[spikers,-2] = S1OnOff_Off_PSC_F[spikers,-1]
            S1OnOff_Off_PSC_F[spikers,-2] = S1OnOff_Off_PSC_F[spikers,-1]
            S1OnOff_Off_PSC_P[spikers,-2] = S1OnOff_Off_PSC_P[spikers,-1]
            S1OnOff_Off_PSC_x[spikers,-1] = S1OnOff_Off_PSC_x[spikers,-1] + S1OnOff_Off_PSC_q[spikers,-1]
            S1OnOff_Off_PSC_q[spikers,-1] = S1OnOff_Off_PSC_F[spikers,-1] * S1OnOff_Off_PSC_P[spikers,-1]
            S1OnOff_Off_PSC_F[spikers,-1] = S1OnOff_Off_PSC_F[spikers,-1] + S1OnOff_Off_PSC_fF*(S1OnOff_Off_PSC_maxF-S1OnOff_Off_PSC_F[spikers,-1])
            S1OnOff_Off_PSC_P[spikers,-1] = S1OnOff_Off_PSC_P[spikers,-1] * (1 - S1OnOff_Off_PSC_fP)
        mask = np.any((helper[t] == (R1On_tspike + R2On_R1On_PSC_delay)),axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R2On_R1On_PSC_x[spikers,-2] = R2On_R1On_PSC_x[spikers,-1]
            R2On_R1On_PSC_q[spikers,-2] = R2On_R1On_PSC_F[spikers,-1]
            R2On_R1On_PSC_F[spikers,-2] = R2On_R1On_PSC_F[spikers,-1]
            R2On_R1On_PSC_P[spikers,-2] = R2On_R1On_PSC_P[spikers,-1]
            R2On_R1On_PSC_x[spikers,-1] = R2On_R1On_PSC_x[spikers,-1] + R2On_R1On_PSC_q[spikers,-1]
            R2On_R1On_PSC_q[spikers,-1] = R2On_R1On_PSC_F[spikers,-1] * R2On_R1On_PSC_P[spikers,-1]
            R2On_R1On_PSC_F[spikers,-1] = R2On_R1On_PSC_F[spikers,-1] + R2On_R1On_PSC_fF*(R2On_R1On_PSC_maxF-R2On_R1On_PSC_F[spikers,-1])
            R2On_R1On_PSC_P[spikers,-1] = R2On_R1On_PSC_P[spikers,-1] * (1 - R2On_R1On_PSC_fP)
        mask = np.any((helper[t] == (R1On_tspike + S2OnOff_R1On_PSC_delay)),axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            S2OnOff_R1On_PSC_x[spikers,-2] = S2OnOff_R1On_PSC_x[spikers,-1]
            S2OnOff_R1On_PSC_q[spikers,-2] = S2OnOff_R1On_PSC_F[spikers,-1]
            S2OnOff_R1On_PSC_F[spikers,-2] = S2OnOff_R1On_PSC_F[spikers,-1]
            S2OnOff_R1On_PSC_P[spikers,-2] = S2OnOff_R1On_PSC_P[spikers,-1]
            S2OnOff_R1On_PSC_x[spikers,-1] = S2OnOff_R1On_PSC_x[spikers,-1] + S2OnOff_R1On_PSC_q[spikers,-1]
            S2OnOff_R1On_PSC_q[spikers,-1] = S2OnOff_R1On_PSC_F[spikers,-1] * S2OnOff_R1On_PSC_P[spikers,-1]
            S2OnOff_R1On_PSC_F[spikers,-1] = S2OnOff_R1On_PSC_F[spikers,-1] + S2OnOff_R1On_PSC_fF*(S2OnOff_R1On_PSC_maxF-S2OnOff_R1On_PSC_F[spikers,-1])
            S2OnOff_R1On_PSC_P[spikers,-1] = S2OnOff_R1On_PSC_P[spikers,-1] * (1 - S2OnOff_R1On_PSC_fP)
        mask = np.any((helper[t] == (S2OnOff_tspike + R2On_S2OnOff_PSC_delay)),axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R2On_S2OnOff_PSC_x[spikers,-2] = R2On_S2OnOff_PSC_x[spikers,-1]
            R2On_S2OnOff_PSC_q[spikers,-2] = R2On_S2OnOff_PSC_F[spikers,-1]
            R2On_S2OnOff_PSC_F[spikers,-2] = R2On_S2OnOff_PSC_F[spikers,-1]
            R2On_S2OnOff_PSC_P[spikers,-2] = R2On_S2OnOff_PSC_P[spikers,-1]
            R2On_S2OnOff_PSC_x[spikers,-1] = R2On_S2OnOff_PSC_x[spikers,-1] + R2On_S2OnOff_PSC_q[spikers,-1]
            R2On_S2OnOff_PSC_q[spikers,-1] = R2On_S2OnOff_PSC_F[spikers,-1] * R2On_S2OnOff_PSC_P[spikers,-1]
            R2On_S2OnOff_PSC_F[spikers,-1] = R2On_S2OnOff_PSC_F[spikers,-1] + R2On_S2OnOff_PSC_fF*(R2On_S2OnOff_PSC_maxF-R2On_S2OnOff_PSC_F[spikers,-1])
            R2On_S2OnOff_PSC_P[spikers,-1] = R2On_S2OnOff_PSC_P[spikers,-1] * (1 - R2On_S2OnOff_PSC_fP)
        mask = np.any((helper[t] == (S2OnOff_tspike + R2Off_S2OnOff_PSC_delay)),axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R2Off_S2OnOff_PSC_x[spikers,-2] = R2Off_S2OnOff_PSC_x[spikers,-1]
            R2Off_S2OnOff_PSC_q[spikers,-2] = R2Off_S2OnOff_PSC_F[spikers,-1]
            R2Off_S2OnOff_PSC_F[spikers,-2] = R2Off_S2OnOff_PSC_F[spikers,-1]
            R2Off_S2OnOff_PSC_P[spikers,-2] = R2Off_S2OnOff_PSC_P[spikers,-1]
            R2Off_S2OnOff_PSC_x[spikers,-1] = R2Off_S2OnOff_PSC_x[spikers,-1] + R2Off_S2OnOff_PSC_q[spikers,-1]
            R2Off_S2OnOff_PSC_q[spikers,-1] = R2Off_S2OnOff_PSC_F[spikers,-1] * R2Off_S2OnOff_PSC_P[spikers,-1]
            R2Off_S2OnOff_PSC_F[spikers,-1] = R2Off_S2OnOff_PSC_F[spikers,-1] + R2Off_S2OnOff_PSC_fF*(R2Off_S2OnOff_PSC_maxF-R2Off_S2OnOff_PSC_F[spikers,-1])
            R2Off_S2OnOff_PSC_P[spikers,-1] = R2Off_S2OnOff_PSC_P[spikers,-1] * (1 - R2Off_S2OnOff_PSC_fP)
        mask = np.any((helper[t] == (R1Off_tspike + R2Off_R1Off_PSC_delay)),axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            R2Off_R1Off_PSC_x[spikers,-2] = R2Off_R1Off_PSC_x[spikers,-1]
            R2Off_R1Off_PSC_q[spikers,-2] = R2Off_R1Off_PSC_F[spikers,-1]
            R2Off_R1Off_PSC_F[spikers,-2] = R2Off_R1Off_PSC_F[spikers,-1]
            R2Off_R1Off_PSC_P[spikers,-2] = R2Off_R1Off_PSC_P[spikers,-1]
            R2Off_R1Off_PSC_x[spikers,-1] = R2Off_R1Off_PSC_x[spikers,-1] + R2Off_R1Off_PSC_q[spikers,-1]
            R2Off_R1Off_PSC_q[spikers,-1] = R2Off_R1Off_PSC_F[spikers,-1] * R2Off_R1Off_PSC_P[spikers,-1]
            R2Off_R1Off_PSC_F[spikers,-1] = R2Off_R1Off_PSC_F[spikers,-1] + R2Off_R1Off_PSC_fF*(R2Off_R1Off_PSC_maxF-R2Off_R1Off_PSC_F[spikers,-1])
            R2Off_R1Off_PSC_P[spikers,-1] = R2Off_R1Off_PSC_P[spikers,-1] * (1 - R2Off_R1Off_PSC_fP)
        mask = np.any((helper[t] == (R1Off_tspike + S2OnOff_R1Off_PSC_delay)),axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            S2OnOff_R1Off_PSC_x[spikers,-2] = S2OnOff_R1Off_PSC_x[spikers,-1]
            S2OnOff_R1Off_PSC_q[spikers,-2] = S2OnOff_R1Off_PSC_F[spikers,-1]
            S2OnOff_R1Off_PSC_F[spikers,-2] = S2OnOff_R1Off_PSC_F[spikers,-1]
            S2OnOff_R1Off_PSC_P[spikers,-2] = S2OnOff_R1Off_PSC_P[spikers,-1]
            S2OnOff_R1Off_PSC_x[spikers,-1] = S2OnOff_R1Off_PSC_x[spikers,-1] + S2OnOff_R1Off_PSC_q[spikers,-1]
            S2OnOff_R1Off_PSC_q[spikers,-1] = S2OnOff_R1Off_PSC_F[spikers,-1] * S2OnOff_R1Off_PSC_P[spikers,-1]
            S2OnOff_R1Off_PSC_F[spikers,-1] = S2OnOff_R1Off_PSC_F[spikers,-1] + S2OnOff_R1Off_PSC_fF*(S2OnOff_R1Off_PSC_maxF-S2OnOff_R1Off_PSC_F[spikers,-1])
            S2OnOff_R1Off_PSC_P[spikers,-1] = S2OnOff_R1Off_PSC_P[spikers,-1] * (1 - S2OnOff_R1Off_PSC_fP)

    return R2On_V_spikes_holder