import genPoissonTimes
import genPoissonInputs
import matplotlib.pyplot as plt
import gc
import scipy.io
import numpy as np
from numba import njit

@njit
def forwards(ps,scale_factor):

    #Params
    tspan = np.array([0.1, 3500.0*scale_factor])
    downsample_factor = 1.0


    disk_flag = 0.0
    dt = 0.1

    mex_flag = 0.0
    verbose_flag = 0.0
    On_C = 0.1
    On_g_L = 0.005
    On_E_L = -65.0
    On_noise = 0.0
    On_t_ref = 1.0
    On_E_k = -80.0
    On_tau_ad = 5.0
    On_g_inc = 0.0
    On_Itonic = 0.0
    On_V_thresh = -47.0
    On_V_reset = -54.0
    On_Npop = 1
    Off_C = 0.1
    Off_g_L = 0.005
    Off_E_L = -65.0
    Off_noise = 0.0
    Off_t_ref = 1.0
    Off_E_k = -80.0
    Off_tau_ad = 5.0
    Off_g_inc = 0.0
    Off_Itonic = 0.0
    Off_V_thresh = -47.0
    Off_V_reset = -54.0
    Off_Npop = 1
    R1On_C = 0.1
    R1On_g_L = 0.005
    R1On_E_L = -65.0
    R1On_noise = 0.0
    R1On_t_ref = 1.0
    R1On_E_k = -80.0
    R1On_tau_ad = 100.0
    R1On_g_inc = 0.0003
    R1On_Itonic = 0.0
    R1On_V_thresh = -47.0
    R1On_V_reset = -54.0
    R1On_Npop = 1
    R1Off_C = 0.1
    R1Off_g_L = 0.005
    R1Off_E_L = -65.0
    R1Off_noise = 0.0
    R1Off_t_ref = 1.0
    R1Off_E_k = -80.0
    R1Off_tau_ad = 100.0
    R1Off_g_inc = 0.0003
    R1Off_Itonic = 0.0
    R1Off_V_thresh = -47.0
    R1Off_V_reset = -54.0
    R1Off_Npop = 1
    S1OnOff_C = 0.1
    S1OnOff_g_L = 0.01
    S1OnOff_E_L = -57.0
    S1OnOff_noise = 0.0
    S1OnOff_t_ref = 0.5
    S1OnOff_E_k = -80.0
    S1OnOff_tau_ad = 5.0
    S1OnOff_g_inc = 0.0
    S1OnOff_Itonic = 0.0
    S1OnOff_V_thresh = -47.0
    S1OnOff_V_reset = -52.0
    S1OnOff_Npop = 1
    R2On_C = 0.1
    R2On_g_L = 0.005
    R2On_E_L = -65.0
    R2On_noise = 0.0
    R2On_t_ref = 1.0
    R2On_E_k = -80.0
    R2On_tau_ad = 100.0
    R2On_g_inc = 0.0003
    R2On_Itonic = 0.0
    R2On_V_thresh = -47.0
    R2On_V_reset = -54.0
    R2On_Npop = 1
    R2Off_C = 0.1
    R2Off_g_L = 0.005
    R2Off_E_L = -65.0
    R2Off_noise = 0.0
    R2Off_t_ref = 1.0
    R2Off_E_k = -80.0
    R2Off_tau_ad = 100.0
    R2Off_g_inc = 0.0003
    R2Off_Itonic = 0.0
    R2Off_V_thresh = -47.0
    R2Off_V_reset = -54.0
    R2Off_Npop = 1
    S2OnOff_C = 0.1
    S2OnOff_g_L = 0.01
    S2OnOff_E_L = -57.0
    S2OnOff_noise = 0.0
    S2OnOff_t_ref = 0.5
    S2OnOff_E_k = -80.0
    S2OnOff_tau_ad = 5.0
    S2OnOff_g_inc = 0.0
    S2OnOff_Itonic = 0.0
    S2OnOff_V_thresh = -47.0
    S2OnOff_V_reset = -52.0
    S2OnOff_Npop = 1
    On_On_IC_trial = 20.0
    On_On_IC_locNum = 15.0
    On_On_IC_label = 'on'
    On_On_IC_t_ref = 1.0
    On_On_IC_t_ref_rel = 1.0
    On_On_IC_rec = 2.0
    On_On_IC_g_postIC = 0.17
    On_On_IC_E_exc = 0.0
    Off_Off_IC_trial = 20.0
    Off_Off_IC_locNum = 15.0
    Off_Off_IC_label = 'off'
    Off_Off_IC_t_ref = 1.0
    Off_Off_IC_t_ref_rel = 1.0
    Off_Off_IC_rec = 2.0
    Off_Off_IC_g_postIC = 0.17
    Off_Off_IC_E_exc = 0.0
    R1On_On_PSC_ESYN = 0.0
    R1On_On_PSC_tauD = 1.5
    R1On_On_PSC_tauR = 0.7
    R1On_On_PSC_delay = 0.0
    R1On_On_PSC_gSYN = ps[0]
    R1On_On_PSC_fF = 0.0
    R1On_On_PSC_fP = 0.1
    R1On_On_PSC_tauF = 180.0
    R1On_On_PSC_tauP = 30.0
    R1On_On_PSC_maxF = 4.0
    S1OnOff_On_PSC_ESYN = 0.0
    S1OnOff_On_PSC_tauD = 1.0
    S1OnOff_On_PSC_tauR = 0.1
    S1OnOff_On_PSC_delay = 0.0
    S1OnOff_On_PSC_gSYN = ps[1]
    S1OnOff_On_PSC_fF = 0.0
    S1OnOff_On_PSC_fP = 0.2
    S1OnOff_On_PSC_tauF = 180.0
    S1OnOff_On_PSC_tauP = 80.0
    S1OnOff_On_PSC_maxF = 4.0
    R1On_S1OnOff_PSC_ESYN = -80.0
    R1On_S1OnOff_PSC_tauD = 4.5
    R1On_S1OnOff_PSC_tauR = 1.0
    R1On_S1OnOff_PSC_delay = 0.0
    R1On_S1OnOff_PSC_gSYN = ps[2]
    R1On_S1OnOff_PSC_fF = 0.0
    R1On_S1OnOff_PSC_fP = 0.5
    R1On_S1OnOff_PSC_tauF = 180.0
    R1On_S1OnOff_PSC_tauP = 120.0
    R1On_S1OnOff_PSC_maxF = 4.0
    R1Off_S1OnOff_PSC_ESYN = -80.0
    R1Off_S1OnOff_PSC_tauD = 4.5
    R1Off_S1OnOff_PSC_tauR = 1.0
    R1Off_S1OnOff_PSC_delay = 0.0
    R1Off_S1OnOff_PSC_gSYN = ps[3]
    R1Off_S1OnOff_PSC_fF = 0.0
    R1Off_S1OnOff_PSC_fP = 0.5
    R1Off_S1OnOff_PSC_tauF = 180.0
    R1Off_S1OnOff_PSC_tauP = 120.0
    R1Off_S1OnOff_PSC_maxF = 4.0
    R1Off_Off_PSC_ESYN = 0.0
    R1Off_Off_PSC_tauD = 1.5
    R1Off_Off_PSC_tauR = 0.7
    R1Off_Off_PSC_delay = 0.0
    R1Off_Off_PSC_gSYN = ps[4]
    R1Off_Off_PSC_fF = 0.0
    R1Off_Off_PSC_fP = 0.1
    R1Off_Off_PSC_tauF = 180.0
    R1Off_Off_PSC_tauP = 30.0
    R1Off_Off_PSC_maxF = 4.0
    S1OnOff_Off_PSC_ESYN = 0.0
    S1OnOff_Off_PSC_tauD = 1.0
    S1OnOff_Off_PSC_tauR = 0.1
    S1OnOff_Off_PSC_delay = 0.0
    S1OnOff_Off_PSC_gSYN = ps[5]
    S1OnOff_Off_PSC_fF = 0.0
    S1OnOff_Off_PSC_fP = 0.0
    S1OnOff_Off_PSC_tauF = 180.0
    S1OnOff_Off_PSC_tauP = 80.0
    S1OnOff_Off_PSC_maxF = 4.0
    R2On_R1On_PSC_ESYN = 0.0
    R2On_R1On_PSC_tauD = 1.5
    R2On_R1On_PSC_tauR = 0.7
    R2On_R1On_PSC_delay = 0.0
    R2On_R1On_PSC_gSYN = ps[6]
    R2On_R1On_PSC_fF = 0.0
    R2On_R1On_PSC_fP = 0.1
    R2On_R1On_PSC_tauF = 180.0
    R2On_R1On_PSC_tauP = 30.0
    R2On_R1On_PSC_maxF = 4.0
    S2OnOff_R1On_PSC_ESYN = 0.0
    S2OnOff_R1On_PSC_tauD = 1.0
    S2OnOff_R1On_PSC_tauR = 0.1
    S2OnOff_R1On_PSC_delay = 0.0
    S2OnOff_R1On_PSC_gSYN = ps[7]
    S2OnOff_R1On_PSC_fF = 0.0
    S2OnOff_R1On_PSC_fP = 0.2
    S2OnOff_R1On_PSC_tauF = 180.0
    S2OnOff_R1On_PSC_tauP = 80.0
    S2OnOff_R1On_PSC_maxF = 4.0
    R2On_S2OnOff_PSC_ESYN = -80.0
    R2On_S2OnOff_PSC_tauD = 4.5
    R2On_S2OnOff_PSC_tauR = 1.0
    R2On_S2OnOff_PSC_delay = 0.0
    R2On_S2OnOff_PSC_gSYN = ps[8]
    R2On_S2OnOff_PSC_fF = 0.0
    R2On_S2OnOff_PSC_fP = 0.5
    R2On_S2OnOff_PSC_tauF = 180.0
    R2On_S2OnOff_PSC_tauP = 120.0
    R2On_S2OnOff_PSC_maxF = 4.0
    R2Off_S2OnOff_PSC_ESYN = -80.0
    R2Off_S2OnOff_PSC_tauD = 4.5
    R2Off_S2OnOff_PSC_tauR = 1.0
    R2Off_S2OnOff_PSC_delay = 0.0
    R2Off_S2OnOff_PSC_gSYN = 0.025
    R2Off_S2OnOff_PSC_fF = 0.0
    R2Off_S2OnOff_PSC_fP = 0.5
    R2Off_S2OnOff_PSC_tauF = 180.0
    R2Off_S2OnOff_PSC_tauP = 120.0
    R2Off_S2OnOff_PSC_maxF = 4.0
    R2Off_R1Off_PSC_ESYN = 0.0
    R2Off_R1Off_PSC_tauD = 1.5
    R2Off_R1Off_PSC_tauR = 0.7
    R2Off_R1Off_PSC_delay = 0.0
    R2Off_R1Off_PSC_gSYN = 0.02
    R2Off_R1Off_PSC_fF = 0.0
    R2Off_R1Off_PSC_fP = 0.1
    R2Off_R1Off_PSC_tauF = 180.0
    R2Off_R1Off_PSC_tauP = 30.0
    R2Off_R1Off_PSC_maxF = 4.0
    S2OnOff_R1Off_PSC_ESYN = 0.0
    S2OnOff_R1Off_PSC_tauD = 1.0
    S2OnOff_R1Off_PSC_tauR = 0.1
    S2OnOff_R1Off_PSC_delay = 0.0
    S2OnOff_R1Off_PSC_gSYN = ps[9]
    S2OnOff_R1Off_PSC_fF = 0.0
    S2OnOff_R1Off_PSC_fP = 0.0
    S2OnOff_R1Off_PSC_tauF = 180.0
    S2OnOff_R1Off_PSC_tauP = 80.0
    S2OnOff_R1Off_PSC_maxF = 4.0
    R2On_R2On_iNoise_V3_FR = 8.0
    R2On_R2On_iNoise_V3_sigma = 0.0
    R2On_R2On_iNoise_V3_dt = 0.1
    R2On_R2On_iNoise_V3_nSYN = 0.015
    R2On_R2On_iNoise_V3_simlen = tspan[1]*10
    R2On_R2On_iNoise_V3_tauD_N = 1.5
    R2On_R2On_iNoise_V3_tauR_N = 0.7
    R2On_R2On_iNoise_V3_E_exc = 0.0
    ROn_X_PSC3_netcon = 1.0
    ROn_SOnOff_PSC3_netcon = 1.0
    C_ROn_PSC3_netcon = 1.0
    dGSYNR1On_On = 0
    dGSYNS1OnOff_On = 0
    dGSYNR1On_S1OnOff = 0
    dGSYNR1Off_S1OnOff = 0
    dGSYNR1Off_Off = 0
    dGSYNS1OnOff_Off = 0
    dGSYNR2On_R1On = 0
    dGSYNS2OnOff_R1On = 0
    dGSYNR2On_S2OnOff = 0
    dGSYNS2OnOff_R1Off = 0
    psc_derivative = []
    voltage_derivative = []

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

    T = len(np.arange(tspan[0],tspan[1],dt))
    helper = np.arange(tspan[0],tspan[1],dt)
    grad_On_V = 0
    grad_Off_V = 0
    grad_R1On_V = 0
    grad_R1Off_V = 0
    grad_S1OnOff_V = 0
    grad_R2On_V = 0
    grad_R2Off_V = 0
    grad_S2OnOff_V = 0

    #Spikes Holders
    On_V_spikes = []
    Off_V_spikes = []
    R1On_V_spikes = []
    R1Off_V_spikes = []
    S1OnOff_V_spikes = []
    R2On_V_spikes = []
    R2Off_V_spikes = []
    S2OnOff_V_spikes = []

    for trial_number in range(10):

        #State Variable Declaration
        On_V = [On_E_L, On_E_L]
        On_g_ad = [0,0]
        Off_V = [Off_E_L, Off_E_L]
        Off_g_ad = [0,0]
        R1On_V = [R1On_E_L, R1On_E_L]
        R1On_g_ad = [0,0]
        R1Off_V = [R1Off_E_L, R1Off_E_L]
        R1Off_g_ad = [0,0]
        S1OnOff_V = [S1OnOff_E_L, S1OnOff_E_L]
        S1OnOff_g_ad = [0,0]
        R2On_V = [R2On_E_L, R2On_E_L]
        R2On_g_ad = [0,0]
        R2Off_V = [R2Off_E_L, R2Off_E_L]
        R2Off_g_ad = [0,0]
        S2OnOff_V = [S2OnOff_E_L, S2OnOff_E_L]
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

        #Monitor Declaration
        On_tspike = -1e32*np.ones((5,On_Npop))
        On_buffer_index = np.ones((1,On_Npop))
        On_V_spikes_holder = []
        Off_tspike = -1e32*np.ones((5,Off_Npop))
        Off_buffer_index = np.ones((1,Off_Npop))
        Off_V_spikes_holder = []
        R1On_tspike = -1e32*np.ones((5,R1On_Npop))
        R1On_buffer_index = np.ones((1,R1On_Npop))
        R1On_V_spikes_holder = []
        R1Off_tspike = -1e32*np.ones((5,R1Off_Npop))
        R1Off_buffer_index = np.ones((1,R1Off_Npop))
        R1Off_V_spikes_holder = []
        S1OnOff_tspike = -1e32*np.ones((5,S1OnOff_Npop))
        S1OnOff_buffer_index = np.ones((1,S1OnOff_Npop))
        S1OnOff_V_spikes_holder = []
        R2On_tspike = -1e32*np.ones((5,R2On_Npop))
        R2On_buffer_index = np.ones((1,R2On_Npop))
        R2On_V_spikes_holder = []
        R2Off_tspike = -1e32*np.ones((5,R2Off_Npop))
        R2Off_buffer_index = np.ones((1,R2Off_Npop))
        R2Off_V_spikes_holder = []
        S2OnOff_tspike = -1e32*np.ones((5,S2OnOff_Npop))
        S2OnOff_buffer_index = np.ones((1,S2OnOff_Npop))
        S2OnOff_V_spikes_holder = []
        On_On_IC_iIC = np.zeros((T, On_Npop))
        Off_Off_IC_iIC = np.zeros((T, Off_Npop))
        R1On_On_PSC_syn = np.zeros((T, R1On_Npop))
        S1OnOff_On_PSC_syn = np.zeros((T, S1OnOff_Npop))
        R1On_S1OnOff_PSC_syn = np.zeros((T, R1On_Npop))
        R1Off_S1OnOff_PSC_syn = np.zeros((T, R1Off_Npop))
        R1Off_Off_PSC_syn = np.zeros((T, R1Off_Npop))
        S1OnOff_Off_PSC_syn = np.zeros((T, S1OnOff_Npop))
        R2On_R1On_PSC_syn = np.zeros((T, R2On_Npop))
        S2OnOff_R1On_PSC_syn = np.zeros((T, S2OnOff_Npop))
        R2On_S2OnOff_PSC_syn = np.zeros((T, R2On_Npop))
        R2Off_S2OnOff_PSC_syn = np.zeros((T, R2Off_Npop))
        R2Off_R1Off_PSC_syn = np.zeros((T, R2Off_Npop))
        S2OnOff_R1Off_PSC_syn = np.zeros((T, S2OnOff_Npop))

        #Delcare Inputs
        On_On_IC_input = genPoissonInputs.gen_poisson_inputs(trial_number,On_On_IC_locNum,On_On_IC_label,On_On_IC_t_ref,On_On_IC_t_ref_rel,On_On_IC_rec,scale_factor)
        Off_Off_IC_input = genPoissonInputs.gen_poisson_inputs(trial_number,Off_Off_IC_locNum,Off_Off_IC_label,Off_Off_IC_t_ref,Off_Off_IC_t_ref_rel,Off_Off_IC_rec,scale_factor)

        for t in range(1,T):

            #ODEs
            On_V_k1 = ( (On_E_L-On_V[-1]) - On_R*On_g_ad[-1]*(On_V[-1]-On_E_k) - On_R*((((On_On_IC_g_postIC*(On_On_IC_input[t]*On_On_IC_netcon)*(On_V[-1]-On_On_IC_E_exc))))) + On_R*On_Itonic*On_Imask  ) / On_tau
            On_g_ad_k1 = -On_g_ad[-1] / On_tau_ad
            Off_V_k1 = ( (Off_E_L-Off_V[-1]) - Off_R*Off_g_ad[-1]*(Off_V[-1]-Off_E_k) - Off_R*((((Off_Off_IC_g_postIC*(Off_Off_IC_input[t]*Off_Off_IC_netcon)*(Off_V[-1]-Off_Off_IC_E_exc))))) + Off_R*Off_Itonic*Off_Imask  ) / Off_tau
            Off_g_ad_k1 = -Off_g_ad[-1] / Off_tau_ad
            R1On_V_k1 = ( (R1On_E_L-R1On_V[-1]) - R1On_R*R1On_g_ad[-1]*(R1On_V[-1]-R1On_E_k) - R1On_R*((((R1On_On_PSC_gSYN*(R1On_On_PSC_s[-1]*R1On_On_PSC_netcon)*(R1On_V[-1]-R1On_On_PSC_ESYN))))+((((R1On_S1OnOff_PSC_gSYN*(R1On_S1OnOff_PSC_s[-1]*R1On_S1OnOff_PSC_netcon)*(R1On_V[-1]-R1On_S1OnOff_PSC_ESYN)))))) + R1On_R*R1On_Itonic*R1On_Imask  ) / R1On_tau
            R1On_g_ad_k1 = -R1On_g_ad[-1] / R1On_tau_ad
            R1Off_V_k1 = ( (R1Off_E_L-R1Off_V[-1]) - R1Off_R*R1Off_g_ad[-1]*(R1Off_V[-1]-R1Off_E_k) - R1Off_R*((((R1Off_S1OnOff_PSC_gSYN*(R1Off_S1OnOff_PSC_s[-1]*R1Off_S1OnOff_PSC_netcon)*(R1Off_V[-1]-R1Off_S1OnOff_PSC_ESYN))))+((((R1Off_Off_PSC_gSYN*(R1Off_Off_PSC_s[-1]*R1Off_Off_PSC_netcon)*(R1Off_V[-1]-R1Off_Off_PSC_ESYN)))))) + R1Off_R*R1Off_Itonic*R1Off_Imask  ) / R1Off_tau
            R1Off_g_ad_k1 = -R1Off_g_ad[-1] / R1Off_tau_ad
            S1OnOff_V_k1 = ( (S1OnOff_E_L-S1OnOff_V[-1]) - S1OnOff_R*S1OnOff_g_ad[-1]*(S1OnOff_V[-1]-S1OnOff_E_k) - S1OnOff_R*((((S1OnOff_On_PSC_gSYN*(S1OnOff_On_PSC_s[-1]*S1OnOff_On_PSC_netcon)*(S1OnOff_V[-1]-S1OnOff_On_PSC_ESYN))))+((((S1OnOff_Off_PSC_gSYN*(S1OnOff_Off_PSC_s[-1]*S1OnOff_Off_PSC_netcon)*(S1OnOff_V[-1]-S1OnOff_Off_PSC_ESYN)))))) + S1OnOff_R*S1OnOff_Itonic*S1OnOff_Imask  ) / S1OnOff_tau
            S1OnOff_g_ad_k1 = -S1OnOff_g_ad[-1] / S1OnOff_tau_ad
            R2On_V_k1 = ( (R2On_E_L-R2On_V[-1]) - R2On_R*R2On_g_ad[-1]*(R2On_V[-1]-R2On_E_k) - R2On_R*((((R2On_R1On_PSC_gSYN*(R2On_R1On_PSC_s[-1]*R2On_R1On_PSC_netcon)*(R2On_V[-1]-R2On_R1On_PSC_ESYN))))+((((R2On_S2OnOff_PSC_gSYN*(R2On_S2OnOff_PSC_s[-1]*R2On_S2OnOff_PSC_netcon)*(R2On_V[-1]-R2On_S2OnOff_PSC_ESYN))))+((((R2On_R2On_iNoise_V3_nSYN*(R2On_R2On_iNoise_V3_sn[-1]*R2On_R2On_iNoise_V3_netcon)*(R2On_V[-1]-R2On_R2On_iNoise_V3_E_exc))))))) + R2On_R*R2On_Itonic*R2On_Imask  ) / R2On_tau
            R2On_g_ad_k1 = -R2On_g_ad[-1] / R2On_tau_ad
            R2Off_V_k1 = ( (R2Off_E_L-R2Off_V[-1]) - R2Off_R*R2Off_g_ad[-1]*(R2Off_V[-1]-R2Off_E_k) - R2Off_R*((((R2Off_S2OnOff_PSC_gSYN*(R2Off_S2OnOff_PSC_s[-1]*R2Off_S2OnOff_PSC_netcon)*(R2Off_V[-1]-R2Off_S2OnOff_PSC_ESYN))))+((((R2Off_R1Off_PSC_gSYN*(R2Off_R1Off_PSC_s[-1]*R2Off_R1Off_PSC_netcon)*(R2Off_V[-1]-R2Off_R1Off_PSC_ESYN)))))) + R2Off_R*R2Off_Itonic*R2Off_Imask  ) / R2Off_tau
            R2Off_g_ad_k1 = -R2Off_g_ad[-1] / R2Off_tau_ad
            S2OnOff_V_k1 = ( (S2OnOff_E_L-S2OnOff_V[-1]) - S2OnOff_R*S2OnOff_g_ad[-1]*(S2OnOff_V[-1]-S2OnOff_E_k) - S2OnOff_R*((((S2OnOff_R1On_PSC_gSYN*(S2OnOff_R1On_PSC_s[-1]*S2OnOff_R1On_PSC_netcon)*(S2OnOff_V[-1]-S2OnOff_R1On_PSC_ESYN))))+((((S2OnOff_R1Off_PSC_gSYN*(S2OnOff_R1Off_PSC_s[-1]*S2OnOff_R1Off_PSC_netcon)*(S2OnOff_V[-1]-S2OnOff_R1Off_PSC_ESYN)))))) + S2OnOff_R*S2OnOff_Itonic*S2OnOff_Imask  ) / S2OnOff_tau
            S2OnOff_g_ad_k1 = -S2OnOff_g_ad[-1] / S2OnOff_tau_ad
            R1On_On_PSC_s_k1 = ( R1On_On_PSC_scale * R1On_On_PSC_x[-1] - R1On_On_PSC_s[-1] )/R1On_On_PSC_tauR
            R1On_On_PSC_x_k1 = -R1On_On_PSC_x[-1]/R1On_On_PSC_tauD
            R1On_On_PSC_F_k1 = (1 - R1On_On_PSC_F[-1])/R1On_On_PSC_tauF
            R1On_On_PSC_P_k1 = (1 - R1On_On_PSC_P[-1])/R1On_On_PSC_tauP
            R1On_On_PSC_q_k1 = 0
            S1OnOff_On_PSC_s_k1 = ( S1OnOff_On_PSC_scale * S1OnOff_On_PSC_x[-1] - S1OnOff_On_PSC_s[-1] )/S1OnOff_On_PSC_tauR
            S1OnOff_On_PSC_x_k1 = -S1OnOff_On_PSC_x[-1]/S1OnOff_On_PSC_tauD
            S1OnOff_On_PSC_F_k1 = (1 - S1OnOff_On_PSC_F[-1])/S1OnOff_On_PSC_tauF
            S1OnOff_On_PSC_P_k1 = (1 - S1OnOff_On_PSC_P[-1])/S1OnOff_On_PSC_tauP
            S1OnOff_On_PSC_q_k1 = 0
            R1On_S1OnOff_PSC_s_k1 = ( R1On_S1OnOff_PSC_scale * R1On_S1OnOff_PSC_x[-1] - R1On_S1OnOff_PSC_s[-1] )/R1On_S1OnOff_PSC_tauR
            R1On_S1OnOff_PSC_x_k1 = -R1On_S1OnOff_PSC_x[-1]/R1On_S1OnOff_PSC_tauD
            R1On_S1OnOff_PSC_F_k1 = (1 - R1On_S1OnOff_PSC_F[-1])/R1On_S1OnOff_PSC_tauF
            R1On_S1OnOff_PSC_P_k1 = (1 - R1On_S1OnOff_PSC_P[-1])/R1On_S1OnOff_PSC_tauP
            R1On_S1OnOff_PSC_q_k1 = 0
            R1Off_S1OnOff_PSC_s_k1 = ( R1Off_S1OnOff_PSC_scale * R1Off_S1OnOff_PSC_x[-1] - R1Off_S1OnOff_PSC_s[-1] )/R1Off_S1OnOff_PSC_tauR
            R1Off_S1OnOff_PSC_x_k1 = -R1Off_S1OnOff_PSC_x[-1]/R1Off_S1OnOff_PSC_tauD
            R1Off_S1OnOff_PSC_F_k1 = (1 - R1Off_S1OnOff_PSC_F[-1])/R1Off_S1OnOff_PSC_tauF
            R1Off_S1OnOff_PSC_P_k1 = (1 - R1Off_S1OnOff_PSC_P[-1])/R1Off_S1OnOff_PSC_tauP
            R1Off_S1OnOff_PSC_q_k1 = 0
            R1Off_Off_PSC_s_k1 = ( R1Off_Off_PSC_scale * R1Off_Off_PSC_x[-1] - R1Off_Off_PSC_s[-1] )/R1Off_Off_PSC_tauR
            R1Off_Off_PSC_x_k1 = -R1Off_Off_PSC_x[-1]/R1Off_Off_PSC_tauD
            R1Off_Off_PSC_F_k1 = (1 - R1Off_Off_PSC_F[-1])/R1Off_Off_PSC_tauF
            R1Off_Off_PSC_P_k1 = (1 - R1Off_Off_PSC_P[-1])/R1Off_Off_PSC_tauP
            R1Off_Off_PSC_q_k1 = 0
            S1OnOff_Off_PSC_s_k1 = ( S1OnOff_Off_PSC_scale * S1OnOff_Off_PSC_x[-1] - S1OnOff_Off_PSC_s[-1] )/S1OnOff_Off_PSC_tauR
            S1OnOff_Off_PSC_x_k1 = -S1OnOff_Off_PSC_x[-1]/S1OnOff_Off_PSC_tauD
            S1OnOff_Off_PSC_F_k1 = (1 - S1OnOff_Off_PSC_F[-1])/S1OnOff_Off_PSC_tauF
            S1OnOff_Off_PSC_P_k1 = (1 - S1OnOff_Off_PSC_P[-1])/S1OnOff_Off_PSC_tauP
            S1OnOff_Off_PSC_q_k1 = 0
            R2On_R1On_PSC_s_k1 = ( R2On_R1On_PSC_scale * R2On_R1On_PSC_x[-1] - R2On_R1On_PSC_s[-1] )/R2On_R1On_PSC_tauR
            R2On_R1On_PSC_x_k1 = -R2On_R1On_PSC_x[-1]/R2On_R1On_PSC_tauD
            R2On_R1On_PSC_F_k1 = (1 - R2On_R1On_PSC_F[-1])/R2On_R1On_PSC_tauF
            R2On_R1On_PSC_P_k1 = (1 - R2On_R1On_PSC_P[-1])/R2On_R1On_PSC_tauP
            R2On_R1On_PSC_q_k1 = 0
            S2OnOff_R1On_PSC_s_k1 = ( S2OnOff_R1On_PSC_scale * S2OnOff_R1On_PSC_x[-1] - S2OnOff_R1On_PSC_s[-1] )/S2OnOff_R1On_PSC_tauR
            S2OnOff_R1On_PSC_x_k1 = -S2OnOff_R1On_PSC_x[-1]/S2OnOff_R1On_PSC_tauD
            S2OnOff_R1On_PSC_F_k1 = (1 - S2OnOff_R1On_PSC_F[-1])/S2OnOff_R1On_PSC_tauF
            S2OnOff_R1On_PSC_P_k1 = (1 - S2OnOff_R1On_PSC_P[-1])/S2OnOff_R1On_PSC_tauP
            S2OnOff_R1On_PSC_q_k1 = 0
            R2On_S2OnOff_PSC_s_k1 = ( R2On_S2OnOff_PSC_scale * R2On_S2OnOff_PSC_x[-1] - R2On_S2OnOff_PSC_s[-1] )/R2On_S2OnOff_PSC_tauR
            R2On_S2OnOff_PSC_x_k1 = -R2On_S2OnOff_PSC_x[-1]/R2On_S2OnOff_PSC_tauD
            R2On_S2OnOff_PSC_F_k1 = (1 - R2On_S2OnOff_PSC_F[-1])/R2On_S2OnOff_PSC_tauF
            R2On_S2OnOff_PSC_P_k1 = (1 - R2On_S2OnOff_PSC_P[-1])/R2On_S2OnOff_PSC_tauP
            R2On_S2OnOff_PSC_q_k1 = 0
            R2Off_S2OnOff_PSC_s_k1 = ( R2Off_S2OnOff_PSC_scale * R2Off_S2OnOff_PSC_x[-1] - R2Off_S2OnOff_PSC_s[-1] )/R2Off_S2OnOff_PSC_tauR
            R2Off_S2OnOff_PSC_x_k1 = -R2Off_S2OnOff_PSC_x[-1]/R2Off_S2OnOff_PSC_tauD
            R2Off_S2OnOff_PSC_F_k1 = (1 - R2Off_S2OnOff_PSC_F[-1])/R2Off_S2OnOff_PSC_tauF
            R2Off_S2OnOff_PSC_P_k1 = (1 - R2Off_S2OnOff_PSC_P[-1])/R2Off_S2OnOff_PSC_tauP
            R2Off_S2OnOff_PSC_q_k1 = 0
            R2Off_R1Off_PSC_s_k1 = ( R2Off_R1Off_PSC_scale * R2Off_R1Off_PSC_x[-1] - R2Off_R1Off_PSC_s[-1] )/R2Off_R1Off_PSC_tauR
            R2Off_R1Off_PSC_x_k1 = -R2Off_R1Off_PSC_x[-1]/R2Off_R1Off_PSC_tauD
            R2Off_R1Off_PSC_F_k1 = (1 - R2Off_R1Off_PSC_F[-1])/R2Off_R1Off_PSC_tauF
            R2Off_R1Off_PSC_P_k1 = (1 - R2Off_R1Off_PSC_P[-1])/R2Off_R1Off_PSC_tauP
            R2Off_R1Off_PSC_q_k1 = 0
            S2OnOff_R1Off_PSC_s_k1 = ( S2OnOff_R1Off_PSC_scale * S2OnOff_R1Off_PSC_x[-1] - S2OnOff_R1Off_PSC_s[-1] )/S2OnOff_R1Off_PSC_tauR
            S2OnOff_R1Off_PSC_x_k1 = -S2OnOff_R1Off_PSC_x[-1]/S2OnOff_R1Off_PSC_tauD
            S2OnOff_R1Off_PSC_F_k1 = (1 - S2OnOff_R1Off_PSC_F[-1])/S2OnOff_R1Off_PSC_tauF
            S2OnOff_R1Off_PSC_P_k1 = (1 - S2OnOff_R1Off_PSC_P[-1])/S2OnOff_R1Off_PSC_tauP
            S2OnOff_R1Off_PSC_q_k1 = 0
            R2On_R2On_iNoise_V3_sn_k1 = ( R2On_R2On_iNoise_V3_scale * R2On_R2On_iNoise_V3_xn[-1] - R2On_R2On_iNoise_V3_sn[-1] )/R2On_R2On_iNoise_V3_tauR_N
            R2On_R2On_iNoise_V3_xn_k1 = -R2On_R2On_iNoise_V3_xn[-1]/R2On_R2On_iNoise_V3_tauD_N + R2On_R2On_iNoise_V3_token[t]/R2On_R2On_iNoise_V3_dt

            #Update Eulers
            On_V[-2] = On_V[-1]
            On_V[-1] = On_V[-1]+dt*On_V_k1
            On_g_ad[-2] = On_g_ad[-1]
            On_g_ad[-1] = On_g_ad[-1]+dt*On_g_ad_k1
            Off_V[-2] = Off_V[-1]
            Off_V[-1] = Off_V[-1]+dt*Off_V_k1
            Off_g_ad[-2] = Off_g_ad[-1]
            Off_g_ad[-1] = Off_g_ad[-1]+dt*Off_g_ad_k1
            R1On_V[-2] = R1On_V[-1]
            R1On_V[-1] = R1On_V[-1]+dt*R1On_V_k1
            R1On_g_ad[-2] = R1On_g_ad[-1]
            R1On_g_ad[-1] = R1On_g_ad[-1]+dt*R1On_g_ad_k1
            R1Off_V[-2] = R1Off_V[-1]
            R1Off_V[-1] = R1Off_V[-1]+dt*R1Off_V_k1
            R1Off_g_ad[-2] = R1Off_g_ad[-1]
            R1Off_g_ad[-1] = R1Off_g_ad[-1]+dt*R1Off_g_ad_k1
            S1OnOff_V[-2] = S1OnOff_V[-1]
            S1OnOff_V[-1] = S1OnOff_V[-1]+dt*S1OnOff_V_k1
            S1OnOff_g_ad[-2] = S1OnOff_g_ad[-1]
            S1OnOff_g_ad[-1] = S1OnOff_g_ad[-1]+dt*S1OnOff_g_ad_k1
            R2On_V[-2] = R2On_V[-1]
            R2On_V[-1] = R2On_V[-1]+dt*R2On_V_k1
            R2On_g_ad[-2] = R2On_g_ad[-1]
            R2On_g_ad[-1] = R2On_g_ad[-1]+dt*R2On_g_ad_k1
            R2Off_V[-2] = R2Off_V[-1]
            R2Off_V[-1] = R2Off_V[-1]+dt*R2Off_V_k1
            R2Off_g_ad[-2] = R2Off_g_ad[-1]
            R2Off_g_ad[-1] = R2Off_g_ad[-1]+dt*R2Off_g_ad_k1
            S2OnOff_V[-2] = S2OnOff_V[-1]
            S2OnOff_V[-1] = S2OnOff_V[-1]+dt*S2OnOff_V_k1
            S2OnOff_g_ad[-2] = S2OnOff_g_ad[-1]
            S2OnOff_g_ad[-1] = S2OnOff_g_ad[-1]+dt*S2OnOff_g_ad_k1
            R1On_On_PSC_s[-2] = R1On_On_PSC_s[-1]
            R1On_On_PSC_s[-1] = R1On_On_PSC_s[-1]+dt*R1On_On_PSC_s_k1
            R1On_On_PSC_x[-2] = R1On_On_PSC_x[-1]
            R1On_On_PSC_x[-1] = R1On_On_PSC_x[-1]+dt*R1On_On_PSC_x_k1
            R1On_On_PSC_F[-2] = R1On_On_PSC_F[-1]
            R1On_On_PSC_F[-1] = R1On_On_PSC_F[-1]+dt*R1On_On_PSC_F_k1
            R1On_On_PSC_P[-2] = R1On_On_PSC_P[-1]
            R1On_On_PSC_P[-1] = R1On_On_PSC_P[-1]+dt*R1On_On_PSC_P_k1
            R1On_On_PSC_q[-2] = R1On_On_PSC_q[-1]
            R1On_On_PSC_q[-1] = R1On_On_PSC_q[-1]+dt*R1On_On_PSC_q_k1
            S1OnOff_On_PSC_s[-2] = S1OnOff_On_PSC_s[-1]
            S1OnOff_On_PSC_s[-1] = S1OnOff_On_PSC_s[-1]+dt*S1OnOff_On_PSC_s_k1
            S1OnOff_On_PSC_x[-2] = S1OnOff_On_PSC_x[-1]
            S1OnOff_On_PSC_x[-1] = S1OnOff_On_PSC_x[-1]+dt*S1OnOff_On_PSC_x_k1
            S1OnOff_On_PSC_F[-2] = S1OnOff_On_PSC_F[-1]
            S1OnOff_On_PSC_F[-1] = S1OnOff_On_PSC_F[-1]+dt*S1OnOff_On_PSC_F_k1
            S1OnOff_On_PSC_P[-2] = S1OnOff_On_PSC_P[-1]
            S1OnOff_On_PSC_P[-1] = S1OnOff_On_PSC_P[-1]+dt*S1OnOff_On_PSC_P_k1
            S1OnOff_On_PSC_q[-2] = S1OnOff_On_PSC_q[-1]
            S1OnOff_On_PSC_q[-1] = S1OnOff_On_PSC_q[-1]+dt*S1OnOff_On_PSC_q_k1
            R1On_S1OnOff_PSC_s[-2] = R1On_S1OnOff_PSC_s[-1]
            R1On_S1OnOff_PSC_s[-1] = R1On_S1OnOff_PSC_s[-1]+dt*R1On_S1OnOff_PSC_s_k1
            R1On_S1OnOff_PSC_x[-2] = R1On_S1OnOff_PSC_x[-1]
            R1On_S1OnOff_PSC_x[-1] = R1On_S1OnOff_PSC_x[-1]+dt*R1On_S1OnOff_PSC_x_k1
            R1On_S1OnOff_PSC_F[-2] = R1On_S1OnOff_PSC_F[-1]
            R1On_S1OnOff_PSC_F[-1] = R1On_S1OnOff_PSC_F[-1]+dt*R1On_S1OnOff_PSC_F_k1
            R1On_S1OnOff_PSC_P[-2] = R1On_S1OnOff_PSC_P[-1]
            R1On_S1OnOff_PSC_P[-1] = R1On_S1OnOff_PSC_P[-1]+dt*R1On_S1OnOff_PSC_P_k1
            R1On_S1OnOff_PSC_q[-2] = R1On_S1OnOff_PSC_q[-1]
            R1On_S1OnOff_PSC_q[-1] = R1On_S1OnOff_PSC_q[-1]+dt*R1On_S1OnOff_PSC_q_k1
            R1Off_S1OnOff_PSC_s[-2] = R1Off_S1OnOff_PSC_s[-1]
            R1Off_S1OnOff_PSC_s[-1] = R1Off_S1OnOff_PSC_s[-1]+dt*R1Off_S1OnOff_PSC_s_k1
            R1Off_S1OnOff_PSC_x[-2] = R1Off_S1OnOff_PSC_x[-1]
            R1Off_S1OnOff_PSC_x[-1] = R1Off_S1OnOff_PSC_x[-1]+dt*R1Off_S1OnOff_PSC_x_k1
            R1Off_S1OnOff_PSC_F[-2] = R1Off_S1OnOff_PSC_F[-1]
            R1Off_S1OnOff_PSC_F[-1] = R1Off_S1OnOff_PSC_F[-1]+dt*R1Off_S1OnOff_PSC_F_k1
            R1Off_S1OnOff_PSC_P[-2] = R1Off_S1OnOff_PSC_P[-1]
            R1Off_S1OnOff_PSC_P[-1] = R1Off_S1OnOff_PSC_P[-1]+dt*R1Off_S1OnOff_PSC_P_k1
            R1Off_S1OnOff_PSC_q[-2] = R1Off_S1OnOff_PSC_q[-1]
            R1Off_S1OnOff_PSC_q[-1] = R1Off_S1OnOff_PSC_q[-1]+dt*R1Off_S1OnOff_PSC_q_k1
            R1Off_Off_PSC_s[-2] = R1Off_Off_PSC_s[-1]
            R1Off_Off_PSC_s[-1] = R1Off_Off_PSC_s[-1]+dt*R1Off_Off_PSC_s_k1
            R1Off_Off_PSC_x[-2] = R1Off_Off_PSC_x[-1]
            R1Off_Off_PSC_x[-1] = R1Off_Off_PSC_x[-1]+dt*R1Off_Off_PSC_x_k1
            R1Off_Off_PSC_F[-2] = R1Off_Off_PSC_F[-1]
            R1Off_Off_PSC_F[-1] = R1Off_Off_PSC_F[-1]+dt*R1Off_Off_PSC_F_k1
            R1Off_Off_PSC_P[-2] = R1Off_Off_PSC_P[-1]
            R1Off_Off_PSC_P[-1] = R1Off_Off_PSC_P[-1]+dt*R1Off_Off_PSC_P_k1
            R1Off_Off_PSC_q[-2] = R1Off_Off_PSC_q[-1]
            R1Off_Off_PSC_q[-1] = R1Off_Off_PSC_q[-1]+dt*R1Off_Off_PSC_q_k1
            S1OnOff_Off_PSC_s[-2] = S1OnOff_Off_PSC_s[-1]
            S1OnOff_Off_PSC_s[-1] = S1OnOff_Off_PSC_s[-1]+dt*S1OnOff_Off_PSC_s_k1
            S1OnOff_Off_PSC_x[-2] = S1OnOff_Off_PSC_x[-1]
            S1OnOff_Off_PSC_x[-1] = S1OnOff_Off_PSC_x[-1]+dt*S1OnOff_Off_PSC_x_k1
            S1OnOff_Off_PSC_F[-2] = S1OnOff_Off_PSC_F[-1]
            S1OnOff_Off_PSC_F[-1] = S1OnOff_Off_PSC_F[-1]+dt*S1OnOff_Off_PSC_F_k1
            S1OnOff_Off_PSC_P[-2] = S1OnOff_Off_PSC_P[-1]
            S1OnOff_Off_PSC_P[-1] = S1OnOff_Off_PSC_P[-1]+dt*S1OnOff_Off_PSC_P_k1
            S1OnOff_Off_PSC_q[-2] = S1OnOff_Off_PSC_q[-1]
            S1OnOff_Off_PSC_q[-1] = S1OnOff_Off_PSC_q[-1]+dt*S1OnOff_Off_PSC_q_k1
            R2On_R1On_PSC_s[-2] = R2On_R1On_PSC_s[-1]
            R2On_R1On_PSC_s[-1] = R2On_R1On_PSC_s[-1]+dt*R2On_R1On_PSC_s_k1
            R2On_R1On_PSC_x[-2] = R2On_R1On_PSC_x[-1]
            R2On_R1On_PSC_x[-1] = R2On_R1On_PSC_x[-1]+dt*R2On_R1On_PSC_x_k1
            R2On_R1On_PSC_F[-2] = R2On_R1On_PSC_F[-1]
            R2On_R1On_PSC_F[-1] = R2On_R1On_PSC_F[-1]+dt*R2On_R1On_PSC_F_k1
            R2On_R1On_PSC_P[-2] = R2On_R1On_PSC_P[-1]
            R2On_R1On_PSC_P[-1] = R2On_R1On_PSC_P[-1]+dt*R2On_R1On_PSC_P_k1
            R2On_R1On_PSC_q[-2] = R2On_R1On_PSC_q[-1]
            R2On_R1On_PSC_q[-1] = R2On_R1On_PSC_q[-1]+dt*R2On_R1On_PSC_q_k1
            S2OnOff_R1On_PSC_s[-2] = S2OnOff_R1On_PSC_s[-1]
            S2OnOff_R1On_PSC_s[-1] = S2OnOff_R1On_PSC_s[-1]+dt*S2OnOff_R1On_PSC_s_k1
            S2OnOff_R1On_PSC_x[-2] = S2OnOff_R1On_PSC_x[-1]
            S2OnOff_R1On_PSC_x[-1] = S2OnOff_R1On_PSC_x[-1]+dt*S2OnOff_R1On_PSC_x_k1
            S2OnOff_R1On_PSC_F[-2] = S2OnOff_R1On_PSC_F[-1]
            S2OnOff_R1On_PSC_F[-1] = S2OnOff_R1On_PSC_F[-1]+dt*S2OnOff_R1On_PSC_F_k1
            S2OnOff_R1On_PSC_P[-2] = S2OnOff_R1On_PSC_P[-1]
            S2OnOff_R1On_PSC_P[-1] = S2OnOff_R1On_PSC_P[-1]+dt*S2OnOff_R1On_PSC_P_k1
            S2OnOff_R1On_PSC_q[-2] = S2OnOff_R1On_PSC_q[-1]
            S2OnOff_R1On_PSC_q[-1] = S2OnOff_R1On_PSC_q[-1]+dt*S2OnOff_R1On_PSC_q_k1
            R2On_S2OnOff_PSC_s[-2] = R2On_S2OnOff_PSC_s[-1]
            R2On_S2OnOff_PSC_s[-1] = R2On_S2OnOff_PSC_s[-1]+dt*R2On_S2OnOff_PSC_s_k1
            R2On_S2OnOff_PSC_x[-2] = R2On_S2OnOff_PSC_x[-1]
            R2On_S2OnOff_PSC_x[-1] = R2On_S2OnOff_PSC_x[-1]+dt*R2On_S2OnOff_PSC_x_k1
            R2On_S2OnOff_PSC_F[-2] = R2On_S2OnOff_PSC_F[-1]
            R2On_S2OnOff_PSC_F[-1] = R2On_S2OnOff_PSC_F[-1]+dt*R2On_S2OnOff_PSC_F_k1
            R2On_S2OnOff_PSC_P[-2] = R2On_S2OnOff_PSC_P[-1]
            R2On_S2OnOff_PSC_P[-1] = R2On_S2OnOff_PSC_P[-1]+dt*R2On_S2OnOff_PSC_P_k1
            R2On_S2OnOff_PSC_q[-2] = R2On_S2OnOff_PSC_q[-1]
            R2On_S2OnOff_PSC_q[-1] = R2On_S2OnOff_PSC_q[-1]+dt*R2On_S2OnOff_PSC_q_k1
            R2Off_S2OnOff_PSC_s[-2] = R2Off_S2OnOff_PSC_s[-1]
            R2Off_S2OnOff_PSC_s[-1] = R2Off_S2OnOff_PSC_s[-1]+dt*R2Off_S2OnOff_PSC_s_k1
            R2Off_S2OnOff_PSC_x[-2] = R2Off_S2OnOff_PSC_x[-1]
            R2Off_S2OnOff_PSC_x[-1] = R2Off_S2OnOff_PSC_x[-1]+dt*R2Off_S2OnOff_PSC_x_k1
            R2Off_S2OnOff_PSC_F[-2] = R2Off_S2OnOff_PSC_F[-1]
            R2Off_S2OnOff_PSC_F[-1] = R2Off_S2OnOff_PSC_F[-1]+dt*R2Off_S2OnOff_PSC_F_k1
            R2Off_S2OnOff_PSC_P[-2] = R2Off_S2OnOff_PSC_P[-1]
            R2Off_S2OnOff_PSC_P[-1] = R2Off_S2OnOff_PSC_P[-1]+dt*R2Off_S2OnOff_PSC_P_k1
            R2Off_S2OnOff_PSC_q[-2] = R2Off_S2OnOff_PSC_q[-1]
            R2Off_S2OnOff_PSC_q[-1] = R2Off_S2OnOff_PSC_q[-1]+dt*R2Off_S2OnOff_PSC_q_k1
            R2Off_R1Off_PSC_s[-2] = R2Off_R1Off_PSC_s[-1]
            R2Off_R1Off_PSC_s[-1] = R2Off_R1Off_PSC_s[-1]+dt*R2Off_R1Off_PSC_s_k1
            R2Off_R1Off_PSC_x[-2] = R2Off_R1Off_PSC_x[-1]
            R2Off_R1Off_PSC_x[-1] = R2Off_R1Off_PSC_x[-1]+dt*R2Off_R1Off_PSC_x_k1
            R2Off_R1Off_PSC_F[-2] = R2Off_R1Off_PSC_F[-1]
            R2Off_R1Off_PSC_F[-1] = R2Off_R1Off_PSC_F[-1]+dt*R2Off_R1Off_PSC_F_k1
            R2Off_R1Off_PSC_P[-2] = R2Off_R1Off_PSC_P[-1]
            R2Off_R1Off_PSC_P[-1] = R2Off_R1Off_PSC_P[-1]+dt*R2Off_R1Off_PSC_P_k1
            R2Off_R1Off_PSC_q[-2] = R2Off_R1Off_PSC_q[-1]
            R2Off_R1Off_PSC_q[-1] = R2Off_R1Off_PSC_q[-1]+dt*R2Off_R1Off_PSC_q_k1
            S2OnOff_R1Off_PSC_s[-2] = S2OnOff_R1Off_PSC_s[-1]
            S2OnOff_R1Off_PSC_s[-1] = S2OnOff_R1Off_PSC_s[-1]+dt*S2OnOff_R1Off_PSC_s_k1
            S2OnOff_R1Off_PSC_x[-2] = S2OnOff_R1Off_PSC_x[-1]
            S2OnOff_R1Off_PSC_x[-1] = S2OnOff_R1Off_PSC_x[-1]+dt*S2OnOff_R1Off_PSC_x_k1
            S2OnOff_R1Off_PSC_F[-2] = S2OnOff_R1Off_PSC_F[-1]
            S2OnOff_R1Off_PSC_F[-1] = S2OnOff_R1Off_PSC_F[-1]+dt*S2OnOff_R1Off_PSC_F_k1
            S2OnOff_R1Off_PSC_P[-2] = S2OnOff_R1Off_PSC_P[-1]
            S2OnOff_R1Off_PSC_P[-1] = S2OnOff_R1Off_PSC_P[-1]+dt*S2OnOff_R1Off_PSC_P_k1
            S2OnOff_R1Off_PSC_q[-2] = S2OnOff_R1Off_PSC_q[-1]
            S2OnOff_R1Off_PSC_q[-1] = S2OnOff_R1Off_PSC_q[-1]+dt*S2OnOff_R1Off_PSC_q_k1
            R2On_R2On_iNoise_V3_sn[-2] = R2On_R2On_iNoise_V3_sn[-1]
            R2On_R2On_iNoise_V3_sn[-1] = R2On_R2On_iNoise_V3_sn[-1]+dt*R2On_R2On_iNoise_V3_sn_k1
            R2On_R2On_iNoise_V3_xn[-2] = R2On_R2On_iNoise_V3_xn[-1]
            R2On_R2On_iNoise_V3_xn[-1] = R2On_R2On_iNoise_V3_xn[-1]+dt*R2On_R2On_iNoise_V3_xn_k1

            #Spiking and conditional actions
            On_V_spikes_holder.append(int(((On_V[-1] >= On_V_thresh) and (On_V[-2] < On_V_thresh))))
            if On_V_spikes_holder[-1]:
                On_tspike[int(On_buffer_index)-1] = helper[t]
                On_buffer_index = (On_buffer_index % 5) + 1
            Off_V_spikes_holder.append(int(((Off_V[-1] >= Off_V_thresh) and (Off_V[-2] < Off_V_thresh))))
            if Off_V_spikes_holder[-1]:
                Off_tspike[int(Off_buffer_index)-1] = helper[t]
                Off_buffer_index = (Off_buffer_index % 5) + 1
            R1On_V_spikes_holder.append(int(((R1On_V[-1] >= R1On_V_thresh) and (R1On_V[-2] < R1On_V_thresh))))
            if R1On_V_spikes_holder[-1]:
                R1On_tspike[int(R1On_buffer_index)-1] = helper[t]
                R1On_buffer_index = (R1On_buffer_index % 5) + 1
            R1Off_V_spikes_holder.append(int(((R1Off_V[-1] >= R1Off_V_thresh) and (R1Off_V[-2] < R1Off_V_thresh))))
            if R1Off_V_spikes_holder[-1]:
                R1Off_tspike[int(R1Off_buffer_index)-1] = helper[t]
                R1Off_buffer_index = (R1Off_buffer_index % 5) + 1
            S1OnOff_V_spikes_holder.append(int(((S1OnOff_V[-1] >= S1OnOff_V_thresh) and (S1OnOff_V[-2] < S1OnOff_V_thresh))))
            if S1OnOff_V_spikes_holder[-1]:
                S1OnOff_tspike[int(S1OnOff_buffer_index)-1] = helper[t]
                S1OnOff_buffer_index = (S1OnOff_buffer_index % 5) + 1
            R2On_V_spikes_holder.append(int(((R2On_V[-1] >= R2On_V_thresh) and (R2On_V[-2] < R2On_V_thresh))))
            if R2On_V_spikes_holder[-1]:
                R2On_tspike[int(R2On_buffer_index)-1] = helper[t]
                R2On_buffer_index = (R2On_buffer_index % 5) + 1
            R2Off_V_spikes_holder.append(int(((R2Off_V[-1] >= R2Off_V_thresh) and (R2Off_V[-2] < R2Off_V_thresh))))
            if R2Off_V_spikes_holder[-1]:
                R2Off_tspike[int(R2Off_buffer_index)-1] = helper[t]
                R2Off_buffer_index = (R2Off_buffer_index % 5) + 1
            S2OnOff_V_spikes_holder.append(int(((S2OnOff_V[-1] >= S2OnOff_V_thresh) and (S2OnOff_V[-2] < S2OnOff_V_thresh))))
            if S2OnOff_V_spikes_holder[-1]:
                S2OnOff_tspike[int(S2OnOff_buffer_index)-1] = helper[t]
                S2OnOff_buffer_index = (S2OnOff_buffer_index % 5) + 1

                #Voltage reset and adaptation
            On_V_test2a = On_V[-1] > On_V_thresh
            if On_V_test2a:
                On_V[-2] = On_V[-1] 
                On_V[-1] = On_V_reset 
                On_g_ad[-2] = On_g_ad[-1]
                On_g_ad[-1] = On_g_ad[-1] + On_g_inc
            On_V_test2b = np.any(helper[t] <= On_tspike + On_t_ref)
            if On_V_test2b:
                On_V[-2] = On_V[-1]
                On_V[-1] = On_V_reset
            Off_V_test2a = Off_V[-1] > Off_V_thresh
            if Off_V_test2a:
                Off_V[-2] = Off_V[-1] 
                Off_V[-1] = Off_V_reset 
                Off_g_ad[-2] = Off_g_ad[-1]
                Off_g_ad[-1] = Off_g_ad[-1] + Off_g_inc
            Off_V_test2b = np.any(helper[t] <= Off_tspike + Off_t_ref)
            if Off_V_test2b:
                Off_V[-2] = Off_V[-1]
                Off_V[-1] = Off_V_reset
            R1On_V_test2a = R1On_V[-1] > R1On_V_thresh
            if R1On_V_test2a:
                R1On_V[-2] = R1On_V[-1] 
                R1On_V[-1] = R1On_V_reset 
                R1On_g_ad[-2] = R1On_g_ad[-1]
                R1On_g_ad[-1] = R1On_g_ad[-1] + R1On_g_inc
            R1On_V_test2b = np.any(helper[t] <= R1On_tspike + R1On_t_ref)
            if R1On_V_test2b:
                R1On_V[-2] = R1On_V[-1]
                R1On_V[-1] = R1On_V_reset
            R1Off_V_test2a = R1Off_V[-1] > R1Off_V_thresh
            if R1Off_V_test2a:
                R1Off_V[-2] = R1Off_V[-1] 
                R1Off_V[-1] = R1Off_V_reset 
                R1Off_g_ad[-2] = R1Off_g_ad[-1]
                R1Off_g_ad[-1] = R1Off_g_ad[-1] + R1Off_g_inc
            R1Off_V_test2b = np.any(helper[t] <= R1Off_tspike + R1Off_t_ref)
            if R1Off_V_test2b:
                R1Off_V[-2] = R1Off_V[-1]
                R1Off_V[-1] = R1Off_V_reset
            S1OnOff_V_test2a = S1OnOff_V[-1] > S1OnOff_V_thresh
            if S1OnOff_V_test2a:
                S1OnOff_V[-2] = S1OnOff_V[-1] 
                S1OnOff_V[-1] = S1OnOff_V_reset 
                S1OnOff_g_ad[-2] = S1OnOff_g_ad[-1]
                S1OnOff_g_ad[-1] = S1OnOff_g_ad[-1] + S1OnOff_g_inc
            S1OnOff_V_test2b = np.any(helper[t] <= S1OnOff_tspike + S1OnOff_t_ref)
            if S1OnOff_V_test2b:
                S1OnOff_V[-2] = S1OnOff_V[-1]
                S1OnOff_V[-1] = S1OnOff_V_reset
            R2On_V_test2a = R2On_V[-1] > R2On_V_thresh
            if R2On_V_test2a:
                R2On_V[-2] = R2On_V[-1] 
                R2On_V[-1] = R2On_V_reset 
                R2On_g_ad[-2] = R2On_g_ad[-1]
                R2On_g_ad[-1] = R2On_g_ad[-1] + R2On_g_inc
            R2On_V_test2b = np.any(helper[t] <= R2On_tspike + R2On_t_ref)
            if R2On_V_test2b:
                R2On_V[-2] = R2On_V[-1]
                R2On_V[-1] = R2On_V_reset
            R2Off_V_test2a = R2Off_V[-1] > R2Off_V_thresh
            if R2Off_V_test2a:
                R2Off_V[-2] = R2Off_V[-1] 
                R2Off_V[-1] = R2Off_V_reset 
                R2Off_g_ad[-2] = R2Off_g_ad[-1]
                R2Off_g_ad[-1] = R2Off_g_ad[-1] + R2Off_g_inc
            R2Off_V_test2b = np.any(helper[t] <= R2Off_tspike + R2Off_t_ref)
            if R2Off_V_test2b:
                R2Off_V[-2] = R2Off_V[-1]
                R2Off_V[-1] = R2Off_V_reset
            S2OnOff_V_test2a = S2OnOff_V[-1] > S2OnOff_V_thresh
            if S2OnOff_V_test2a:
                S2OnOff_V[-2] = S2OnOff_V[-1] 
                S2OnOff_V[-1] = S2OnOff_V_reset 
                S2OnOff_g_ad[-2] = S2OnOff_g_ad[-1]
                S2OnOff_g_ad[-1] = S2OnOff_g_ad[-1] + S2OnOff_g_inc
            S2OnOff_V_test2b = np.any(helper[t] <= S2OnOff_tspike + S2OnOff_t_ref)
            if S2OnOff_V_test2b:
                S2OnOff_V[-2] = S2OnOff_V[-1]
                S2OnOff_V[-1] = S2OnOff_V_reset

            #Update PSC vars
            S2OnOff_V_test3 = np.any(helper[t] == On_tspike + R1On_On_PSC_delay)
            if S2OnOff_V_test3:
                R1On_On_PSC_x[-2] = R1On_On_PSC_x[-1]
                R1On_On_PSC_q[-2] = R1On_On_PSC_F[-1]
                R1On_On_PSC_F[-2] = R1On_On_PSC_F[-1]
                R1On_On_PSC_P[-2] = R1On_On_PSC_P[-1]
                R1On_On_PSC_x[-1] = R1On_On_PSC_x[-1] + R1On_On_PSC_q[-1]
                R1On_On_PSC_q[-1] = R1On_On_PSC_F[-1] * R1On_On_PSC_P[-1]
                R1On_On_PSC_F[-1] = R1On_On_PSC_F[-1] + R1On_On_PSC_fF*(R1On_On_PSC_maxF-R1On_On_PSC_F[-1])
                R1On_On_PSC_P[-1] = R1On_On_PSC_P[-1] * (1 - R1On_On_PSC_fP)
            R1On_On_PSC_de_test3 = np.any(helper[t] == On_tspike + S1OnOff_On_PSC_delay)
            if R1On_On_PSC_de_test3:
                S1OnOff_On_PSC_x[-2] = S1OnOff_On_PSC_x[-1]
                S1OnOff_On_PSC_q[-2] = S1OnOff_On_PSC_F[-1]
                S1OnOff_On_PSC_F[-2] = S1OnOff_On_PSC_F[-1]
                S1OnOff_On_PSC_P[-2] = S1OnOff_On_PSC_P[-1]
                S1OnOff_On_PSC_x[-1] = S1OnOff_On_PSC_x[-1] + S1OnOff_On_PSC_q[-1]
                S1OnOff_On_PSC_q[-1] = S1OnOff_On_PSC_F[-1] * S1OnOff_On_PSC_P[-1]
                S1OnOff_On_PSC_F[-1] = S1OnOff_On_PSC_F[-1] + S1OnOff_On_PSC_fF*(S1OnOff_On_PSC_maxF-S1OnOff_On_PSC_F[-1])
                S1OnOff_On_PSC_P[-1] = S1OnOff_On_PSC_P[-1] * (1 - S1OnOff_On_PSC_fP)
            S1OnOff_On_PSC_de_test3 = np.any(helper[t] == S1OnOff_tspike + R1On_S1OnOff_PSC_delay)
            if S1OnOff_On_PSC_de_test3:
                R1On_S1OnOff_PSC_x[-2] = R1On_S1OnOff_PSC_x[-1]
                R1On_S1OnOff_PSC_q[-2] = R1On_S1OnOff_PSC_F[-1]
                R1On_S1OnOff_PSC_F[-2] = R1On_S1OnOff_PSC_F[-1]
                R1On_S1OnOff_PSC_P[-2] = R1On_S1OnOff_PSC_P[-1]
                R1On_S1OnOff_PSC_x[-1] = R1On_S1OnOff_PSC_x[-1] + R1On_S1OnOff_PSC_q[-1]
                R1On_S1OnOff_PSC_q[-1] = R1On_S1OnOff_PSC_F[-1] * R1On_S1OnOff_PSC_P[-1]
                R1On_S1OnOff_PSC_F[-1] = R1On_S1OnOff_PSC_F[-1] + R1On_S1OnOff_PSC_fF*(R1On_S1OnOff_PSC_maxF-R1On_S1OnOff_PSC_F[-1])
                R1On_S1OnOff_PSC_P[-1] = R1On_S1OnOff_PSC_P[-1] * (1 - R1On_S1OnOff_PSC_fP)
            R1On_S1OnOff_PSC_de_test3 = np.any(helper[t] == S1OnOff_tspike + R1Off_S1OnOff_PSC_delay)
            if R1On_S1OnOff_PSC_de_test3:
                R1Off_S1OnOff_PSC_x[-2] = R1Off_S1OnOff_PSC_x[-1]
                R1Off_S1OnOff_PSC_q[-2] = R1Off_S1OnOff_PSC_F[-1]
                R1Off_S1OnOff_PSC_F[-2] = R1Off_S1OnOff_PSC_F[-1]
                R1Off_S1OnOff_PSC_P[-2] = R1Off_S1OnOff_PSC_P[-1]
                R1Off_S1OnOff_PSC_x[-1] = R1Off_S1OnOff_PSC_x[-1] + R1Off_S1OnOff_PSC_q[-1]
                R1Off_S1OnOff_PSC_q[-1] = R1Off_S1OnOff_PSC_F[-1] * R1Off_S1OnOff_PSC_P[-1]
                R1Off_S1OnOff_PSC_F[-1] = R1Off_S1OnOff_PSC_F[-1] + R1Off_S1OnOff_PSC_fF*(R1Off_S1OnOff_PSC_maxF-R1Off_S1OnOff_PSC_F[-1])
                R1Off_S1OnOff_PSC_P[-1] = R1Off_S1OnOff_PSC_P[-1] * (1 - R1Off_S1OnOff_PSC_fP)
            R1Off_S1OnOff_PSC_de_test3 = np.any(helper[t] == Off_tspike + R1Off_Off_PSC_delay)
            if R1Off_S1OnOff_PSC_de_test3:
                R1Off_Off_PSC_x[-2] = R1Off_Off_PSC_x[-1]
                R1Off_Off_PSC_q[-2] = R1Off_Off_PSC_F[-1]
                R1Off_Off_PSC_F[-2] = R1Off_Off_PSC_F[-1]
                R1Off_Off_PSC_P[-2] = R1Off_Off_PSC_P[-1]
                R1Off_Off_PSC_x[-1] = R1Off_Off_PSC_x[-1] + R1Off_Off_PSC_q[-1]
                R1Off_Off_PSC_q[-1] = R1Off_Off_PSC_F[-1] * R1Off_Off_PSC_P[-1]
                R1Off_Off_PSC_F[-1] = R1Off_Off_PSC_F[-1] + R1Off_Off_PSC_fF*(R1Off_Off_PSC_maxF-R1Off_Off_PSC_F[-1])
                R1Off_Off_PSC_P[-1] = R1Off_Off_PSC_P[-1] * (1 - R1Off_Off_PSC_fP)
            R1Off_Off_PSC_de_test3 = np.any(helper[t] == Off_tspike + S1OnOff_Off_PSC_delay)
            if R1Off_Off_PSC_de_test3:
                S1OnOff_Off_PSC_x[-2] = S1OnOff_Off_PSC_x[-1]
                S1OnOff_Off_PSC_q[-2] = S1OnOff_Off_PSC_F[-1]
                S1OnOff_Off_PSC_F[-2] = S1OnOff_Off_PSC_F[-1]
                S1OnOff_Off_PSC_P[-2] = S1OnOff_Off_PSC_P[-1]
                S1OnOff_Off_PSC_x[-1] = S1OnOff_Off_PSC_x[-1] + S1OnOff_Off_PSC_q[-1]
                S1OnOff_Off_PSC_q[-1] = S1OnOff_Off_PSC_F[-1] * S1OnOff_Off_PSC_P[-1]
                S1OnOff_Off_PSC_F[-1] = S1OnOff_Off_PSC_F[-1] + S1OnOff_Off_PSC_fF*(S1OnOff_Off_PSC_maxF-S1OnOff_Off_PSC_F[-1])
                S1OnOff_Off_PSC_P[-1] = S1OnOff_Off_PSC_P[-1] * (1 - S1OnOff_Off_PSC_fP)
            S1OnOff_Off_PSC_de_test3 = np.any(helper[t] == R1On_tspike + R2On_R1On_PSC_delay)
            if S1OnOff_Off_PSC_de_test3:
                R2On_R1On_PSC_x[-2] = R2On_R1On_PSC_x[-1]
                R2On_R1On_PSC_q[-2] = R2On_R1On_PSC_F[-1]
                R2On_R1On_PSC_F[-2] = R2On_R1On_PSC_F[-1]
                R2On_R1On_PSC_P[-2] = R2On_R1On_PSC_P[-1]
                R2On_R1On_PSC_x[-1] = R2On_R1On_PSC_x[-1] + R2On_R1On_PSC_q[-1]
                R2On_R1On_PSC_q[-1] = R2On_R1On_PSC_F[-1] * R2On_R1On_PSC_P[-1]
                R2On_R1On_PSC_F[-1] = R2On_R1On_PSC_F[-1] + R2On_R1On_PSC_fF*(R2On_R1On_PSC_maxF-R2On_R1On_PSC_F[-1])
                R2On_R1On_PSC_P[-1] = R2On_R1On_PSC_P[-1] * (1 - R2On_R1On_PSC_fP)
            R2On_R1On_PSC_de_test3 = np.any(helper[t] == R1On_tspike + S2OnOff_R1On_PSC_delay)
            if R2On_R1On_PSC_de_test3:
                S2OnOff_R1On_PSC_x[-2] = S2OnOff_R1On_PSC_x[-1]
                S2OnOff_R1On_PSC_q[-2] = S2OnOff_R1On_PSC_F[-1]
                S2OnOff_R1On_PSC_F[-2] = S2OnOff_R1On_PSC_F[-1]
                S2OnOff_R1On_PSC_P[-2] = S2OnOff_R1On_PSC_P[-1]
                S2OnOff_R1On_PSC_x[-1] = S2OnOff_R1On_PSC_x[-1] + S2OnOff_R1On_PSC_q[-1]
                S2OnOff_R1On_PSC_q[-1] = S2OnOff_R1On_PSC_F[-1] * S2OnOff_R1On_PSC_P[-1]
                S2OnOff_R1On_PSC_F[-1] = S2OnOff_R1On_PSC_F[-1] + S2OnOff_R1On_PSC_fF*(S2OnOff_R1On_PSC_maxF-S2OnOff_R1On_PSC_F[-1])
                S2OnOff_R1On_PSC_P[-1] = S2OnOff_R1On_PSC_P[-1] * (1 - S2OnOff_R1On_PSC_fP)
            S2OnOff_R1On_PSC_de_test3 = np.any(helper[t] == S2OnOff_tspike + R2On_S2OnOff_PSC_delay)
            if S2OnOff_R1On_PSC_de_test3:
                R2On_S2OnOff_PSC_x[-2] = R2On_S2OnOff_PSC_x[-1]
                R2On_S2OnOff_PSC_q[-2] = R2On_S2OnOff_PSC_F[-1]
                R2On_S2OnOff_PSC_F[-2] = R2On_S2OnOff_PSC_F[-1]
                R2On_S2OnOff_PSC_P[-2] = R2On_S2OnOff_PSC_P[-1]
                R2On_S2OnOff_PSC_x[-1] = R2On_S2OnOff_PSC_x[-1] + R2On_S2OnOff_PSC_q[-1]
                R2On_S2OnOff_PSC_q[-1] = R2On_S2OnOff_PSC_F[-1] * R2On_S2OnOff_PSC_P[-1]
                R2On_S2OnOff_PSC_F[-1] = R2On_S2OnOff_PSC_F[-1] + R2On_S2OnOff_PSC_fF*(R2On_S2OnOff_PSC_maxF-R2On_S2OnOff_PSC_F[-1])
                R2On_S2OnOff_PSC_P[-1] = R2On_S2OnOff_PSC_P[-1] * (1 - R2On_S2OnOff_PSC_fP)
            R2On_S2OnOff_PSC_de_test3 = np.any(helper[t] == S2OnOff_tspike + R2Off_S2OnOff_PSC_delay)
            if R2On_S2OnOff_PSC_de_test3:
                R2Off_S2OnOff_PSC_x[-2] = R2Off_S2OnOff_PSC_x[-1]
                R2Off_S2OnOff_PSC_q[-2] = R2Off_S2OnOff_PSC_F[-1]
                R2Off_S2OnOff_PSC_F[-2] = R2Off_S2OnOff_PSC_F[-1]
                R2Off_S2OnOff_PSC_P[-2] = R2Off_S2OnOff_PSC_P[-1]
                R2Off_S2OnOff_PSC_x[-1] = R2Off_S2OnOff_PSC_x[-1] + R2Off_S2OnOff_PSC_q[-1]
                R2Off_S2OnOff_PSC_q[-1] = R2Off_S2OnOff_PSC_F[-1] * R2Off_S2OnOff_PSC_P[-1]
                R2Off_S2OnOff_PSC_F[-1] = R2Off_S2OnOff_PSC_F[-1] + R2Off_S2OnOff_PSC_fF*(R2Off_S2OnOff_PSC_maxF-R2Off_S2OnOff_PSC_F[-1])
                R2Off_S2OnOff_PSC_P[-1] = R2Off_S2OnOff_PSC_P[-1] * (1 - R2Off_S2OnOff_PSC_fP)
            R2Off_S2OnOff_PSC_de_test3 = np.any(helper[t] == R1Off_tspike + R2Off_R1Off_PSC_delay)
            if R2Off_S2OnOff_PSC_de_test3:
                R2Off_R1Off_PSC_x[-2] = R2Off_R1Off_PSC_x[-1]
                R2Off_R1Off_PSC_q[-2] = R2Off_R1Off_PSC_F[-1]
                R2Off_R1Off_PSC_F[-2] = R2Off_R1Off_PSC_F[-1]
                R2Off_R1Off_PSC_P[-2] = R2Off_R1Off_PSC_P[-1]
                R2Off_R1Off_PSC_x[-1] = R2Off_R1Off_PSC_x[-1] + R2Off_R1Off_PSC_q[-1]
                R2Off_R1Off_PSC_q[-1] = R2Off_R1Off_PSC_F[-1] * R2Off_R1Off_PSC_P[-1]
                R2Off_R1Off_PSC_F[-1] = R2Off_R1Off_PSC_F[-1] + R2Off_R1Off_PSC_fF*(R2Off_R1Off_PSC_maxF-R2Off_R1Off_PSC_F[-1])
                R2Off_R1Off_PSC_P[-1] = R2Off_R1Off_PSC_P[-1] * (1 - R2Off_R1Off_PSC_fP)
            R2Off_R1Off_PSC_de_test3 = np.any(helper[t] == R1Off_tspike + S2OnOff_R1Off_PSC_delay)
            if R2Off_R1Off_PSC_de_test3:
                S2OnOff_R1Off_PSC_x[-2] = S2OnOff_R1Off_PSC_x[-1]
                S2OnOff_R1Off_PSC_q[-2] = S2OnOff_R1Off_PSC_F[-1]
                S2OnOff_R1Off_PSC_F[-2] = S2OnOff_R1Off_PSC_F[-1]
                S2OnOff_R1Off_PSC_P[-2] = S2OnOff_R1Off_PSC_P[-1]
                S2OnOff_R1Off_PSC_x[-1] = S2OnOff_R1Off_PSC_x[-1] + S2OnOff_R1Off_PSC_q[-1]
                S2OnOff_R1Off_PSC_q[-1] = S2OnOff_R1Off_PSC_F[-1] * S2OnOff_R1Off_PSC_P[-1]
                S2OnOff_R1Off_PSC_F[-1] = S2OnOff_R1Off_PSC_F[-1] + S2OnOff_R1Off_PSC_fF*(S2OnOff_R1Off_PSC_maxF-S2OnOff_R1Off_PSC_F[-1])
                S2OnOff_R1Off_PSC_P[-1] = S2OnOff_R1Off_PSC_P[-1] * (1 - S2OnOff_R1Off_PSC_fP)

            #Grad Calculations


            #Surrogate Spike Related Derivates
            dspike_dR1On_V = (((10*np.exp(-(0.1)*(R1On_V[-1] - R1On_V_thresh)))/(1+np.exp(-(0.1)*(R1On_V[-1] - R1On_V_thresh)))**2))/500
            dspike_dR1Off_V = (((10*np.exp(-(0.1)*(R1Off_V[-1] - R1Off_V_thresh)))/(1+np.exp(-(0.1)*(R1Off_V[-1] - R1Off_V_thresh)))**2))/500
            dspike_dS1OnOff_V = (((10*np.exp(-(0.1)*(S1OnOff_V[-1] - S1OnOff_V_thresh)))/(1+np.exp(-(0.1)*(S1OnOff_V[-1] - S1OnOff_V_thresh)))**2))/500
            dspike_dR2On_V = (((10*np.exp(-(0.1)*(R2On_V[-1] - R2On_V_thresh)))/(1+np.exp(-(0.1)*(R2On_V[-1] - R2On_V_thresh)))**2))/500
            dspike_dS2OnOff_V = (((10*np.exp(-(0.1)*(S2OnOff_V[-1] - S2OnOff_V_thresh)))/(1+np.exp(-(0.1)*(S2OnOff_V[-1] - S2OnOff_V_thresh)))**2))/500


            #PSC & Parameter Related Derivates
            dv_dR1On_On_PSC_gSYN = -(dt*R1On_R*R1On_On_PSC_s[-1]*R1On_On_PSC_netcon*(R1On_V[-1]-R1On_On_PSC_ESYN)/R1On_tau)/15
            dR1On_On_PSC_dUk = -((dt*R1On_On_PSC_scale*2*(R1On_On_PSC_x[-1]+R1On_On_PSC_q[-1])/R1On_On_PSC_tauR)*helper[t]*sum(((On_tspike+R1On_On_PSC_delay)-helper[t])*np.exp(-1*((On_tspike+R1On_On_PSC_delay)-helper[t])**2)))/2500
            dv_dR1On_On_PSC = -(dt*R1On_R*R1On_On_PSC_gSYN*R1On_On_PSC_netcon*(R1On_V[-1]-R1On_On_PSC_ESYN)/R1On_tau)/10
            dv_dS1OnOff_On_PSC_gSYN = -(dt*S1OnOff_R*S1OnOff_On_PSC_s[-1]*S1OnOff_On_PSC_netcon*(S1OnOff_V[-1]-S1OnOff_On_PSC_ESYN)/S1OnOff_tau)/15
            dS1OnOff_On_PSC_dUk = -((dt*S1OnOff_On_PSC_scale*2*(S1OnOff_On_PSC_x[-1]+S1OnOff_On_PSC_q[-1])/S1OnOff_On_PSC_tauR)*helper[t]*sum(((On_tspike+S1OnOff_On_PSC_delay)-helper[t])*np.exp(-1*((On_tspike+S1OnOff_On_PSC_delay)-helper[t])**2)))/2500
            dv_dS1OnOff_On_PSC = -(dt*S1OnOff_R*S1OnOff_On_PSC_gSYN*S1OnOff_On_PSC_netcon*(S1OnOff_V[-1]-S1OnOff_On_PSC_ESYN)/S1OnOff_tau)/10
            dv_dR1On_S1OnOff_PSC_gSYN = -(dt*R1On_R*R1On_S1OnOff_PSC_s[-1]*R1On_S1OnOff_PSC_netcon*(R1On_V[-1]-R1On_S1OnOff_PSC_ESYN)/R1On_tau)/15
            dR1On_S1OnOff_PSC_dUk = -((dt*R1On_S1OnOff_PSC_scale*2*(R1On_S1OnOff_PSC_x[-1]+R1On_S1OnOff_PSC_q[-1])/R1On_S1OnOff_PSC_tauR)*helper[t]*sum(((S1OnOff_tspike+R1On_S1OnOff_PSC_delay)-helper[t])*np.exp(-1*((S1OnOff_tspike+R1On_S1OnOff_PSC_delay)-helper[t])**2)))/2500
            dv_dR1On_S1OnOff_PSC = -(dt*R1On_R*R1On_S1OnOff_PSC_gSYN*R1On_S1OnOff_PSC_netcon*(R1On_V[-1]-R1On_S1OnOff_PSC_ESYN)/R1On_tau)/10
            dv_dR1Off_S1OnOff_PSC_gSYN = -(dt*R1Off_R*R1Off_S1OnOff_PSC_s[-1]*R1Off_S1OnOff_PSC_netcon*(R1Off_V[-1]-R1Off_S1OnOff_PSC_ESYN)/R1Off_tau)/15
            dR1Off_S1OnOff_PSC_dUk = -((dt*R1Off_S1OnOff_PSC_scale*2*(R1Off_S1OnOff_PSC_x[-1]+R1Off_S1OnOff_PSC_q[-1])/R1Off_S1OnOff_PSC_tauR)*helper[t]*sum(((S1OnOff_tspike+R1Off_S1OnOff_PSC_delay)-helper[t])*np.exp(-1*((S1OnOff_tspike+R1Off_S1OnOff_PSC_delay)-helper[t])**2)))/2500
            dv_dR1Off_S1OnOff_PSC = -(dt*R1Off_R*R1Off_S1OnOff_PSC_gSYN*R1Off_S1OnOff_PSC_netcon*(R1Off_V[-1]-R1Off_S1OnOff_PSC_ESYN)/R1Off_tau)/10
            dv_dR1Off_Off_PSC_gSYN = -(dt*R1Off_R*R1Off_Off_PSC_s[-1]*R1Off_Off_PSC_netcon*(R1Off_V[-1]-R1Off_Off_PSC_ESYN)/R1Off_tau)/15
            dR1Off_Off_PSC_dUk = -((dt*R1Off_Off_PSC_scale*2*(R1Off_Off_PSC_x[-1]+R1Off_Off_PSC_q[-1])/R1Off_Off_PSC_tauR)*helper[t]*sum(((Off_tspike+R1Off_Off_PSC_delay)-helper[t])*np.exp(-1*((Off_tspike+R1Off_Off_PSC_delay)-helper[t])**2)))/2500
            dv_dR1Off_Off_PSC = -(dt*R1Off_R*R1Off_Off_PSC_gSYN*R1Off_Off_PSC_netcon*(R1Off_V[-1]-R1Off_Off_PSC_ESYN)/R1Off_tau)/10
            dv_dS1OnOff_Off_PSC_gSYN = -(dt*S1OnOff_R*S1OnOff_Off_PSC_s[-1]*S1OnOff_Off_PSC_netcon*(S1OnOff_V[-1]-S1OnOff_Off_PSC_ESYN)/S1OnOff_tau)/15
            dS1OnOff_Off_PSC_dUk = -((dt*S1OnOff_Off_PSC_scale*2*(S1OnOff_Off_PSC_x[-1]+S1OnOff_Off_PSC_q[-1])/S1OnOff_Off_PSC_tauR)*helper[t]*sum(((Off_tspike+S1OnOff_Off_PSC_delay)-helper[t])*np.exp(-1*((Off_tspike+S1OnOff_Off_PSC_delay)-helper[t])**2)))/2500
            dv_dS1OnOff_Off_PSC = -(dt*S1OnOff_R*S1OnOff_Off_PSC_gSYN*S1OnOff_Off_PSC_netcon*(S1OnOff_V[-1]-S1OnOff_Off_PSC_ESYN)/S1OnOff_tau)/10
            dv_dR2On_R1On_PSC_gSYN = -(dt*R2On_R*R2On_R1On_PSC_s[-1]*R2On_R1On_PSC_netcon*(R2On_V[-1]-R2On_R1On_PSC_ESYN)/R2On_tau)/15
            dR2On_R1On_PSC_dUk = -((dt*R2On_R1On_PSC_scale*2*(R2On_R1On_PSC_x[-1]+R2On_R1On_PSC_q[-1])/R2On_R1On_PSC_tauR)*helper[t]*sum(((R1On_tspike+R2On_R1On_PSC_delay)-helper[t])*np.exp(-1*((R1On_tspike+R2On_R1On_PSC_delay)-helper[t])**2)))/2500
            dv_dR2On_R1On_PSC = -(dt*R2On_R*R2On_R1On_PSC_gSYN*R2On_R1On_PSC_netcon*(R2On_V[-1]-R2On_R1On_PSC_ESYN)/R2On_tau)/10
            dv_dS2OnOff_R1On_PSC_gSYN = -(dt*S2OnOff_R*S2OnOff_R1On_PSC_s[-1]*S2OnOff_R1On_PSC_netcon*(S2OnOff_V[-1]-S2OnOff_R1On_PSC_ESYN)/S2OnOff_tau)/15
            dS2OnOff_R1On_PSC_dUk = -((dt*S2OnOff_R1On_PSC_scale*2*(S2OnOff_R1On_PSC_x[-1]+S2OnOff_R1On_PSC_q[-1])/S2OnOff_R1On_PSC_tauR)*helper[t]*sum(((R1On_tspike+S2OnOff_R1On_PSC_delay)-helper[t])*np.exp(-1*((R1On_tspike+S2OnOff_R1On_PSC_delay)-helper[t])**2)))/2500
            dv_dS2OnOff_R1On_PSC = -(dt*S2OnOff_R*S2OnOff_R1On_PSC_gSYN*S2OnOff_R1On_PSC_netcon*(S2OnOff_V[-1]-S2OnOff_R1On_PSC_ESYN)/S2OnOff_tau)/10
            dv_dR2On_S2OnOff_PSC_gSYN = -(dt*R2On_R*R2On_S2OnOff_PSC_s[-1]*R2On_S2OnOff_PSC_netcon*(R2On_V[-1]-R2On_S2OnOff_PSC_ESYN)/R2On_tau)/15
            dR2On_S2OnOff_PSC_dUk = -((dt*R2On_S2OnOff_PSC_scale*2*(R2On_S2OnOff_PSC_x[-1]+R2On_S2OnOff_PSC_q[-1])/R2On_S2OnOff_PSC_tauR)*helper[t]*sum(((S2OnOff_tspike+R2On_S2OnOff_PSC_delay)-helper[t])*np.exp(-1*((S2OnOff_tspike+R2On_S2OnOff_PSC_delay)-helper[t])**2)))/2500
            dv_dR2On_S2OnOff_PSC = -(dt*R2On_R*R2On_S2OnOff_PSC_gSYN*R2On_S2OnOff_PSC_netcon*(R2On_V[-1]-R2On_S2OnOff_PSC_ESYN)/R2On_tau)/10
            dv_dS2OnOff_R1Off_PSC_gSYN = -(dt*S2OnOff_R*S2OnOff_R1Off_PSC_s[-1]*S2OnOff_R1Off_PSC_netcon*(S2OnOff_V[-1]-S2OnOff_R1Off_PSC_ESYN)/S2OnOff_tau)/15
            dS2OnOff_R1Off_PSC_dUk = -((dt*S2OnOff_R1Off_PSC_scale*2*(S2OnOff_R1Off_PSC_x[-1]+S2OnOff_R1Off_PSC_q[-1])/S2OnOff_R1Off_PSC_tauR)*helper[t]*sum(((R1Off_tspike+S2OnOff_R1Off_PSC_delay)-helper[t])*np.exp(-1*((R1Off_tspike+S2OnOff_R1Off_PSC_delay)-helper[t])**2)))/2500
            dv_dS2OnOff_R1Off_PSC = -(dt*S2OnOff_R*S2OnOff_R1Off_PSC_gSYN*S2OnOff_R1Off_PSC_netcon*(S2OnOff_V[-1]-S2OnOff_R1Off_PSC_ESYN)/S2OnOff_tau)/10

            #Build derivs
            dGSYNR1On_On += dspike_dR2On_V*dv_dR2On_R1On_PSC*dR2On_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_On_PSC_gSYN+dspike_dR2On_V*dv_dR2On_S2OnOff_PSC*dR2On_S2OnOff_PSC_dUk*dspike_dS2OnOff_V*dv_dS2OnOff_R1On_PSC*dS2OnOff_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_On_PSC_gSYN
            dGSYNS1OnOff_On += dspike_dR2On_V*dv_dR2On_R1On_PSC*dR2On_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_S1OnOff_PSC*dR1On_S1OnOff_PSC_dUk*dspike_dS1OnOff_V*dv_dS1OnOff_On_PSC_gSYN+dspike_dR2On_V*dv_dR2On_S2OnOff_PSC*dR2On_S2OnOff_PSC_dUk*dspike_dS2OnOff_V*dv_dS2OnOff_R1Off_PSC*dS2OnOff_R1Off_PSC_dUk*dspike_dR1Off_V*dv_dR1Off_S1OnOff_PSC*dR1Off_S1OnOff_PSC_dUk*dspike_dS1OnOff_V*dv_dS1OnOff_On_PSC_gSYN+dspike_dR2On_V*dv_dR2On_S2OnOff_PSC*dR2On_S2OnOff_PSC_dUk*dspike_dS2OnOff_V*dv_dS2OnOff_R1On_PSC*dS2OnOff_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_S1OnOff_PSC*dR1On_S1OnOff_PSC_dUk*dspike_dS1OnOff_V*dv_dS1OnOff_On_PSC_gSYN
            dGSYNR1On_S1OnOff += dspike_dR2On_V*dv_dR2On_R1On_PSC*dR2On_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_S1OnOff_PSC_gSYN+dspike_dR2On_V*dv_dR2On_S2OnOff_PSC*dR2On_S2OnOff_PSC_dUk*dspike_dS2OnOff_V*dv_dS2OnOff_R1On_PSC*dS2OnOff_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_S1OnOff_PSC_gSYN
            dGSYNR1Off_S1OnOff += dspike_dR2On_V*dv_dR2On_S2OnOff_PSC*dR2On_S2OnOff_PSC_dUk*dspike_dS2OnOff_V*dv_dS2OnOff_R1Off_PSC*dS2OnOff_R1Off_PSC_dUk*dspike_dR1Off_V*dv_dR1Off_S1OnOff_PSC_gSYN
            dGSYNR1Off_Off += dspike_dR2On_V*dv_dR2On_S2OnOff_PSC*dR2On_S2OnOff_PSC_dUk*dspike_dS2OnOff_V*dv_dS2OnOff_R1Off_PSC*dS2OnOff_R1Off_PSC_dUk*dspike_dR1Off_V*dv_dR1Off_Off_PSC_gSYN
            dGSYNS1OnOff_Off += dspike_dR2On_V*dv_dR2On_R1On_PSC*dR2On_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_S1OnOff_PSC*dR1On_S1OnOff_PSC_dUk*dspike_dS1OnOff_V*dv_dS1OnOff_Off_PSC_gSYN+dspike_dR2On_V*dv_dR2On_S2OnOff_PSC*dR2On_S2OnOff_PSC_dUk*dspike_dS2OnOff_V*dv_dS2OnOff_R1Off_PSC*dS2OnOff_R1Off_PSC_dUk*dspike_dR1Off_V*dv_dR1Off_S1OnOff_PSC*dR1Off_S1OnOff_PSC_dUk*dspike_dS1OnOff_V*dv_dS1OnOff_Off_PSC_gSYN+dspike_dR2On_V*dv_dR2On_S2OnOff_PSC*dR2On_S2OnOff_PSC_dUk*dspike_dS2OnOff_V*dv_dS2OnOff_R1On_PSC*dS2OnOff_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_S1OnOff_PSC*dR1On_S1OnOff_PSC_dUk*dspike_dS1OnOff_V*dv_dS1OnOff_Off_PSC_gSYN
            dGSYNR2On_R1On += dspike_dR2On_V*dv_dR2On_R1On_PSC_gSYN
            dGSYNS2OnOff_R1On += dspike_dR2On_V*dv_dR2On_S2OnOff_PSC*dR2On_S2OnOff_PSC_dUk*dspike_dS2OnOff_V*dv_dS2OnOff_R1On_PSC_gSYN
            dGSYNR2On_S2OnOff += dspike_dR2On_V*dv_dR2On_S2OnOff_PSC_gSYN
            dGSYNS2OnOff_R1Off += dspike_dR2On_V*dv_dR2On_S2OnOff_PSC*dR2On_S2OnOff_PSC_dUk*dspike_dS2OnOff_V*dv_dS2OnOff_R1Off_PSC_gSYN

        #Append Spikes
        On_V_spikes.append(On_V_spikes_holder)
        Off_V_spikes.append(Off_V_spikes_holder)
        R1On_V_spikes.append(R1On_V_spikes_holder)
        R1Off_V_spikes.append(R1Off_V_spikes_holder)
        S1OnOff_V_spikes.append(S1OnOff_V_spikes_holder)
        R2On_V_spikes.append(R2On_V_spikes_holder)
        R2Off_V_spikes.append(R2Off_V_spikes_holder)
        S2OnOff_V_spikes.append(S2OnOff_V_spikes_holder)

    return R2On_V_spikes, [dGSYNR1On_On, dGSYNS1OnOff_On, dGSYNR1On_S1OnOff, dGSYNR1Off_S1OnOff, dGSYNR1Off_Off, dGSYNS1OnOff_Off, dGSYNR2On_R1On, dGSYNS2OnOff_R1On, dGSYNR2On_S2OnOff, dGSYNS2OnOff_R1Off]

def main():
        num_epochs = 1

        p = np.array([1,1,1,1,1,1,1,1,1,1])*0.025   #Initial parameter value

        #Adam
        m = np.zeros((10))
        v = np.zeros((10))
        beta1, beta2 = 0.92, 0.9995
        eps = 1e-6
        t = 0
        lr = 1e-3

        scale_factor = 0.2


        matfile_path = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting"
        filename = f"{matfile_path}/goalPSTH.mat"
        data = scipy.io.loadmat(filename)


        #target_spikes = np.array(data['ans'][0])
        target_spikes = 100

        losses = []
        param_tracker = []

        for epoch in range(num_epochs):

            #print('here')

            output, grads = forwards(p,scale_factor)  # forward pass

            grad_holder2 = []
            for z in grads:
                #print(z[0])
                grad_holder2.append(z[0][0])

            grads = grad_holder2

            grads = [float(x) for x in grads] 

            param_tracker.append(p)

            print(f'parameter = {p}')

            print(np.shape(output))
            output = np.reshape(output,(1,int((35000*scale_factor-2)*10)))

            print('Avg Firing Rate')
            print(output.sum()/10/(3*scale_factor))

            fr = output.sum()/10/(3*scale_factor)  #total spikes/num_trials/num_seconds

            #print(target_spikes)

            loss = (target_spikes-fr)**2

            print(float(2*(fr-target_spikes)))
            print(grads)




            #out_grad = float(2*(fr-target_spikes))*grads

            #A negetive comes out here because we are doing dL/dspikes and the inner derivactive of -x is -1
            #Target spikes is not what is being calculated. It is actually fr. So no negetive actually
            scale = float(2*(fr - target_spikes))
            out_grad = [scale * g for g in grads] 

            print('grad below')
            print(out_grad)

            #p = Update_Grads.grads_update(grads,p)


            t += 1

            print('here7')
            print(m)
            print(len(m))
            print(np.shape(m))

            m = [beta1*m[ms] + (1-beta1) * out_grad[ms] for ms in range(len(m))]
            v = [beta2*v[vs] + (1-beta2) * (out_grad[vs]**2) for vs in range(len(v))]

            #m = beta1 * m + (1 - beta1) * out_grad
            #v = beta2 * v + (1 - beta2) * (out_grad ** 2)

            print('m')
            print(m)

            m_hat = [m[ms]/(1 - beta1 ** t) for ms in range(len(m))]
            v_hat = [v[vs]/(1 - beta2 ** t) for vs in range(len(v))]

            #m_hat = m / (1 - beta1 ** t)
            #v_hat = v / (1 - beta2 ** t)

            print('v_hat')
            print(v_hat)

            p = [p[vs] - lr*m_hat[vs]/(np.sqrt(v_hat[vs]) + eps) for vs in range(len(v))]

            #p = p - lr * m_hat / (np.sqrt(v_hat) + eps)

            #print('p')
            #print(p)



            print('p below')
            print(p)

            #loss = ((binned_counts - target_spikes)**2).mean()
            losses.append(loss)


            print(f"Epoch {epoch}: Loss = {loss.item()}",flush=True) 

        return losses, output, param_tracker


