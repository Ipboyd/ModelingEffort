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
    R1On_On_PSC_gSYN = ps[0]
    S1OnOff_On_PSC_gSYN = ps[1]
    R1On_S1OnOff_PSC_gSYN = ps[2]
    S1OnOff_Off_PSC_gSYN = ps[3]
    On_t_ref = ps[4]
    Off_t_ref = ps[5]
    R1On_t_ref = ps[6]
    S1OnOff_t_ref = ps[7]
    On_On_IC_g_postIC = ps[8]
    Off_Off_IC_g_postIC = ps[9]
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
    S1OnOff_E_k = -80
    S1OnOff_tau_ad = 5
    S1OnOff_g_inc = 0
    S1OnOff_Itonic = 0
    S1OnOff_V_thresh = -47
    S1OnOff_V_reset = -52
    S1OnOff_Npop = 1
    On_On_IC_trial = 20
    On_On_IC_locNum = 15
    On_On_IC_label = 'on'
    On_On_IC_t_ref = 1
    On_On_IC_t_ref_rel = 1
    On_On_IC_rec = 2
    On_On_IC_E_exc = 0
    Off_Off_IC_trial = 20
    Off_Off_IC_locNum = 15
    Off_Off_IC_label = 'off'
    Off_Off_IC_t_ref = 1
    Off_Off_IC_t_ref_rel = 1
    Off_Off_IC_rec = 2
    Off_Off_IC_E_exc = 0
    R1On_On_PSC_ESYN = 0
    R1On_On_PSC_tauD = 1.5
    R1On_On_PSC_tauR = 0.7
    R1On_On_PSC_delay = 0
    R1On_On_PSC_fF = 0
    R1On_On_PSC_fP = 0.1
    R1On_On_PSC_tauF = 180
    R1On_On_PSC_tauP = 30
    R1On_On_PSC_maxF = 4
    S1OnOff_On_PSC_ESYN = 0
    S1OnOff_On_PSC_tauD = 1
    S1OnOff_On_PSC_tauR = 0.1
    S1OnOff_On_PSC_delay = 0
    S1OnOff_On_PSC_fF = 0
    S1OnOff_On_PSC_fP = 0.2
    S1OnOff_On_PSC_tauF = 180
    S1OnOff_On_PSC_tauP = 80
    S1OnOff_On_PSC_maxF = 4
    R1On_S1OnOff_PSC_ESYN = -80
    R1On_S1OnOff_PSC_tauD = 4.5
    R1On_S1OnOff_PSC_tauR = 1
    R1On_S1OnOff_PSC_delay = 0
    R1On_S1OnOff_PSC_fF = 0
    R1On_S1OnOff_PSC_fP = 0.5
    R1On_S1OnOff_PSC_tauF = 180
    R1On_S1OnOff_PSC_tauP = 120
    R1On_S1OnOff_PSC_maxF = 4
    R1Off_S1OnOff_PSC_ESYN = -80
    R1Off_S1OnOff_PSC_tauD = 4.5
    R1Off_S1OnOff_PSC_tauR = 1
    R1Off_S1OnOff_PSC_delay = 0
    R1Off_S1OnOff_PSC_gSYN = 0.025
    R1Off_S1OnOff_PSC_fF = 0
    R1Off_S1OnOff_PSC_fP = 0.5
    R1Off_S1OnOff_PSC_tauF = 180
    R1Off_S1OnOff_PSC_tauP = 120
    R1Off_S1OnOff_PSC_maxF = 4
    R1Off_Off_PSC_ESYN = 0
    R1Off_Off_PSC_tauD = 1.5
    R1Off_Off_PSC_tauR = 0.7
    R1Off_Off_PSC_delay = 0
    R1Off_Off_PSC_gSYN = 0.02
    R1Off_Off_PSC_fF = 0
    R1Off_Off_PSC_fP = 0.1
    R1Off_Off_PSC_tauF = 180
    R1Off_Off_PSC_tauP = 30
    R1Off_Off_PSC_maxF = 4
    S1OnOff_Off_PSC_ESYN = 0
    S1OnOff_Off_PSC_tauD = 1
    S1OnOff_Off_PSC_tauR = 0.1
    S1OnOff_Off_PSC_delay = 0
    S1OnOff_Off_PSC_fF = 0
    S1OnOff_Off_PSC_fP = 0
    S1OnOff_Off_PSC_tauF = 180
    S1OnOff_Off_PSC_tauP = 80
    S1OnOff_Off_PSC_maxF = 4
    R1On_R1On_iNoise_V3_FR = 8
    R1On_R1On_iNoise_V3_sigma = 0
    R1On_R1On_iNoise_V3_dt = 0.1
    R1On_R1On_iNoise_V3_nSYN = 0.015
    R1On_R1On_iNoise_V3_simlen = 29801
    R1On_R1On_iNoise_V3_tauD_N = 1.5
    R1On_R1On_iNoise_V3_tauR_N = 0.7
    R1On_R1On_iNoise_V3_E_exc = 0
    ROn_X_PSC3_netcon = 1
    ROn_SOnOff_PSC3_netcon = 1
    C_ROn_PSC3_netcon = 1
    dv1dOn_V_holder = []
    dv1dOff_V_holder = []
    dv1dR1On_V_holder = []
    dv1dS1OnOff_V_holder = []
    dGSYNR1On_On = np.zeros((400))
    dGSYNS1OnOff_On = np.zeros((400))
    dGSYNR1On_S1OnOff = np.zeros((400))
    dGSYNS1OnOff_Off = np.zeros((400))

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
    R1On_R1On_iNoise_V3_netcon = np.eye(R1On_Npop, R1On_Npop)
    R1On_R1On_iNoise_V3_token = genPoissonTimes.gen_poisson_times(R1On_Npop,R1On_R1On_iNoise_V3_dt,R1On_R1On_iNoise_V3_FR,R1On_R1On_iNoise_V3_sigma,R1On_R1On_iNoise_V3_simlen)
    R1On_R1On_iNoise_V3_scale = (R1On_R1On_iNoise_V3_tauD_N/R1On_R1On_iNoise_V3_tauR_N)**(R1On_R1On_iNoise_V3_tauR_N/(R1On_R1On_iNoise_V3_tauD_N-R1On_R1On_iNoise_V3_tauR_N))

    T = len(np.arange(tspan[0],tspan[1]+(dt),dt))
    helper = np.arange(tspan[0],tspan[1]+(dt),dt)

    #State Variable Declaration
    On_V = np.ones((400,2)) * [On_E_L, On_E_L]
    On_g_ad = np.ones((400,2)) * [0,0]
    Off_V = np.ones((400,2)) * [Off_E_L, Off_E_L]
    Off_g_ad = np.ones((400,2)) * [0,0]
    R1On_V = np.ones((400,2)) * [R1On_E_L, R1On_E_L]
    R1On_g_ad = np.ones((400,2)) * [0,0]
    R1Off_V = np.ones((400,2)) * [R1Off_E_L, R1Off_E_L]
    R1Off_g_ad = np.ones((400,2)) * [0,0]
    S1OnOff_V = np.ones((400,2)) * [S1OnOff_E_L, S1OnOff_E_L]
    S1OnOff_g_ad = np.ones((400,2)) * [0,0]
    R1On_On_PSC_s = np.ones((400,2)) * [0,0]
    R1On_On_PSC_x = np.ones((400,2)) * [0,0]
    R1On_On_PSC_F = np.ones((400,2)) * [1,1]
    R1On_On_PSC_P = np.ones((400,2)) * [1,1]
    R1On_On_PSC_q = np.ones((400,2)) * [1,1]
    S1OnOff_On_PSC_s = np.ones((400,2)) * [0,0]
    S1OnOff_On_PSC_x = np.ones((400,2)) * [0,0]
    S1OnOff_On_PSC_F = np.ones((400,2)) * [1,1]
    S1OnOff_On_PSC_P = np.ones((400,2)) * [1,1]
    S1OnOff_On_PSC_q = np.ones((400,2)) * [1,1]
    R1On_S1OnOff_PSC_s = np.ones((400,2)) * [0,0]
    R1On_S1OnOff_PSC_x = np.ones((400,2)) * [0,0]
    R1On_S1OnOff_PSC_F = np.ones((400,2)) * [1,1]
    R1On_S1OnOff_PSC_P = np.ones((400,2)) * [1,1]
    R1On_S1OnOff_PSC_q = np.ones((400,2)) * [1,1]
    R1Off_S1OnOff_PSC_s = np.ones((400,2)) * [0,0]
    R1Off_S1OnOff_PSC_x = np.ones((400,2)) * [0,0]
    R1Off_S1OnOff_PSC_F = np.ones((400,2)) * [1,1]
    R1Off_S1OnOff_PSC_P = np.ones((400,2)) * [1,1]
    R1Off_S1OnOff_PSC_q = np.ones((400,2)) * [1,1]
    R1Off_Off_PSC_s = np.ones((400,2)) * [0,0]
    R1Off_Off_PSC_x = np.ones((400,2)) * [0,0]
    R1Off_Off_PSC_F = np.ones((400,2)) * [1,1]
    R1Off_Off_PSC_P = np.ones((400,2)) * [1,1]
    R1Off_Off_PSC_q = np.ones((400,2)) * [1,1]
    S1OnOff_Off_PSC_s = np.ones((400,2)) * [0,0]
    S1OnOff_Off_PSC_x = np.ones((400,2)) * [0,0]
    S1OnOff_Off_PSC_F = np.ones((400,2)) * [1,1]
    S1OnOff_Off_PSC_P = np.ones((400,2)) * [1,1]
    S1OnOff_Off_PSC_q = np.ones((400,2)) * [1,1]
    R1On_R1On_iNoise_V3_sn = np.ones((400,2)) * [0, 0]
    R1On_R1On_iNoise_V3_xn = np.ones((400,2)) * [0, 0]

    #Monitor Declaration
    On_tspike = -30*np.ones((400, 5, On_Npop))
    On_buffer_index = np.ones((400))
    On_V_spikes_holder = []
    Off_tspike = -30*np.ones((400, 5, Off_Npop))
    Off_buffer_index = np.ones((400))
    Off_V_spikes_holder = []
    R1On_tspike = -30*np.ones((400, 5, R1On_Npop))
    R1On_buffer_index = np.ones((400))
    R1On_V_spikes_holder = []
    R1Off_tspike = -30*np.ones((400, 5, R1Off_Npop))
    R1Off_buffer_index = np.ones((400))
    R1Off_V_spikes_holder = []
    S1OnOff_tspike = -30*np.ones((400, 5, S1OnOff_Npop))
    S1OnOff_buffer_index = np.ones((400))
    S1OnOff_V_spikes_holder = []
    On_On_IC_iIC = 0
    Off_Off_IC_iIC = 0
    R1On_On_PSC_syn = 0
    S1OnOff_On_PSC_syn = 0
    R1On_S1OnOff_PSC_syn = 0
    R1Off_S1OnOff_PSC_syn = 0
    R1Off_Off_PSC_syn = 0
    S1OnOff_Off_PSC_syn = 0
    dv1dOn_V = np.ones((400,2))
    dv2dOn_V = np.ones((400,2))
    dv2_dOn_tref = np.ones((400,2))
    spikers_On = np.zeros((400)).astype(np.int8)
    dv1dOff_V = np.ones((400,2))
    dv2dOff_V = np.ones((400,2))
    dv2_dOff_tref = np.ones((400,2))
    spikers_Off = np.zeros((400)).astype(np.int8)
    dv1dR1On_V = np.ones((400,2))
    dv2dR1On_V = np.ones((400,2))
    dv2_dR1On_tref = np.ones((400,2))
    spikers_R1On = np.zeros((400)).astype(np.int8)
    dv1dS1OnOff_V = np.ones((400,2))
    dv2dS1OnOff_V = np.ones((400,2))
    dv2_dS1OnOff_tref = np.ones((400,2))
    spikers_S1OnOff = np.zeros((400)).astype(np.int8)

    #Delcare Inputs
    On_On_IC_input = genPoissonInputs.gen_poisson_inputs(trial_number,On_On_IC_locNum,On_On_IC_label,On_On_IC_t_ref,On_On_IC_t_ref_rel,On_On_IC_rec,scale_factor)
    Off_Off_IC_input = genPoissonInputs.gen_poisson_inputs(trial_number,Off_Off_IC_locNum,Off_Off_IC_label,Off_Off_IC_t_ref,Off_Off_IC_t_ref_rel,Off_Off_IC_rec,scale_factor,"Layer4")

    for t in range(0,T):

        #ODEs
        On_V_k1 = ( (On_E_L-On_V[:,-1]) - On_R*On_g_ad[:,-1]*(On_V[:,-1]-On_E_k) - On_R*((((On_On_IC_g_postIC*(On_On_IC_input[t]*On_On_IC_netcon)*(On_V[:,-1]-On_On_IC_E_exc))))) + On_R*On_Itonic*On_Imask  ) / On_tau
        On_g_ad_k1 = -On_g_ad[:,-1] / On_tau_ad
        Off_V_k1 = ( (Off_E_L-Off_V[:,-1]) - Off_R*Off_g_ad[:,-1]*(Off_V[:,-1]-Off_E_k) - Off_R*((((Off_Off_IC_g_postIC*(Off_Off_IC_input[t]*Off_Off_IC_netcon)*(Off_V[:,-1]-Off_Off_IC_E_exc))))) + Off_R*Off_Itonic*Off_Imask  ) / Off_tau
        Off_g_ad_k1 = -Off_g_ad[:,-1] / Off_tau_ad
        R1On_V_k1 = ( (R1On_E_L-R1On_V[:,-1]) - R1On_R*R1On_g_ad[:,-1]*(R1On_V[:,-1]-R1On_E_k) - R1On_R*((((R1On_On_PSC_gSYN*(R1On_On_PSC_s[:,-1]*R1On_On_PSC_netcon)*(R1On_V[:,-1]-R1On_On_PSC_ESYN))))+((((R1On_S1OnOff_PSC_gSYN*(R1On_S1OnOff_PSC_s[:,-1]*R1On_S1OnOff_PSC_netcon)*(R1On_V[:,-1]-R1On_S1OnOff_PSC_ESYN))))+((((R1On_R1On_iNoise_V3_nSYN*(R1On_R1On_iNoise_V3_sn[:,-1]*R1On_R1On_iNoise_V3_netcon)*(R1On_V[:,-1]-R1On_R1On_iNoise_V3_E_exc))))))) + R1On_R*R1On_Itonic*R1On_Imask  ) / R1On_tau
        R1On_g_ad_k1 = -R1On_g_ad[:,-1] / R1On_tau_ad
        R1Off_V_k1 = ( (R1Off_E_L-R1Off_V[:,-1]) - R1Off_R*R1Off_g_ad[:,-1]*(R1Off_V[:,-1]-R1Off_E_k) - R1Off_R*((((R1Off_S1OnOff_PSC_gSYN*(R1Off_S1OnOff_PSC_s[:,-1]*R1Off_S1OnOff_PSC_netcon)*(R1Off_V[:,-1]-R1Off_S1OnOff_PSC_ESYN))))+((((R1Off_Off_PSC_gSYN*(R1Off_Off_PSC_s[:,-1]*R1Off_Off_PSC_netcon)*(R1Off_V[:,-1]-R1Off_Off_PSC_ESYN)))))) + R1Off_R*R1Off_Itonic*R1Off_Imask  ) / R1Off_tau
        R1Off_g_ad_k1 = -R1Off_g_ad[:,-1] / R1Off_tau_ad
        S1OnOff_V_k1 = ( (S1OnOff_E_L-S1OnOff_V[:,-1]) - S1OnOff_R*S1OnOff_g_ad[:,-1]*(S1OnOff_V[:,-1]-S1OnOff_E_k) - S1OnOff_R*((((S1OnOff_On_PSC_gSYN*(S1OnOff_On_PSC_s[:,-1]*S1OnOff_On_PSC_netcon)*(S1OnOff_V[:,-1]-S1OnOff_On_PSC_ESYN))))+((((S1OnOff_Off_PSC_gSYN*(S1OnOff_Off_PSC_s[:,-1]*S1OnOff_Off_PSC_netcon)*(S1OnOff_V[:,-1]-S1OnOff_Off_PSC_ESYN)))))) + S1OnOff_R*S1OnOff_Itonic*S1OnOff_Imask  ) / S1OnOff_tau
        S1OnOff_g_ad_k1 = -S1OnOff_g_ad[:,-1] / S1OnOff_tau_ad
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
        R1On_R1On_iNoise_V3_sn_k1 = ( R1On_R1On_iNoise_V3_scale * R1On_R1On_iNoise_V3_xn[:,-1] - R1On_R1On_iNoise_V3_sn[:,-1] )/R1On_R1On_iNoise_V3_tauR_N
        R1On_R1On_iNoise_V3_xn_k1 = -R1On_R1On_iNoise_V3_xn[:,-1]/R1On_R1On_iNoise_V3_tauD_N + R1On_R1On_iNoise_V3_token[t]/R1On_R1On_iNoise_V3_dt

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
        R1On_R1On_iNoise_V3_sn[:,-2] = R1On_R1On_iNoise_V3_sn[:,-1]
        R1On_R1On_iNoise_V3_sn[:,-1] = R1On_R1On_iNoise_V3_sn[:,-1]+dt*R1On_R1On_iNoise_V3_sn_k1
        R1On_R1On_iNoise_V3_xn[:,-2] = R1On_R1On_iNoise_V3_xn[:,-1]
        R1On_R1On_iNoise_V3_xn[:,-1] = R1On_R1On_iNoise_V3_xn[:,-1]+dt*R1On_R1On_iNoise_V3_xn_k1

        #Spiking and conditional actions
        mask = ((On_V[:,-1] >= On_V_thresh) & (On_V[:,-2] < On_V_thresh)).astype(np.int8).tolist()
        On_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers_On = np.flatnonzero(mask)
            On_tspike[spikers_On, On_buffer_index[spikers_On].astype(np.int8)-1] = helper[t]
            On_buffer_index[spikers_On] = (On_buffer_index[spikers_On] % 5) + 1
        mask = ((Off_V[:,-1] >= Off_V_thresh) & (Off_V[:,-2] < Off_V_thresh)).astype(np.int8).tolist()
        Off_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers_Off = np.flatnonzero(mask)
            Off_tspike[spikers_Off, Off_buffer_index[spikers_Off].astype(np.int8)-1] = helper[t]
            Off_buffer_index[spikers_Off] = (Off_buffer_index[spikers_Off] % 5) + 1
        mask = ((R1On_V[:,-1] >= R1On_V_thresh) & (R1On_V[:,-2] < R1On_V_thresh)).astype(np.int8).tolist()
        R1On_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers_R1On = np.flatnonzero(mask)
            R1On_tspike[spikers_R1On, R1On_buffer_index[spikers_R1On].astype(np.int8)-1] = helper[t]
            R1On_buffer_index[spikers_R1On] = (R1On_buffer_index[spikers_R1On] % 5) + 1
        mask = ((R1Off_V[:,-1] >= R1Off_V_thresh) & (R1Off_V[:,-2] < R1Off_V_thresh)).astype(np.int8).tolist()
        R1Off_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers_R1Off = np.flatnonzero(mask)
            R1Off_tspike[spikers_R1Off, R1Off_buffer_index[spikers_R1Off].astype(np.int8)-1] = helper[t]
            R1Off_buffer_index[spikers_R1Off] = (R1Off_buffer_index[spikers_R1Off] % 5) + 1
        mask = ((S1OnOff_V[:,-1] >= S1OnOff_V_thresh) & (S1OnOff_V[:,-2] < S1OnOff_V_thresh)).astype(np.int8).tolist()
        S1OnOff_V_spikes_holder.append(mask)
        if np.any(mask):
            spikers_S1OnOff = np.flatnonzero(mask)
            S1OnOff_tspike[spikers_S1OnOff, S1OnOff_buffer_index[spikers_S1OnOff].astype(np.int8)-1] = helper[t]
            S1OnOff_buffer_index[spikers_S1OnOff] = (S1OnOff_buffer_index[spikers_S1OnOff] % 5) + 1

            #Voltage reset and adaptation
        mask = (On_V[:,-1] > On_V_thresh)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            On_V[spikers,-2] = On_V[spikers,-1] 
            On_V[spikers,-1] = On_V_reset 
            On_g_ad[spikers,-2] = On_g_ad[spikers,-1]
            On_g_ad[spikers,-1] = On_g_ad[spikers,-1] + On_g_inc
        mask = np.any((helper[t] <= (np.squeeze(On_tspike) + On_t_ref[:, None])), axis = 1)
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
        mask = np.any((helper[t] <= (np.squeeze(Off_tspike) + Off_t_ref[:, None])), axis = 1)
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
        mask = np.any((helper[t] <= (np.squeeze(R1On_tspike) + R1On_t_ref[:, None])), axis = 1)
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
        mask = np.any((helper[t] <= (np.squeeze(R1Off_tspike) + R1Off_t_ref)), axis = 1)
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
        mask = np.any((helper[t] <= (np.squeeze(S1OnOff_tspike) + S1OnOff_t_ref[:, None])), axis = 1)
        if np.any(mask):
            spikers = np.flatnonzero(mask) 
            S1OnOff_V[spikers,-2] = S1OnOff_V[spikers,-1]
            S1OnOff_V[spikers,-1] = S1OnOff_V_reset

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

        #Grad Calculations


        #Surrogate Spike Related Derivates
        #dspike_dOn_V = (((10*np.exp(-(0.1)*(On_V[:,-1] - On_V_thresh)))/(1+np.exp(-(0.1)*(On_V[:,-1] - On_V_thresh)))**2))/500
        #dspike_dOff_V = (((10*np.exp(-(0.1)*(Off_V[:,-1] - Off_V_thresh)))/(1+np.exp(-(0.1)*(Off_V[:,-1] - Off_V_thresh)))**2))/500
        #dspike_dR1On_V = (((10*np.exp(-(0.1)*(R1On_V[:,-1] - R1On_V_thresh)))/(1+np.exp(-(0.1)*(R1On_V[:,-1] - R1On_V_thresh)))**2))/500
        #dspike_dS1OnOff_V = (((10*np.exp(-(0.1)*(S1OnOff_V[:,-1] - S1OnOff_V_thresh)))/(1+np.exp(-(0.1)*(S1OnOff_V[:,-1] - S1OnOff_V_thresh)))**2))/500
        dv1dOn_V[:,0] = dv1dOn_V[:,1]
        dv1dOn_V[:,1] = ((1+np.exp(On_V[:,-1]-On_V_thresh))-((On_V[:,-1]-On_V_reset)*np.exp(On_V[:,-1]-On_V_thresh)))/(1+np.exp(On_V[:,-1]-On_V_thresh))**2
        dv2dOn_V[:,0] = dv2dOn_V[:,1]
        dv2dOn_V[:,1] = np.squeeze(np.sum(1/(1+np.exp(-(helper[t]-(np.squeeze(On_tspike)+On_t_ref[:,None])))),axis=1))/5 #Nominally 5
        dukdv2On_V = (-np.squeeze((np.max(On_tspike,axis=1)-np.partition(On_tspike, -2, axis=1)[:, -2]))*(-np.exp(-(On_V[:,-1]-On_V_thresh)))*(1+np.exp((On_V[:,-2]-On_V_thresh))))/((1+np.exp(-(On_V[:,-1]-On_V_thresh)))*(1+np.exp((On_V[:,-2]-On_V_thresh))))**2
        dv1dOn_V_holder.append(-np.squeeze((np.max(On_tspike,axis=1))))
        dspike_dOn_V = dukdv2On_V*dv2dOn_V[:,0]*dv1dOn_V[:,0]
        dv1dOff_V[:,0] = dv1dOff_V[:,1]
        dv1dOff_V[:,1] = ((1+np.exp(Off_V[:,-1]-Off_V_thresh))-((Off_V[:,-1]-Off_V_reset)*np.exp(Off_V[:,-1]-Off_V_thresh)))/(1+np.exp(Off_V[:,-1]-Off_V_thresh))**2
        dv2dOff_V[:,0] = dv2dOff_V[:,1]
        dv2dOff_V[:,1] = np.squeeze(np.sum(1/(1+np.exp(-(helper[t]-(np.squeeze(Off_tspike)+Off_t_ref[:,None])))),axis=1))/5 #Nominally 5
        dukdv2Off_V = (-np.squeeze((np.max(Off_tspike,axis=1)-np.partition(Off_tspike, -2, axis=1)[:, -2]))*(-np.exp(-(Off_V[:,-1]-Off_V_thresh)))*(1+np.exp((Off_V[:,-2]-Off_V_thresh))))/((1+np.exp(-(Off_V[:,-1]-Off_V_thresh)))*(1+np.exp((Off_V[:,-2]-Off_V_thresh))))**2
        dv1dOff_V_holder.append(-np.squeeze((np.max(Off_tspike,axis=1))))
        dspike_dOff_V = dukdv2Off_V*dv2dOff_V[:,0]*dv1dOff_V[:,0]
        dv1dR1On_V[:,0] = dv1dR1On_V[:,1]
        dv1dR1On_V[:,1] = ((1+np.exp(R1On_V[:,-1]-R1On_V_thresh))-((R1On_V[:,-1]-R1On_V_reset)*np.exp(R1On_V[:,-1]-R1On_V_thresh)))/(1+np.exp(R1On_V[:,-1]-R1On_V_thresh))**2
        dv2dR1On_V[:,0] = dv2dR1On_V[:,1]
        dv2dR1On_V[:,1] = np.squeeze(np.sum(1/(1+np.exp(-(helper[t]-(np.squeeze(R1On_tspike)+R1On_t_ref[:,None])))),axis=1))/5 #Nominally 5
        dukdv2R1On_V = (-np.squeeze((np.max(R1On_tspike,axis=1)-np.partition(R1On_tspike, -2, axis=1)[:, -2]))*(-np.exp(-(R1On_V[:,-1]-R1On_V_thresh)))*(1+np.exp((R1On_V[:,-2]-R1On_V_thresh))))/((1+np.exp(-(R1On_V[:,-1]-R1On_V_thresh)))*(1+np.exp((R1On_V[:,-2]-R1On_V_thresh))))**2
        dv1dR1On_V_holder.append(-np.squeeze((np.max(R1On_tspike,axis=1))))
        dspike_dR1On_V = dukdv2R1On_V*dv2dR1On_V[:,0]*dv1dR1On_V[:,0]
        dv1dS1OnOff_V[:,0] = dv1dS1OnOff_V[:,1]
        dv1dS1OnOff_V[:,1] = ((1+np.exp(S1OnOff_V[:,-1]-S1OnOff_V_thresh))-((S1OnOff_V[:,-1]-S1OnOff_V_reset)*np.exp(S1OnOff_V[:,-1]-S1OnOff_V_thresh)))/(1+np.exp(S1OnOff_V[:,-1]-S1OnOff_V_thresh))**2
        dv2dS1OnOff_V[:,0] = dv2dS1OnOff_V[:,1]
        dv2dS1OnOff_V[:,1] = np.squeeze(np.sum(1/(1+np.exp(-(helper[t]-(np.squeeze(S1OnOff_tspike)+S1OnOff_t_ref[:,None])))),axis=1))/5 #Nominally 5
        dukdv2S1OnOff_V = (-np.squeeze((np.max(S1OnOff_tspike,axis=1)-np.partition(S1OnOff_tspike, -2, axis=1)[:, -2]))*(-np.exp(-(S1OnOff_V[:,-1]-S1OnOff_V_thresh)))*(1+np.exp((S1OnOff_V[:,-2]-S1OnOff_V_thresh))))/((1+np.exp(-(S1OnOff_V[:,-1]-S1OnOff_V_thresh)))*(1+np.exp((S1OnOff_V[:,-2]-S1OnOff_V_thresh))))**2
        dv1dS1OnOff_V_holder.append(-np.squeeze((np.max(S1OnOff_tspike,axis=1))))
        dspike_dS1OnOff_V = dukdv2S1OnOff_V*dv2dS1OnOff_V[:,0]*dv1dS1OnOff_V[:,0]


        #PSC & Parameter Related Derivates
        dv_dR1On_On_PSC_gSYN = np.squeeze(-(dt*R1On_R*R1On_On_PSC_s[:,-1]*R1On_On_PSC_netcon*(R1On_V[:,-1]-R1On_On_PSC_ESYN)/R1On_tau)/15)
        dR1On_On_PSC_dUk = np.squeeze(-((dt*R1On_On_PSC_scale*2*(R1On_On_PSC_x[:,-1]+R1On_On_PSC_q[:,-1])/R1On_On_PSC_tauR)*helper[t]*np.squeeze(np.sum((((On_tspike+R1On_On_PSC_delay)-helper[t])*np.exp(-1*((On_tspike+R1On_On_PSC_delay)-helper[t])**2)),axis=1)))/2500)
        dv_dR1On_On_PSC = np.squeeze(-(dt*R1On_R*R1On_On_PSC_gSYN*R1On_On_PSC_netcon*(R1On_V[:,-1]-R1On_On_PSC_ESYN)/R1On_tau)/10)
        dv_dS1OnOff_On_PSC_gSYN = np.squeeze(-(dt*S1OnOff_R*S1OnOff_On_PSC_s[:,-1]*S1OnOff_On_PSC_netcon*(S1OnOff_V[:,-1]-S1OnOff_On_PSC_ESYN)/S1OnOff_tau)/15)
        dS1OnOff_On_PSC_dUk = np.squeeze(-((dt*S1OnOff_On_PSC_scale*2*(S1OnOff_On_PSC_x[:,-1]+S1OnOff_On_PSC_q[:,-1])/S1OnOff_On_PSC_tauR)*helper[t]*np.squeeze(np.sum((((On_tspike+S1OnOff_On_PSC_delay)-helper[t])*np.exp(-1*((On_tspike+S1OnOff_On_PSC_delay)-helper[t])**2)),axis=1)))/2500)
        dv_dS1OnOff_On_PSC = np.squeeze(-(dt*S1OnOff_R*S1OnOff_On_PSC_gSYN*S1OnOff_On_PSC_netcon*(S1OnOff_V[:,-1]-S1OnOff_On_PSC_ESYN)/S1OnOff_tau)/10)
        dv_dR1On_S1OnOff_PSC_gSYN = np.squeeze(-(dt*R1On_R*R1On_S1OnOff_PSC_s[:,-1]*R1On_S1OnOff_PSC_netcon*(R1On_V[:,-1]-R1On_S1OnOff_PSC_ESYN)/R1On_tau)/15)
        dR1On_S1OnOff_PSC_dUk = np.squeeze(-((dt*R1On_S1OnOff_PSC_scale*2*(R1On_S1OnOff_PSC_x[:,-1]+R1On_S1OnOff_PSC_q[:,-1])/R1On_S1OnOff_PSC_tauR)*helper[t]*np.squeeze(np.sum((((S1OnOff_tspike+R1On_S1OnOff_PSC_delay)-helper[t])*np.exp(-1*((S1OnOff_tspike+R1On_S1OnOff_PSC_delay)-helper[t])**2)),axis=1)))/2500)
        dv_dR1On_S1OnOff_PSC = np.squeeze(-(dt*R1On_R*R1On_S1OnOff_PSC_gSYN*R1On_S1OnOff_PSC_netcon*(R1On_V[:,-1]-R1On_S1OnOff_PSC_ESYN)/R1On_tau)/10)
        dv_dS1OnOff_Off_PSC_gSYN = np.squeeze(-(dt*S1OnOff_R*S1OnOff_Off_PSC_s[:,-1]*S1OnOff_Off_PSC_netcon*(S1OnOff_V[:,-1]-S1OnOff_Off_PSC_ESYN)/S1OnOff_tau)/15)
        dS1OnOff_Off_PSC_dUk = np.squeeze(-((dt*S1OnOff_Off_PSC_scale*2*(S1OnOff_Off_PSC_x[:,-1]+S1OnOff_Off_PSC_q[:,-1])/S1OnOff_Off_PSC_tauR)*helper[t]*np.squeeze(np.sum((((Off_tspike+S1OnOff_Off_PSC_delay)-helper[t])*np.exp(-1*((Off_tspike+S1OnOff_Off_PSC_delay)-helper[t])**2)),axis=1)))/2500)
        dv_dS1OnOff_Off_PSC = np.squeeze(-(dt*S1OnOff_R*S1OnOff_Off_PSC_gSYN*S1OnOff_Off_PSC_netcon*(S1OnOff_V[:,-1]-S1OnOff_Off_PSC_ESYN)/S1OnOff_tau)/10)


        #Tref Related Derivates
        dv2_dOn_tref[:,0] = dv2_dOn_tref[:,1]
        dv2_dOn_tref[:,1] = np.squeeze(np.sum(-(On_V[:,-1]-On_V_reset)[:,None]*np.squeeze(np.exp(-(helper[t]-(np.squeeze(S1OnOff_tspike)+S1OnOff_t_ref[:,None]))))/(1+np.squeeze(np.exp(-(helper[t]-(np.squeeze(S1OnOff_tspike)+S1OnOff_t_ref[:,None])))))**2,axis=1))
        dspike_dOn_tref = dukdv2On_V*dv2_dOn_tref[:,0]
        dv2_dOff_tref[:,0] = dv2_dOff_tref[:,1]
        dv2_dOff_tref[:,1] = np.squeeze(np.sum(-(Off_V[:,-1]-Off_V_reset)[:,None]*np.squeeze(np.exp(-(helper[t]-(np.squeeze(S1OnOff_tspike)+S1OnOff_t_ref[:,None]))))/(1+np.squeeze(np.exp(-(helper[t]-(np.squeeze(S1OnOff_tspike)+S1OnOff_t_ref[:,None])))))**2,axis=1))
        dspike_dOff_tref = dukdv2Off_V*dv2_dOff_tref[:,0]
        dv2_dR1On_tref[:,0] = dv2_dR1On_tref[:,1]
        dv2_dR1On_tref[:,1] = np.squeeze(np.sum(-(R1On_V[:,-1]-R1On_V_reset)[:,None]*np.squeeze(np.exp(-(helper[t]-(np.squeeze(S1OnOff_tspike)+S1OnOff_t_ref[:,None]))))/(1+np.squeeze(np.exp(-(helper[t]-(np.squeeze(S1OnOff_tspike)+S1OnOff_t_ref[:,None])))))**2,axis=1))
        dspike_dR1On_tref = dukdv2R1On_V*dv2_dR1On_tref[:,0]
        dv2_dS1OnOff_tref[:,0] = dv2_dS1OnOff_tref[:,1]
        dv2_dS1OnOff_tref[:,1] = np.squeeze(np.sum(-(S1OnOff_V[:,-1]-S1OnOff_V_reset)[:,None]*np.squeeze(np.exp(-(helper[t]-(np.squeeze(S1OnOff_tspike)+S1OnOff_t_ref[:,None]))))/(1+np.squeeze(np.exp(-(helper[t]-(np.squeeze(S1OnOff_tspike)+S1OnOff_t_ref[:,None])))))**2,axis=1))
        dspike_dS1OnOff_tref = dukdv2S1OnOff_V*dv2_dS1OnOff_tref[:,0]


        #Input Related Derivates
        dv_dOff_input = (-Off_R*Off_Off_IC_input[t]*Off_Off_IC_netcon*(Off_V[:,-1]-Off_Off_IC_E_exc)/Off_tau)/1000 #Nominally 1000
        dv_dOn_input = (-On_R*On_On_IC_input[t]*On_On_IC_netcon*(On_V[:,-1]-On_On_IC_E_exc)/On_tau)/1000 #Nominally 1000

        #Build derivs
        dGSYNR1On_On += dspike_dR1On_V*dv_dR1On_On_PSC_gSYN
        dGSYNS1OnOff_On += dspike_dR1On_V*dv_dR1On_S1OnOff_PSC*dR1On_S1OnOff_PSC_dUk*dspike_dS1OnOff_V*dv_dS1OnOff_On_PSC_gSYN
        dGSYNR1On_S1OnOff += dspike_dR1On_V*dv_dR1On_S1OnOff_PSC_gSYN
        dGSYNS1OnOff_Off += dspike_dR1On_V*dv_dR1On_S1OnOff_PSC*dR1On_S1OnOff_PSC_dUk*dspike_dS1OnOff_V*dv_dS1OnOff_Off_PSC_gSYN


        #Build T_ref derivs
        dtref_On = dspike_dR1On_V*dv_dR1On_On_PSC*dR1On_On_PSC_dUk*dspike_dOn_tref+dspike_dR1On_V*dv_dR1On_S1OnOff_PSC*dR1On_S1OnOff_PSC_dUk*dspike_dS1OnOff_V*dv_dS1OnOff_On_PSC*dS1OnOff_On_PSC_dUk*dspike_dOn_tref
        dtref_Off = dspike_dR1On_V*dv_dR1On_S1OnOff_PSC*dR1On_S1OnOff_PSC_dUk*dspike_dS1OnOff_V*dv_dS1OnOff_Off_PSC*dS1OnOff_Off_PSC_dUk*dspike_dOff_tref
        dtref_R1On = dspike_dR1On_tref
        dtref_S1OnOff = dspike_dR1On_V*dv_dR1On_S1OnOff_PSC*dR1On_S1OnOff_PSC_dUk*dspike_dS1OnOff_tref


        #Build Input derivs
        dinput_On = dspike_dR1On_V*dv_dR1On_On_PSC*dR1On_On_PSC_dUk*dspike_dOn_V*dv_dOn_input+dspike_dR1On_V*dv_dR1On_S1OnOff_PSC*dR1On_S1OnOff_PSC_dUk*dspike_dS1OnOff_V*dv_dS1OnOff_On_PSC*dS1OnOff_On_PSC_dUk*dspike_dOn_V*dv_dOn_input
        dinput_Off = dspike_dR1On_V*dv_dR1On_S1OnOff_PSC*dR1On_S1OnOff_PSC_dUk*dspike_dS1OnOff_V*dv_dS1OnOff_Off_PSC*dS1OnOff_Off_PSC_dUk*dspike_dOff_V*dv_dOff_input


    return R1On_V_spikes_holder, [dGSYNR1On_On, dGSYNS1OnOff_On, dGSYNR1On_S1OnOff, dGSYNS1OnOff_Off, dtref_On, dtref_Off, dtref_R1On, dtref_S1OnOff, dinput_Off, dinput_On]