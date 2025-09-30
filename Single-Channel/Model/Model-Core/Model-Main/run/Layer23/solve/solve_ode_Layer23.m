function [T,On_V,On_g_ad,Off_V,Off_g_ad,R1On_V,R1On_g_ad,R1Off_V,R1Off_g_ad,S1OnOff_V,S1OnOff_g_ad,R2On_V,R2On_g_ad,R2Off_V,R2Off_g_ad,S2OnOff_V,S2OnOff_g_ad,R1On_On_PSC_s,R1On_On_PSC_x,R1On_On_PSC_F,R1On_On_PSC_P,R1On_On_PSC_q,S1OnOff_On_PSC_s,S1OnOff_On_PSC_x,S1OnOff_On_PSC_F,S1OnOff_On_PSC_P,S1OnOff_On_PSC_q,R1On_S1OnOff_PSC_s,R1On_S1OnOff_PSC_x,R1On_S1OnOff_PSC_F,R1On_S1OnOff_PSC_P,R1On_S1OnOff_PSC_q,R1Off_S1OnOff_PSC_s,R1Off_S1OnOff_PSC_x,R1Off_S1OnOff_PSC_F,R1Off_S1OnOff_PSC_P,R1Off_S1OnOff_PSC_q,R1Off_Off_PSC_s,R1Off_Off_PSC_x,R1Off_Off_PSC_F,R1Off_Off_PSC_P,R1Off_Off_PSC_q,S1OnOff_Off_PSC_s,S1OnOff_Off_PSC_x,S1OnOff_Off_PSC_F,S1OnOff_Off_PSC_P,S1OnOff_Off_PSC_q,R2On_R1On_PSC_s,R2On_R1On_PSC_x,R2On_R1On_PSC_F,R2On_R1On_PSC_P,R2On_R1On_PSC_q,S2OnOff_R1On_PSC_s,S2OnOff_R1On_PSC_x,S2OnOff_R1On_PSC_F,S2OnOff_R1On_PSC_P,S2OnOff_R1On_PSC_q,R2On_S2OnOff_PSC_s,R2On_S2OnOff_PSC_x,R2On_S2OnOff_PSC_F,R2On_S2OnOff_PSC_P,R2On_S2OnOff_PSC_q,R2Off_S2OnOff_PSC_s,R2Off_S2OnOff_PSC_x,R2Off_S2OnOff_PSC_F,R2Off_S2OnOff_PSC_P,R2Off_S2OnOff_PSC_q,R2Off_R1Off_PSC_s,R2Off_R1Off_PSC_x,R2Off_R1Off_PSC_F,R2Off_R1Off_PSC_P,R2Off_R1Off_PSC_q,S2OnOff_R1Off_PSC_s,S2OnOff_R1Off_PSC_x,S2OnOff_R1Off_PSC_F,S2OnOff_R1Off_PSC_P,S2OnOff_R1Off_PSC_q,R2On_R2On_iNoise_V3_sn,R2On_R2On_iNoise_V3_xn,On_V_spikes,Off_V_spikes,R1On_V_spikes,R1Off_V_spikes,S1OnOff_V_spikes,R2On_V_spikes,R2Off_V_spikes,S2OnOff_V_spikes,On_On_IC_iIC,Off_Off_IC_iIC,R1On_On_PSC_syn,S1OnOff_On_PSC_syn,R1On_S1OnOff_PSC_syn,R1Off_S1OnOff_PSC_syn,R1Off_Off_PSC_syn,S1OnOff_Off_PSC_syn,R2On_R1On_PSC_syn,S2OnOff_R1On_PSC_syn,R2On_S2OnOff_PSC_syn,R2Off_S2OnOff_PSC_syn,R2Off_R1Off_PSC_syn,S2OnOff_R1Off_PSC_syn,On_R,On_tau,On_Imask,Off_R,Off_tau,Off_Imask,R1On_R,R1On_tau,R1On_Imask,R1Off_R,R1Off_tau,R1Off_Imask,S1OnOff_R,S1OnOff_tau,S1OnOff_Imask,R2On_R,R2On_tau,R2On_Imask,R2Off_R,R2Off_tau,R2Off_Imask,S2OnOff_R,S2OnOff_tau,S2OnOff_Imask,On_On_IC_netcon,On_On_IC_input,Off_Off_IC_netcon,Off_Off_IC_input,R1On_On_PSC_netcon,R1On_On_PSC_scale,S1OnOff_On_PSC_netcon,S1OnOff_On_PSC_scale,R1On_S1OnOff_PSC_netcon,R1On_S1OnOff_PSC_scale,R1Off_S1OnOff_PSC_netcon,R1Off_S1OnOff_PSC_scale,R1Off_Off_PSC_netcon,R1Off_Off_PSC_scale,S1OnOff_Off_PSC_netcon,S1OnOff_Off_PSC_scale,R2On_R1On_PSC_netcon,R2On_R1On_PSC_scale,S2OnOff_R1On_PSC_netcon,S2OnOff_R1On_PSC_scale,R2On_S2OnOff_PSC_netcon,R2On_S2OnOff_PSC_scale,R2Off_S2OnOff_PSC_netcon,R2Off_S2OnOff_PSC_scale,R2Off_R1Off_PSC_netcon,R2Off_R1Off_PSC_scale,S2OnOff_R1Off_PSC_netcon,S2OnOff_R1Off_PSC_scale,R2On_R2On_iNoise_V3_netcon,R2On_R2On_iNoise_V3_token,R2On_R2On_iNoise_V3_scale]=solve_ode(tspan, downsample_factor, random_seed, solver, disk_flag, dt, datafile, mex_flag, verbose_flag, On_C, On_g_L, On_E_L, On_noise, On_t_ref, On_E_k, On_tau_ad, On_g_inc, On_Itonic, On_V_thresh, On_V_reset, On_Npop, Off_C, Off_g_L, Off_E_L, Off_noise, Off_t_ref, Off_E_k, Off_tau_ad, Off_g_inc, Off_Itonic, Off_V_thresh, Off_V_reset, Off_Npop, R1On_C, R1On_g_L, R1On_E_L, R1On_noise, R1On_t_ref, R1On_E_k, R1On_tau_ad, R1On_g_inc, R1On_Itonic, R1On_V_thresh, R1On_V_reset, R1On_Npop, R1Off_C, R1Off_g_L, R1Off_E_L, R1Off_noise, R1Off_t_ref, R1Off_E_k, R1Off_tau_ad, R1Off_g_inc, R1Off_Itonic, R1Off_V_thresh, R1Off_V_reset, R1Off_Npop, S1OnOff_C, S1OnOff_g_L, S1OnOff_E_L, S1OnOff_noise, S1OnOff_t_ref, S1OnOff_E_k, S1OnOff_tau_ad, S1OnOff_g_inc, S1OnOff_Itonic, S1OnOff_V_thresh, S1OnOff_V_reset, S1OnOff_Npop, R2On_C, R2On_g_L, R2On_E_L, R2On_noise, R2On_t_ref, R2On_E_k, R2On_tau_ad, R2On_g_inc, R2On_Itonic, R2On_V_thresh, R2On_V_reset, R2On_Npop, R2Off_C, R2Off_g_L, R2Off_E_L, R2Off_noise, R2Off_t_ref, R2Off_E_k, R2Off_tau_ad, R2Off_g_inc, R2Off_Itonic, R2Off_V_thresh, R2Off_V_reset, R2Off_Npop, S2OnOff_C, S2OnOff_g_L, S2OnOff_E_L, S2OnOff_noise, S2OnOff_t_ref, S2OnOff_E_k, S2OnOff_tau_ad, S2OnOff_g_inc, S2OnOff_Itonic, S2OnOff_V_thresh, S2OnOff_V_reset, S2OnOff_Npop, On_On_IC_trial, On_On_IC_locNum, On_On_IC_label, On_On_IC_t_ref, On_On_IC_t_ref_rel, On_On_IC_rec, On_On_IC_g_postIC, On_On_IC_E_exc, Off_Off_IC_trial, Off_Off_IC_locNum, Off_Off_IC_label, Off_Off_IC_t_ref, Off_Off_IC_t_ref_rel, Off_Off_IC_rec, Off_Off_IC_g_postIC, Off_Off_IC_E_exc, R1On_On_PSC_ESYN, R1On_On_PSC_tauD, R1On_On_PSC_tauR, R1On_On_PSC_delay, R1On_On_PSC_gSYN, R1On_On_PSC_fF, R1On_On_PSC_fP, R1On_On_PSC_tauF, R1On_On_PSC_tauP, R1On_On_PSC_maxF, S1OnOff_On_PSC_ESYN, S1OnOff_On_PSC_tauD, S1OnOff_On_PSC_tauR, S1OnOff_On_PSC_delay, S1OnOff_On_PSC_gSYN, S1OnOff_On_PSC_fF, S1OnOff_On_PSC_fP, S1OnOff_On_PSC_tauF, S1OnOff_On_PSC_tauP, S1OnOff_On_PSC_maxF, R1On_S1OnOff_PSC_ESYN, R1On_S1OnOff_PSC_tauD, R1On_S1OnOff_PSC_tauR, R1On_S1OnOff_PSC_delay, R1On_S1OnOff_PSC_gSYN, R1On_S1OnOff_PSC_fF, R1On_S1OnOff_PSC_fP, R1On_S1OnOff_PSC_tauF, R1On_S1OnOff_PSC_tauP, R1On_S1OnOff_PSC_maxF, R1Off_S1OnOff_PSC_ESYN, R1Off_S1OnOff_PSC_tauD, R1Off_S1OnOff_PSC_tauR, R1Off_S1OnOff_PSC_delay, R1Off_S1OnOff_PSC_gSYN, R1Off_S1OnOff_PSC_fF, R1Off_S1OnOff_PSC_fP, R1Off_S1OnOff_PSC_tauF, R1Off_S1OnOff_PSC_tauP, R1Off_S1OnOff_PSC_maxF, R1Off_Off_PSC_ESYN, R1Off_Off_PSC_tauD, R1Off_Off_PSC_tauR, R1Off_Off_PSC_delay, R1Off_Off_PSC_gSYN, R1Off_Off_PSC_fF, R1Off_Off_PSC_fP, R1Off_Off_PSC_tauF, R1Off_Off_PSC_tauP, R1Off_Off_PSC_maxF, S1OnOff_Off_PSC_ESYN, S1OnOff_Off_PSC_tauD, S1OnOff_Off_PSC_tauR, S1OnOff_Off_PSC_delay, S1OnOff_Off_PSC_gSYN, S1OnOff_Off_PSC_fF, S1OnOff_Off_PSC_fP, S1OnOff_Off_PSC_tauF, S1OnOff_Off_PSC_tauP, S1OnOff_Off_PSC_maxF, R2On_R1On_PSC_ESYN, R2On_R1On_PSC_tauD, R2On_R1On_PSC_tauR, R2On_R1On_PSC_delay, R2On_R1On_PSC_gSYN, R2On_R1On_PSC_fF, R2On_R1On_PSC_fP, R2On_R1On_PSC_tauF, R2On_R1On_PSC_tauP, R2On_R1On_PSC_maxF, S2OnOff_R1On_PSC_ESYN, S2OnOff_R1On_PSC_tauD, S2OnOff_R1On_PSC_tauR, S2OnOff_R1On_PSC_delay, S2OnOff_R1On_PSC_gSYN, S2OnOff_R1On_PSC_fF, S2OnOff_R1On_PSC_fP, S2OnOff_R1On_PSC_tauF, S2OnOff_R1On_PSC_tauP, S2OnOff_R1On_PSC_maxF, R2On_S2OnOff_PSC_ESYN, R2On_S2OnOff_PSC_tauD, R2On_S2OnOff_PSC_tauR, R2On_S2OnOff_PSC_delay, R2On_S2OnOff_PSC_gSYN, R2On_S2OnOff_PSC_fF, R2On_S2OnOff_PSC_fP, R2On_S2OnOff_PSC_tauF, R2On_S2OnOff_PSC_tauP, R2On_S2OnOff_PSC_maxF, R2Off_S2OnOff_PSC_ESYN, R2Off_S2OnOff_PSC_tauD, R2Off_S2OnOff_PSC_tauR, R2Off_S2OnOff_PSC_delay, R2Off_S2OnOff_PSC_gSYN, R2Off_S2OnOff_PSC_fF, R2Off_S2OnOff_PSC_fP, R2Off_S2OnOff_PSC_tauF, R2Off_S2OnOff_PSC_tauP, R2Off_S2OnOff_PSC_maxF, R2Off_R1Off_PSC_ESYN, R2Off_R1Off_PSC_tauD, R2Off_R1Off_PSC_tauR, R2Off_R1Off_PSC_delay, R2Off_R1Off_PSC_gSYN, R2Off_R1Off_PSC_fF, R2Off_R1Off_PSC_fP, R2Off_R1Off_PSC_tauF, R2Off_R1Off_PSC_tauP, R2Off_R1Off_PSC_maxF, S2OnOff_R1Off_PSC_ESYN, S2OnOff_R1Off_PSC_tauD, S2OnOff_R1Off_PSC_tauR, S2OnOff_R1Off_PSC_delay, S2OnOff_R1Off_PSC_gSYN, S2OnOff_R1Off_PSC_fF, S2OnOff_R1Off_PSC_fP, S2OnOff_R1Off_PSC_tauF, S2OnOff_R1Off_PSC_tauP, S2OnOff_R1Off_PSC_maxF, R2On_R2On_iNoise_V3_FR, R2On_R2On_iNoise_V3_sigma, R2On_R2On_iNoise_V3_dt, R2On_R2On_iNoise_V3_nSYN, R2On_R2On_iNoise_V3_simlen, R2On_R2On_iNoise_V3_tauD_N, R2On_R2On_iNoise_V3_tauR_N, R2On_R2On_iNoise_V3_E_exc, ROn_X_PSC3_netcon, ROn_SOnOff_PSC3_netcon, C_ROn_PSC3_netcon)

% ------------------------------------------------------------
% Parameters:
% ------------------------------------------------------------
params = load('params.mat','p');
p = params.p;
storeFields = fieldnames(p);
downsample_factor=p.downsample_factor;
dt=p.dt;
T=(p.tspan(1):dt:p.tspan(2))';
ntime=length(T);
nsamp=length(1:downsample_factor:ntime);

% the 'shuffle' random seed has been set in advance.

% ------------------------------------------------------------
% Fixed variables:
% ------------------------------------------------------------
On_R = 1/On_g_L;
On_tau = On_C*On_R;
On_Imask =  ones(1,On_Npop);
Off_R = 1/Off_g_L;
Off_tau = Off_C*Off_R;
Off_Imask =  ones(1,Off_Npop);
R1On_R = 1/R1On_g_L;
R1On_tau = R1On_C*R1On_R;
R1On_Imask =  ones(1,R1On_Npop);
R1Off_R = 1/R1Off_g_L;
R1Off_tau = R1Off_C*R1Off_R;
R1Off_Imask =  ones(1,R1Off_Npop);
S1OnOff_R = 1/S1OnOff_g_L;
S1OnOff_tau = S1OnOff_C*S1OnOff_R;
S1OnOff_Imask =  ones(1,S1OnOff_Npop);
R2On_R = 1/R2On_g_L;
R2On_tau = R2On_C*R2On_R;
R2On_Imask =  ones(1,R2On_Npop);
R2Off_R = 1/R2Off_g_L;
R2Off_tau = R2Off_C*R2Off_R;
R2Off_Imask =  ones(1,R2Off_Npop);
S2OnOff_R = 1/S2OnOff_g_L;
S2OnOff_tau = S2OnOff_C*S2OnOff_R;
S2OnOff_Imask =  ones(1,S2OnOff_Npop);
On_On_IC_netcon = +1.000000000000000e+00;
On_On_IC_input =  genPoissonInputs(On_On_IC_trial,On_On_IC_locNum,On_On_IC_label,On_On_IC_t_ref,On_On_IC_t_ref_rel,On_On_IC_rec);
Off_Off_IC_netcon = +1.000000000000000e+00;
Off_Off_IC_input =  genPoissonInputs(Off_Off_IC_trial,Off_Off_IC_locNum,Off_Off_IC_label,Off_Off_IC_t_ref,Off_Off_IC_t_ref_rel,Off_Off_IC_rec);
R1On_On_PSC_netcon = eye(On_Npop,R1On_Npop);
R1On_On_PSC_scale = (R1On_On_PSC_tauD/R1On_On_PSC_tauR)^(R1On_On_PSC_tauR/(R1On_On_PSC_tauD-R1On_On_PSC_tauR));
S1OnOff_On_PSC_netcon = eye(On_Npop,S1OnOff_Npop);
S1OnOff_On_PSC_scale = (S1OnOff_On_PSC_tauD/S1OnOff_On_PSC_tauR)^(S1OnOff_On_PSC_tauR/(S1OnOff_On_PSC_tauD-S1OnOff_On_PSC_tauR));
R1On_S1OnOff_PSC_netcon = eye(S1OnOff_Npop,R1On_Npop);
R1On_S1OnOff_PSC_scale = (R1On_S1OnOff_PSC_tauD/R1On_S1OnOff_PSC_tauR)^(R1On_S1OnOff_PSC_tauR/(R1On_S1OnOff_PSC_tauD-R1On_S1OnOff_PSC_tauR));
R1Off_S1OnOff_PSC_netcon = eye(S1OnOff_Npop,R1Off_Npop);
R1Off_S1OnOff_PSC_scale = (R1Off_S1OnOff_PSC_tauD/R1Off_S1OnOff_PSC_tauR)^(R1Off_S1OnOff_PSC_tauR/(R1Off_S1OnOff_PSC_tauD-R1Off_S1OnOff_PSC_tauR));
R1Off_Off_PSC_netcon = eye(Off_Npop,R1Off_Npop);
R1Off_Off_PSC_scale = (R1Off_Off_PSC_tauD/R1Off_Off_PSC_tauR)^(R1Off_Off_PSC_tauR/(R1Off_Off_PSC_tauD-R1Off_Off_PSC_tauR));
S1OnOff_Off_PSC_netcon = eye(Off_Npop,S1OnOff_Npop);
S1OnOff_Off_PSC_scale = (S1OnOff_Off_PSC_tauD/S1OnOff_Off_PSC_tauR)^(S1OnOff_Off_PSC_tauR/(S1OnOff_Off_PSC_tauD-S1OnOff_Off_PSC_tauR));
R2On_R1On_PSC_netcon = eye(R1On_Npop,R2On_Npop);
R2On_R1On_PSC_scale = (R2On_R1On_PSC_tauD/R2On_R1On_PSC_tauR)^(R2On_R1On_PSC_tauR/(R2On_R1On_PSC_tauD-R2On_R1On_PSC_tauR));
S2OnOff_R1On_PSC_netcon = eye(R1On_Npop,S2OnOff_Npop);
S2OnOff_R1On_PSC_scale = (S2OnOff_R1On_PSC_tauD/S2OnOff_R1On_PSC_tauR)^(S2OnOff_R1On_PSC_tauR/(S2OnOff_R1On_PSC_tauD-S2OnOff_R1On_PSC_tauR));
R2On_S2OnOff_PSC_netcon = eye(S2OnOff_Npop,R2On_Npop);
R2On_S2OnOff_PSC_scale = (R2On_S2OnOff_PSC_tauD/R2On_S2OnOff_PSC_tauR)^(R2On_S2OnOff_PSC_tauR/(R2On_S2OnOff_PSC_tauD-R2On_S2OnOff_PSC_tauR));
R2Off_S2OnOff_PSC_netcon = eye(S2OnOff_Npop,R2Off_Npop);
R2Off_S2OnOff_PSC_scale = (R2Off_S2OnOff_PSC_tauD/R2Off_S2OnOff_PSC_tauR)^(R2Off_S2OnOff_PSC_tauR/(R2Off_S2OnOff_PSC_tauD-R2Off_S2OnOff_PSC_tauR));
R2Off_R1Off_PSC_netcon = eye(R1Off_Npop,R2Off_Npop);
R2Off_R1Off_PSC_scale = (R2Off_R1Off_PSC_tauD/R2Off_R1Off_PSC_tauR)^(R2Off_R1Off_PSC_tauR/(R2Off_R1Off_PSC_tauD-R2Off_R1Off_PSC_tauR));
S2OnOff_R1Off_PSC_netcon = eye(R1Off_Npop,S2OnOff_Npop);
S2OnOff_R1Off_PSC_scale = (S2OnOff_R1Off_PSC_tauD/S2OnOff_R1Off_PSC_tauR)^(S2OnOff_R1Off_PSC_tauR/(S2OnOff_R1Off_PSC_tauD-S2OnOff_R1Off_PSC_tauR));
R2On_R2On_iNoise_V3_netcon =  eye(R2On_Npop,R2On_Npop);
R2On_R2On_iNoise_V3_token = genPoissonTimes(R2On_Npop,R2On_R2On_iNoise_V3_dt,R2On_R2On_iNoise_V3_FR,R2On_R2On_iNoise_V3_sigma,R2On_R2On_iNoise_V3_simlen);
R2On_R2On_iNoise_V3_scale =  (R2On_R2On_iNoise_V3_tauD_N/R2On_R2On_iNoise_V3_tauR_N)^(R2On_R2On_iNoise_V3_tauR_N/(R2On_R2On_iNoise_V3_tauD_N-R2On_R2On_iNoise_V3_tauR_N));

% ------------------------------------------------------------
% Initial conditions:
% ------------------------------------------------------------
t=0; k=1;

% STATE_VARIABLES:
On_V = zeros(2,On_Npop);
On_V(1,:) =  On_E_L*ones(1,On_Npop);
On_g_ad = zeros(2,On_Npop);
On_g_ad(1,:) =  zeros(1,On_Npop);
Off_V = zeros(2,Off_Npop);
Off_V(1,:) =  Off_E_L*ones(1,Off_Npop);
Off_g_ad = zeros(2,Off_Npop);
Off_g_ad(1,:) =  zeros(1,Off_Npop);
R1On_V = zeros(2,R1On_Npop);
R1On_V(1,:) =  R1On_E_L*ones(1,R1On_Npop);
R1On_g_ad = zeros(2,R1On_Npop);
R1On_g_ad(1,:) =  zeros(1,R1On_Npop);
R1Off_V = zeros(2,R1Off_Npop);
R1Off_V(1,:) =  R1Off_E_L*ones(1,R1Off_Npop);
R1Off_g_ad = zeros(2,R1Off_Npop);
R1Off_g_ad(1,:) =  zeros(1,R1Off_Npop);
S1OnOff_V = zeros(2,S1OnOff_Npop);
S1OnOff_V(1,:) =  S1OnOff_E_L*ones(1,S1OnOff_Npop);
S1OnOff_g_ad = zeros(2,S1OnOff_Npop);
S1OnOff_g_ad(1,:) =  zeros(1,S1OnOff_Npop);
R2On_V = zeros(2,R2On_Npop);
R2On_V(1,:) =  R2On_E_L*ones(1,R2On_Npop);
R2On_g_ad = zeros(2,R2On_Npop);
R2On_g_ad(1,:) =  zeros(1,R2On_Npop);
R2Off_V = zeros(2,R2Off_Npop);
R2Off_V(1,:) =  R2Off_E_L*ones(1,R2Off_Npop);
R2Off_g_ad = zeros(2,R2Off_Npop);
R2Off_g_ad(1,:) =  zeros(1,R2Off_Npop);
S2OnOff_V = zeros(2,S2OnOff_Npop);
S2OnOff_V(1,:) =  S2OnOff_E_L*ones(1,S2OnOff_Npop);
S2OnOff_g_ad = zeros(2,S2OnOff_Npop);
S2OnOff_g_ad(1,:) =  zeros(1,S2OnOff_Npop);
R1On_On_PSC_s = zeros(2,On_Npop);
R1On_On_PSC_s(1,:) =  zeros(1,On_Npop);
R1On_On_PSC_x = zeros(2,On_Npop);
R1On_On_PSC_x(1,:) =  zeros(1,On_Npop);
R1On_On_PSC_F = zeros(2,On_Npop);
R1On_On_PSC_F(1,:) =  ones(1,On_Npop);
R1On_On_PSC_P = zeros(2,On_Npop);
R1On_On_PSC_P(1,:) =  ones(1,On_Npop);
R1On_On_PSC_q = zeros(2,On_Npop);
R1On_On_PSC_q(1,:) =  ones(1,On_Npop);
S1OnOff_On_PSC_s = zeros(2,On_Npop);
S1OnOff_On_PSC_s(1,:) =  zeros(1,On_Npop);
S1OnOff_On_PSC_x = zeros(2,On_Npop);
S1OnOff_On_PSC_x(1,:) =  zeros(1,On_Npop);
S1OnOff_On_PSC_F = zeros(2,On_Npop);
S1OnOff_On_PSC_F(1,:) =  ones(1,On_Npop);
S1OnOff_On_PSC_P = zeros(2,On_Npop);
S1OnOff_On_PSC_P(1,:) =  ones(1,On_Npop);
S1OnOff_On_PSC_q = zeros(2,On_Npop);
S1OnOff_On_PSC_q(1,:) =  ones(1,On_Npop);
R1On_S1OnOff_PSC_s = zeros(2,S1OnOff_Npop);
R1On_S1OnOff_PSC_s(1,:) =  zeros(1,S1OnOff_Npop);
R1On_S1OnOff_PSC_x = zeros(2,S1OnOff_Npop);
R1On_S1OnOff_PSC_x(1,:) =  zeros(1,S1OnOff_Npop);
R1On_S1OnOff_PSC_F = zeros(2,S1OnOff_Npop);
R1On_S1OnOff_PSC_F(1,:) =  ones(1,S1OnOff_Npop);
R1On_S1OnOff_PSC_P = zeros(2,S1OnOff_Npop);
R1On_S1OnOff_PSC_P(1,:) =  ones(1,S1OnOff_Npop);
R1On_S1OnOff_PSC_q = zeros(2,S1OnOff_Npop);
R1On_S1OnOff_PSC_q(1,:) =  ones(1,S1OnOff_Npop);
R1Off_S1OnOff_PSC_s = zeros(2,S1OnOff_Npop);
R1Off_S1OnOff_PSC_s(1,:) =  zeros(1,S1OnOff_Npop);
R1Off_S1OnOff_PSC_x = zeros(2,S1OnOff_Npop);
R1Off_S1OnOff_PSC_x(1,:) =  zeros(1,S1OnOff_Npop);
R1Off_S1OnOff_PSC_F = zeros(2,S1OnOff_Npop);
R1Off_S1OnOff_PSC_F(1,:) =  ones(1,S1OnOff_Npop);
R1Off_S1OnOff_PSC_P = zeros(2,S1OnOff_Npop);
R1Off_S1OnOff_PSC_P(1,:) =  ones(1,S1OnOff_Npop);
R1Off_S1OnOff_PSC_q = zeros(2,S1OnOff_Npop);
R1Off_S1OnOff_PSC_q(1,:) =  ones(1,S1OnOff_Npop);
R1Off_Off_PSC_s = zeros(2,Off_Npop);
R1Off_Off_PSC_s(1,:) =  zeros(1,Off_Npop);
R1Off_Off_PSC_x = zeros(2,Off_Npop);
R1Off_Off_PSC_x(1,:) =  zeros(1,Off_Npop);
R1Off_Off_PSC_F = zeros(2,Off_Npop);
R1Off_Off_PSC_F(1,:) =  ones(1,Off_Npop);
R1Off_Off_PSC_P = zeros(2,Off_Npop);
R1Off_Off_PSC_P(1,:) =  ones(1,Off_Npop);
R1Off_Off_PSC_q = zeros(2,Off_Npop);
R1Off_Off_PSC_q(1,:) =  ones(1,Off_Npop);
S1OnOff_Off_PSC_s = zeros(2,Off_Npop);
S1OnOff_Off_PSC_s(1,:) =  zeros(1,Off_Npop);
S1OnOff_Off_PSC_x = zeros(2,Off_Npop);
S1OnOff_Off_PSC_x(1,:) =  zeros(1,Off_Npop);
S1OnOff_Off_PSC_F = zeros(2,Off_Npop);
S1OnOff_Off_PSC_F(1,:) =  ones(1,Off_Npop);
S1OnOff_Off_PSC_P = zeros(2,Off_Npop);
S1OnOff_Off_PSC_P(1,:) =  ones(1,Off_Npop);
S1OnOff_Off_PSC_q = zeros(2,Off_Npop);
S1OnOff_Off_PSC_q(1,:) =  ones(1,Off_Npop);
R2On_R1On_PSC_s = zeros(2,R1On_Npop);
R2On_R1On_PSC_s(1,:) =  zeros(1,R1On_Npop);
R2On_R1On_PSC_x = zeros(2,R1On_Npop);
R2On_R1On_PSC_x(1,:) =  zeros(1,R1On_Npop);
R2On_R1On_PSC_F = zeros(2,R1On_Npop);
R2On_R1On_PSC_F(1,:) =  ones(1,R1On_Npop);
R2On_R1On_PSC_P = zeros(2,R1On_Npop);
R2On_R1On_PSC_P(1,:) =  ones(1,R1On_Npop);
R2On_R1On_PSC_q = zeros(2,R1On_Npop);
R2On_R1On_PSC_q(1,:) =  ones(1,R1On_Npop);
S2OnOff_R1On_PSC_s = zeros(2,R1On_Npop);
S2OnOff_R1On_PSC_s(1,:) =  zeros(1,R1On_Npop);
S2OnOff_R1On_PSC_x = zeros(2,R1On_Npop);
S2OnOff_R1On_PSC_x(1,:) =  zeros(1,R1On_Npop);
S2OnOff_R1On_PSC_F = zeros(2,R1On_Npop);
S2OnOff_R1On_PSC_F(1,:) =  ones(1,R1On_Npop);
S2OnOff_R1On_PSC_P = zeros(2,R1On_Npop);
S2OnOff_R1On_PSC_P(1,:) =  ones(1,R1On_Npop);
S2OnOff_R1On_PSC_q = zeros(2,R1On_Npop);
S2OnOff_R1On_PSC_q(1,:) =  ones(1,R1On_Npop);
R2On_S2OnOff_PSC_s = zeros(2,S2OnOff_Npop);
R2On_S2OnOff_PSC_s(1,:) =  zeros(1,S2OnOff_Npop);
R2On_S2OnOff_PSC_x = zeros(2,S2OnOff_Npop);
R2On_S2OnOff_PSC_x(1,:) =  zeros(1,S2OnOff_Npop);
R2On_S2OnOff_PSC_F = zeros(2,S2OnOff_Npop);
R2On_S2OnOff_PSC_F(1,:) =  ones(1,S2OnOff_Npop);
R2On_S2OnOff_PSC_P = zeros(2,S2OnOff_Npop);
R2On_S2OnOff_PSC_P(1,:) =  ones(1,S2OnOff_Npop);
R2On_S2OnOff_PSC_q = zeros(2,S2OnOff_Npop);
R2On_S2OnOff_PSC_q(1,:) =  ones(1,S2OnOff_Npop);
R2Off_S2OnOff_PSC_s = zeros(2,S2OnOff_Npop);
R2Off_S2OnOff_PSC_s(1,:) =  zeros(1,S2OnOff_Npop);
R2Off_S2OnOff_PSC_x = zeros(2,S2OnOff_Npop);
R2Off_S2OnOff_PSC_x(1,:) =  zeros(1,S2OnOff_Npop);
R2Off_S2OnOff_PSC_F = zeros(2,S2OnOff_Npop);
R2Off_S2OnOff_PSC_F(1,:) =  ones(1,S2OnOff_Npop);
R2Off_S2OnOff_PSC_P = zeros(2,S2OnOff_Npop);
R2Off_S2OnOff_PSC_P(1,:) =  ones(1,S2OnOff_Npop);
R2Off_S2OnOff_PSC_q = zeros(2,S2OnOff_Npop);
R2Off_S2OnOff_PSC_q(1,:) =  ones(1,S2OnOff_Npop);
R2Off_R1Off_PSC_s = zeros(2,R1Off_Npop);
R2Off_R1Off_PSC_s(1,:) =  zeros(1,R1Off_Npop);
R2Off_R1Off_PSC_x = zeros(2,R1Off_Npop);
R2Off_R1Off_PSC_x(1,:) =  zeros(1,R1Off_Npop);
R2Off_R1Off_PSC_F = zeros(2,R1Off_Npop);
R2Off_R1Off_PSC_F(1,:) =  ones(1,R1Off_Npop);
R2Off_R1Off_PSC_P = zeros(2,R1Off_Npop);
R2Off_R1Off_PSC_P(1,:) =  ones(1,R1Off_Npop);
R2Off_R1Off_PSC_q = zeros(2,R1Off_Npop);
R2Off_R1Off_PSC_q(1,:) =  ones(1,R1Off_Npop);
S2OnOff_R1Off_PSC_s = zeros(2,R1Off_Npop);
S2OnOff_R1Off_PSC_s(1,:) =  zeros(1,R1Off_Npop);
S2OnOff_R1Off_PSC_x = zeros(2,R1Off_Npop);
S2OnOff_R1Off_PSC_x(1,:) =  zeros(1,R1Off_Npop);
S2OnOff_R1Off_PSC_F = zeros(2,R1Off_Npop);
S2OnOff_R1Off_PSC_F(1,:) =  ones(1,R1Off_Npop);
S2OnOff_R1Off_PSC_P = zeros(2,R1Off_Npop);
S2OnOff_R1Off_PSC_P(1,:) =  ones(1,R1Off_Npop);
S2OnOff_R1Off_PSC_q = zeros(2,R1Off_Npop);
S2OnOff_R1Off_PSC_q(1,:) =  ones(1,R1Off_Npop);
R2On_R2On_iNoise_V3_sn = zeros(2,R2On_Npop);
R2On_R2On_iNoise_V3_sn(1,:) =  0 * ones(1,R2On_Npop);
R2On_R2On_iNoise_V3_xn = zeros(2,R2On_Npop);
R2On_R2On_iNoise_V3_xn(1,:) =  0 * ones(1,R2On_Npop);

% MONITORS:
On_tspike = -1e32*ones(5,On_Npop);
On_buffer_index = ones(1,On_Npop);
On_V_spikes = zeros(nsamp,On_Npop);
Off_tspike = -1e32*ones(5,Off_Npop);
Off_buffer_index = ones(1,Off_Npop);
Off_V_spikes = zeros(nsamp,Off_Npop);
R1On_tspike = -1e32*ones(5,R1On_Npop);
R1On_buffer_index = ones(1,R1On_Npop);
R1On_V_spikes = zeros(nsamp,R1On_Npop);
R1Off_tspike = -1e32*ones(5,R1Off_Npop);
R1Off_buffer_index = ones(1,R1Off_Npop);
R1Off_V_spikes = zeros(nsamp,R1Off_Npop);
S1OnOff_tspike = -1e32*ones(5,S1OnOff_Npop);
S1OnOff_buffer_index = ones(1,S1OnOff_Npop);
S1OnOff_V_spikes = zeros(nsamp,S1OnOff_Npop);
R2On_tspike = -1e32*ones(5,R2On_Npop);
R2On_buffer_index = ones(1,R2On_Npop);
R2On_V_spikes = zeros(nsamp,R2On_Npop);
R2Off_tspike = -1e32*ones(5,R2Off_Npop);
R2Off_buffer_index = ones(1,R2Off_Npop);
R2Off_V_spikes = zeros(nsamp,R2Off_Npop);
S2OnOff_tspike = -1e32*ones(5,S2OnOff_Npop);
S2OnOff_buffer_index = ones(1,S2OnOff_Npop);
S2OnOff_V_spikes = zeros(nsamp,S2OnOff_Npop);
On_On_IC_iIC = zeros(nsamp,On_Npop);
On_On_IC_iIC(1,:)=On_On_IC_g_postIC*(On_On_IC_input(k,:)*On_On_IC_netcon).*(On_V(1,:)-On_On_IC_E_exc);
Off_Off_IC_iIC = zeros(nsamp,Off_Npop);
Off_Off_IC_iIC(1,:)=Off_Off_IC_g_postIC*(Off_Off_IC_input(k,:)*Off_Off_IC_netcon).*(Off_V(1,:)-Off_Off_IC_E_exc);
R1On_On_PSC_syn = zeros(nsamp,R1On_Npop);
R1On_On_PSC_syn(1,:)=R1On_On_PSC_gSYN.*(R1On_On_PSC_s(1,:)*R1On_On_PSC_netcon).*(R1On_V(1,:)-R1On_On_PSC_ESYN);
S1OnOff_On_PSC_syn = zeros(nsamp,S1OnOff_Npop);
S1OnOff_On_PSC_syn(1,:)=S1OnOff_On_PSC_gSYN.*(S1OnOff_On_PSC_s(1,:)*S1OnOff_On_PSC_netcon).*(S1OnOff_V(1,:)-S1OnOff_On_PSC_ESYN);
R1On_S1OnOff_PSC_syn = zeros(nsamp,R1On_Npop);
R1On_S1OnOff_PSC_syn(1,:)=R1On_S1OnOff_PSC_gSYN.*(R1On_S1OnOff_PSC_s(1,:)*R1On_S1OnOff_PSC_netcon).*(R1On_V(1,:)-R1On_S1OnOff_PSC_ESYN);
R1Off_S1OnOff_PSC_syn = zeros(nsamp,R1Off_Npop);
R1Off_S1OnOff_PSC_syn(1,:)=R1Off_S1OnOff_PSC_gSYN.*(R1Off_S1OnOff_PSC_s(1,:)*R1Off_S1OnOff_PSC_netcon).*(R1Off_V(1,:)-R1Off_S1OnOff_PSC_ESYN);
R1Off_Off_PSC_syn = zeros(nsamp,R1Off_Npop);
R1Off_Off_PSC_syn(1,:)=R1Off_Off_PSC_gSYN.*(R1Off_Off_PSC_s(1,:)*R1Off_Off_PSC_netcon).*(R1Off_V(1,:)-R1Off_Off_PSC_ESYN);
S1OnOff_Off_PSC_syn = zeros(nsamp,S1OnOff_Npop);
S1OnOff_Off_PSC_syn(1,:)=S1OnOff_Off_PSC_gSYN.*(S1OnOff_Off_PSC_s(1,:)*S1OnOff_Off_PSC_netcon).*(S1OnOff_V(1,:)-S1OnOff_Off_PSC_ESYN);
R2On_R1On_PSC_syn = zeros(nsamp,R2On_Npop);
R2On_R1On_PSC_syn(1,:)=R2On_R1On_PSC_gSYN.*(R2On_R1On_PSC_s(1,:)*R2On_R1On_PSC_netcon).*(R2On_V(1,:)-R2On_R1On_PSC_ESYN);
S2OnOff_R1On_PSC_syn = zeros(nsamp,S2OnOff_Npop);
S2OnOff_R1On_PSC_syn(1,:)=S2OnOff_R1On_PSC_gSYN.*(S2OnOff_R1On_PSC_s(1,:)*S2OnOff_R1On_PSC_netcon).*(S2OnOff_V(1,:)-S2OnOff_R1On_PSC_ESYN);
R2On_S2OnOff_PSC_syn = zeros(nsamp,R2On_Npop);
R2On_S2OnOff_PSC_syn(1,:)=R2On_S2OnOff_PSC_gSYN.*(R2On_S2OnOff_PSC_s(1,:)*R2On_S2OnOff_PSC_netcon).*(R2On_V(1,:)-R2On_S2OnOff_PSC_ESYN);
R2Off_S2OnOff_PSC_syn = zeros(nsamp,R2Off_Npop);
R2Off_S2OnOff_PSC_syn(1,:)=R2Off_S2OnOff_PSC_gSYN.*(R2Off_S2OnOff_PSC_s(1,:)*R2Off_S2OnOff_PSC_netcon).*(R2Off_V(1,:)-R2Off_S2OnOff_PSC_ESYN);
R2Off_R1Off_PSC_syn = zeros(nsamp,R2Off_Npop);
R2Off_R1Off_PSC_syn(1,:)=R2Off_R1Off_PSC_gSYN.*(R2Off_R1Off_PSC_s(1,:)*R2Off_R1Off_PSC_netcon).*(R2Off_V(1,:)-R2Off_R1Off_PSC_ESYN);
S2OnOff_R1Off_PSC_syn = zeros(nsamp,S2OnOff_Npop);
S2OnOff_R1Off_PSC_syn(1,:)=S2OnOff_R1Off_PSC_gSYN.*(S2OnOff_R1Off_PSC_s(1,:)*S2OnOff_R1Off_PSC_netcon).*(S2OnOff_V(1,:)-S2OnOff_R1Off_PSC_ESYN);

% ###########################################################
% Numerical integration:
% ###########################################################
n=2;
for k=2:ntime
  t=T(k-1);
  On_V_k1 = ( (On_E_L-On_V(n-1,:)) - On_R*On_g_ad(n-1,:).*(On_V(n-1,:)-On_E_k) - On_R*((((On_On_IC_g_postIC*(On_On_IC_input(k,:)*On_On_IC_netcon).*(On_V(n-1,:)-On_On_IC_E_exc))))) + On_R*On_Itonic.*On_Imask + On_R*On_noise.*randn(1,On_Npop) ) / On_tau;
  On_g_ad_k1 = -On_g_ad(n-1,:) / On_tau_ad;
  Off_V_k1 = ( (Off_E_L-Off_V(n-1,:)) - Off_R*Off_g_ad(n-1,:).*(Off_V(n-1,:)-Off_E_k) - Off_R*((((Off_Off_IC_g_postIC*(Off_Off_IC_input(k,:)*Off_Off_IC_netcon).*(Off_V(n-1,:)-Off_Off_IC_E_exc))))) + Off_R*Off_Itonic.*Off_Imask + Off_R*Off_noise.*randn(1,Off_Npop) ) / Off_tau;
  Off_g_ad_k1 = -Off_g_ad(n-1,:) / Off_tau_ad;
  R1On_V_k1 = ( (R1On_E_L-R1On_V(n-1,:)) - R1On_R*R1On_g_ad(n-1,:).*(R1On_V(n-1,:)-R1On_E_k) - R1On_R*((((R1On_On_PSC_gSYN.*(R1On_On_PSC_s(n-1,:)*R1On_On_PSC_netcon).*(R1On_V(n-1,:)-R1On_On_PSC_ESYN))))+((((R1On_S1OnOff_PSC_gSYN.*(R1On_S1OnOff_PSC_s(n-1,:)*R1On_S1OnOff_PSC_netcon).*(R1On_V(n-1,:)-R1On_S1OnOff_PSC_ESYN)))))) + R1On_R*R1On_Itonic.*R1On_Imask + R1On_R*R1On_noise.*randn(1,R1On_Npop) ) / R1On_tau;
  R1On_g_ad_k1 = -R1On_g_ad(n-1,:) / R1On_tau_ad;
  R1Off_V_k1 = ( (R1Off_E_L-R1Off_V(n-1,:)) - R1Off_R*R1Off_g_ad(n-1,:).*(R1Off_V(n-1,:)-R1Off_E_k) - R1Off_R*((((R1Off_S1OnOff_PSC_gSYN.*(R1Off_S1OnOff_PSC_s(n-1,:)*R1Off_S1OnOff_PSC_netcon).*(R1Off_V(n-1,:)-R1Off_S1OnOff_PSC_ESYN))))+((((R1Off_Off_PSC_gSYN.*(R1Off_Off_PSC_s(n-1,:)*R1Off_Off_PSC_netcon).*(R1Off_V(n-1,:)-R1Off_Off_PSC_ESYN)))))) + R1Off_R*R1Off_Itonic.*R1Off_Imask + R1Off_R*R1Off_noise.*randn(1,R1Off_Npop) ) / R1Off_tau;
  R1Off_g_ad_k1 = -R1Off_g_ad(n-1,:) / R1Off_tau_ad;
  S1OnOff_V_k1 = ( (S1OnOff_E_L-S1OnOff_V(n-1,:)) - S1OnOff_R*S1OnOff_g_ad(n-1,:).*(S1OnOff_V(n-1,:)-S1OnOff_E_k) - S1OnOff_R*((((S1OnOff_On_PSC_gSYN.*(S1OnOff_On_PSC_s(n-1,:)*S1OnOff_On_PSC_netcon).*(S1OnOff_V(n-1,:)-S1OnOff_On_PSC_ESYN))))+((((S1OnOff_Off_PSC_gSYN.*(S1OnOff_Off_PSC_s(n-1,:)*S1OnOff_Off_PSC_netcon).*(S1OnOff_V(n-1,:)-S1OnOff_Off_PSC_ESYN)))))) + S1OnOff_R*S1OnOff_Itonic.*S1OnOff_Imask + S1OnOff_R*S1OnOff_noise.*randn(1,S1OnOff_Npop) ) / S1OnOff_tau;
  S1OnOff_g_ad_k1 = -S1OnOff_g_ad(n-1,:) / S1OnOff_tau_ad;
  R2On_V_k1 = ( (R2On_E_L-R2On_V(n-1,:)) - R2On_R*R2On_g_ad(n-1,:).*(R2On_V(n-1,:)-R2On_E_k) - R2On_R*((((R2On_R1On_PSC_gSYN.*(R2On_R1On_PSC_s(n-1,:)*R2On_R1On_PSC_netcon).*(R2On_V(n-1,:)-R2On_R1On_PSC_ESYN))))+((((R2On_S2OnOff_PSC_gSYN.*(R2On_S2OnOff_PSC_s(n-1,:)*R2On_S2OnOff_PSC_netcon).*(R2On_V(n-1,:)-R2On_S2OnOff_PSC_ESYN))))+((((R2On_R2On_iNoise_V3_nSYN.*(R2On_R2On_iNoise_V3_sn(n-1,:)*R2On_R2On_iNoise_V3_netcon).*(R2On_V(n-1,:)-R2On_R2On_iNoise_V3_E_exc))))))) + R2On_R*R2On_Itonic.*R2On_Imask + R2On_R*R2On_noise.*randn(1,R2On_Npop) ) / R2On_tau;
  R2On_g_ad_k1 = -R2On_g_ad(n-1,:) / R2On_tau_ad;
  R2Off_V_k1 = ( (R2Off_E_L-R2Off_V(n-1,:)) - R2Off_R*R2Off_g_ad(n-1,:).*(R2Off_V(n-1,:)-R2Off_E_k) - R2Off_R*((((R2Off_S2OnOff_PSC_gSYN.*(R2Off_S2OnOff_PSC_s(n-1,:)*R2Off_S2OnOff_PSC_netcon).*(R2Off_V(n-1,:)-R2Off_S2OnOff_PSC_ESYN))))+((((R2Off_R1Off_PSC_gSYN.*(R2Off_R1Off_PSC_s(n-1,:)*R2Off_R1Off_PSC_netcon).*(R2Off_V(n-1,:)-R2Off_R1Off_PSC_ESYN)))))) + R2Off_R*R2Off_Itonic.*R2Off_Imask + R2Off_R*R2Off_noise.*randn(1,R2Off_Npop) ) / R2Off_tau;
  R2Off_g_ad_k1 = -R2Off_g_ad(n-1,:) / R2Off_tau_ad;
  S2OnOff_V_k1 = ( (S2OnOff_E_L-S2OnOff_V(n-1,:)) - S2OnOff_R*S2OnOff_g_ad(n-1,:).*(S2OnOff_V(n-1,:)-S2OnOff_E_k) - S2OnOff_R*((((S2OnOff_R1On_PSC_gSYN.*(S2OnOff_R1On_PSC_s(n-1,:)*S2OnOff_R1On_PSC_netcon).*(S2OnOff_V(n-1,:)-S2OnOff_R1On_PSC_ESYN))))+((((S2OnOff_R1Off_PSC_gSYN.*(S2OnOff_R1Off_PSC_s(n-1,:)*S2OnOff_R1Off_PSC_netcon).*(S2OnOff_V(n-1,:)-S2OnOff_R1Off_PSC_ESYN)))))) + S2OnOff_R*S2OnOff_Itonic.*S2OnOff_Imask + S2OnOff_R*S2OnOff_noise.*randn(1,S2OnOff_Npop) ) / S2OnOff_tau;
  S2OnOff_g_ad_k1 = -S2OnOff_g_ad(n-1,:) / S2OnOff_tau_ad;
  R1On_On_PSC_s_k1 = ( R1On_On_PSC_scale * R1On_On_PSC_x(n-1,:) - R1On_On_PSC_s(n-1,:) )/R1On_On_PSC_tauR;
  R1On_On_PSC_x_k1 = -R1On_On_PSC_x(n-1,:)/R1On_On_PSC_tauD;
  R1On_On_PSC_F_k1 = (1 - R1On_On_PSC_F(n-1,:))/R1On_On_PSC_tauF;
  R1On_On_PSC_P_k1 = (1 - R1On_On_PSC_P(n-1,:))/R1On_On_PSC_tauP;
  R1On_On_PSC_q_k1 = 0;
  S1OnOff_On_PSC_s_k1 = ( S1OnOff_On_PSC_scale * S1OnOff_On_PSC_x(n-1,:) - S1OnOff_On_PSC_s(n-1,:) )/S1OnOff_On_PSC_tauR;
  S1OnOff_On_PSC_x_k1 = -S1OnOff_On_PSC_x(n-1,:)/S1OnOff_On_PSC_tauD;
  S1OnOff_On_PSC_F_k1 = (1 - S1OnOff_On_PSC_F(n-1,:))/S1OnOff_On_PSC_tauF;
  S1OnOff_On_PSC_P_k1 = (1 - S1OnOff_On_PSC_P(n-1,:))/S1OnOff_On_PSC_tauP;
  S1OnOff_On_PSC_q_k1 = 0;
  R1On_S1OnOff_PSC_s_k1 = ( R1On_S1OnOff_PSC_scale * R1On_S1OnOff_PSC_x(n-1,:) - R1On_S1OnOff_PSC_s(n-1,:) )/R1On_S1OnOff_PSC_tauR;
  R1On_S1OnOff_PSC_x_k1 = -R1On_S1OnOff_PSC_x(n-1,:)/R1On_S1OnOff_PSC_tauD;
  R1On_S1OnOff_PSC_F_k1 = (1 - R1On_S1OnOff_PSC_F(n-1,:))/R1On_S1OnOff_PSC_tauF;
  R1On_S1OnOff_PSC_P_k1 = (1 - R1On_S1OnOff_PSC_P(n-1,:))/R1On_S1OnOff_PSC_tauP;
  R1On_S1OnOff_PSC_q_k1 = 0;
  R1Off_S1OnOff_PSC_s_k1 = ( R1Off_S1OnOff_PSC_scale * R1Off_S1OnOff_PSC_x(n-1,:) - R1Off_S1OnOff_PSC_s(n-1,:) )/R1Off_S1OnOff_PSC_tauR;
  R1Off_S1OnOff_PSC_x_k1 = -R1Off_S1OnOff_PSC_x(n-1,:)/R1Off_S1OnOff_PSC_tauD;
  R1Off_S1OnOff_PSC_F_k1 = (1 - R1Off_S1OnOff_PSC_F(n-1,:))/R1Off_S1OnOff_PSC_tauF;
  R1Off_S1OnOff_PSC_P_k1 = (1 - R1Off_S1OnOff_PSC_P(n-1,:))/R1Off_S1OnOff_PSC_tauP;
  R1Off_S1OnOff_PSC_q_k1 = 0;
  R1Off_Off_PSC_s_k1 = ( R1Off_Off_PSC_scale * R1Off_Off_PSC_x(n-1,:) - R1Off_Off_PSC_s(n-1,:) )/R1Off_Off_PSC_tauR;
  R1Off_Off_PSC_x_k1 = -R1Off_Off_PSC_x(n-1,:)/R1Off_Off_PSC_tauD;
  R1Off_Off_PSC_F_k1 = (1 - R1Off_Off_PSC_F(n-1,:))/R1Off_Off_PSC_tauF;
  R1Off_Off_PSC_P_k1 = (1 - R1Off_Off_PSC_P(n-1,:))/R1Off_Off_PSC_tauP;
  R1Off_Off_PSC_q_k1 = 0;
  S1OnOff_Off_PSC_s_k1 = ( S1OnOff_Off_PSC_scale * S1OnOff_Off_PSC_x(n-1,:) - S1OnOff_Off_PSC_s(n-1,:) )/S1OnOff_Off_PSC_tauR;
  S1OnOff_Off_PSC_x_k1 = -S1OnOff_Off_PSC_x(n-1,:)/S1OnOff_Off_PSC_tauD;
  S1OnOff_Off_PSC_F_k1 = (1 - S1OnOff_Off_PSC_F(n-1,:))/S1OnOff_Off_PSC_tauF;
  S1OnOff_Off_PSC_P_k1 = (1 - S1OnOff_Off_PSC_P(n-1,:))/S1OnOff_Off_PSC_tauP;
  S1OnOff_Off_PSC_q_k1 = 0;
  R2On_R1On_PSC_s_k1 = ( R2On_R1On_PSC_scale * R2On_R1On_PSC_x(n-1,:) - R2On_R1On_PSC_s(n-1,:) )/R2On_R1On_PSC_tauR;
  R2On_R1On_PSC_x_k1 = -R2On_R1On_PSC_x(n-1,:)/R2On_R1On_PSC_tauD;
  R2On_R1On_PSC_F_k1 = (1 - R2On_R1On_PSC_F(n-1,:))/R2On_R1On_PSC_tauF;
  R2On_R1On_PSC_P_k1 = (1 - R2On_R1On_PSC_P(n-1,:))/R2On_R1On_PSC_tauP;
  R2On_R1On_PSC_q_k1 = 0;
  S2OnOff_R1On_PSC_s_k1 = ( S2OnOff_R1On_PSC_scale * S2OnOff_R1On_PSC_x(n-1,:) - S2OnOff_R1On_PSC_s(n-1,:) )/S2OnOff_R1On_PSC_tauR;
  S2OnOff_R1On_PSC_x_k1 = -S2OnOff_R1On_PSC_x(n-1,:)/S2OnOff_R1On_PSC_tauD;
  S2OnOff_R1On_PSC_F_k1 = (1 - S2OnOff_R1On_PSC_F(n-1,:))/S2OnOff_R1On_PSC_tauF;
  S2OnOff_R1On_PSC_P_k1 = (1 - S2OnOff_R1On_PSC_P(n-1,:))/S2OnOff_R1On_PSC_tauP;
  S2OnOff_R1On_PSC_q_k1 = 0;
  R2On_S2OnOff_PSC_s_k1 = ( R2On_S2OnOff_PSC_scale * R2On_S2OnOff_PSC_x(n-1,:) - R2On_S2OnOff_PSC_s(n-1,:) )/R2On_S2OnOff_PSC_tauR;
  R2On_S2OnOff_PSC_x_k1 = -R2On_S2OnOff_PSC_x(n-1,:)/R2On_S2OnOff_PSC_tauD;
  R2On_S2OnOff_PSC_F_k1 = (1 - R2On_S2OnOff_PSC_F(n-1,:))/R2On_S2OnOff_PSC_tauF;
  R2On_S2OnOff_PSC_P_k1 = (1 - R2On_S2OnOff_PSC_P(n-1,:))/R2On_S2OnOff_PSC_tauP;
  R2On_S2OnOff_PSC_q_k1 = 0;
  R2Off_S2OnOff_PSC_s_k1 = ( R2Off_S2OnOff_PSC_scale * R2Off_S2OnOff_PSC_x(n-1,:) - R2Off_S2OnOff_PSC_s(n-1,:) )/R2Off_S2OnOff_PSC_tauR;
  R2Off_S2OnOff_PSC_x_k1 = -R2Off_S2OnOff_PSC_x(n-1,:)/R2Off_S2OnOff_PSC_tauD;
  R2Off_S2OnOff_PSC_F_k1 = (1 - R2Off_S2OnOff_PSC_F(n-1,:))/R2Off_S2OnOff_PSC_tauF;
  R2Off_S2OnOff_PSC_P_k1 = (1 - R2Off_S2OnOff_PSC_P(n-1,:))/R2Off_S2OnOff_PSC_tauP;
  R2Off_S2OnOff_PSC_q_k1 = 0;
  R2Off_R1Off_PSC_s_k1 = ( R2Off_R1Off_PSC_scale * R2Off_R1Off_PSC_x(n-1,:) - R2Off_R1Off_PSC_s(n-1,:) )/R2Off_R1Off_PSC_tauR;
  R2Off_R1Off_PSC_x_k1 = -R2Off_R1Off_PSC_x(n-1,:)/R2Off_R1Off_PSC_tauD;
  R2Off_R1Off_PSC_F_k1 = (1 - R2Off_R1Off_PSC_F(n-1,:))/R2Off_R1Off_PSC_tauF;
  R2Off_R1Off_PSC_P_k1 = (1 - R2Off_R1Off_PSC_P(n-1,:))/R2Off_R1Off_PSC_tauP;
  R2Off_R1Off_PSC_q_k1 = 0;
  S2OnOff_R1Off_PSC_s_k1 = ( S2OnOff_R1Off_PSC_scale * S2OnOff_R1Off_PSC_x(n-1,:) - S2OnOff_R1Off_PSC_s(n-1,:) )/S2OnOff_R1Off_PSC_tauR;
  S2OnOff_R1Off_PSC_x_k1 = -S2OnOff_R1Off_PSC_x(n-1,:)/S2OnOff_R1Off_PSC_tauD;
  S2OnOff_R1Off_PSC_F_k1 = (1 - S2OnOff_R1Off_PSC_F(n-1,:))/S2OnOff_R1Off_PSC_tauF;
  S2OnOff_R1Off_PSC_P_k1 = (1 - S2OnOff_R1Off_PSC_P(n-1,:))/S2OnOff_R1Off_PSC_tauP;
  S2OnOff_R1Off_PSC_q_k1 = 0;
  R2On_R2On_iNoise_V3_sn_k1 = ( R2On_R2On_iNoise_V3_scale * R2On_R2On_iNoise_V3_xn(n-1,:) - R2On_R2On_iNoise_V3_sn(n-1,:) )/R2On_R2On_iNoise_V3_tauR_N;
  R2On_R2On_iNoise_V3_xn_k1 = -R2On_R2On_iNoise_V3_xn(n-1,:)/R2On_R2On_iNoise_V3_tauD_N + R2On_R2On_iNoise_V3_token(k,:)/R2On_R2On_iNoise_V3_dt;

  % ------------------------------------------------------------
  % Update state variables:
  % ------------------------------------------------------------
  On_V(n,:) = On_V(n-1,:)+dt*On_V_k1;
  On_g_ad(n,:) = On_g_ad(n-1,:)+dt*On_g_ad_k1;
  Off_V(n,:) = Off_V(n-1,:)+dt*Off_V_k1;
  Off_g_ad(n,:) = Off_g_ad(n-1,:)+dt*Off_g_ad_k1;
  R1On_V(n,:) = R1On_V(n-1,:)+dt*R1On_V_k1;
  R1On_g_ad(n,:) = R1On_g_ad(n-1,:)+dt*R1On_g_ad_k1;
  R1Off_V(n,:) = R1Off_V(n-1,:)+dt*R1Off_V_k1;
  R1Off_g_ad(n,:) = R1Off_g_ad(n-1,:)+dt*R1Off_g_ad_k1;
  S1OnOff_V(n,:) = S1OnOff_V(n-1,:)+dt*S1OnOff_V_k1;
  S1OnOff_g_ad(n,:) = S1OnOff_g_ad(n-1,:)+dt*S1OnOff_g_ad_k1;
  R2On_V(n,:) = R2On_V(n-1,:)+dt*R2On_V_k1;
  R2On_g_ad(n,:) = R2On_g_ad(n-1,:)+dt*R2On_g_ad_k1;
  R2Off_V(n,:) = R2Off_V(n-1,:)+dt*R2Off_V_k1;
  R2Off_g_ad(n,:) = R2Off_g_ad(n-1,:)+dt*R2Off_g_ad_k1;
  S2OnOff_V(n,:) = S2OnOff_V(n-1,:)+dt*S2OnOff_V_k1;
  S2OnOff_g_ad(n,:) = S2OnOff_g_ad(n-1,:)+dt*S2OnOff_g_ad_k1;
  R1On_On_PSC_s(n,:) = R1On_On_PSC_s(n-1,:)+dt*R1On_On_PSC_s_k1;
  R1On_On_PSC_x(n,:) = R1On_On_PSC_x(n-1,:)+dt*R1On_On_PSC_x_k1;
  R1On_On_PSC_F(n,:) = R1On_On_PSC_F(n-1,:)+dt*R1On_On_PSC_F_k1;
  R1On_On_PSC_P(n,:) = R1On_On_PSC_P(n-1,:)+dt*R1On_On_PSC_P_k1;
  R1On_On_PSC_q(n,:) = R1On_On_PSC_q(n-1,:)+dt*R1On_On_PSC_q_k1;
  S1OnOff_On_PSC_s(n,:) = S1OnOff_On_PSC_s(n-1,:)+dt*S1OnOff_On_PSC_s_k1;
  S1OnOff_On_PSC_x(n,:) = S1OnOff_On_PSC_x(n-1,:)+dt*S1OnOff_On_PSC_x_k1;
  S1OnOff_On_PSC_F(n,:) = S1OnOff_On_PSC_F(n-1,:)+dt*S1OnOff_On_PSC_F_k1;
  S1OnOff_On_PSC_P(n,:) = S1OnOff_On_PSC_P(n-1,:)+dt*S1OnOff_On_PSC_P_k1;
  S1OnOff_On_PSC_q(n,:) = S1OnOff_On_PSC_q(n-1,:)+dt*S1OnOff_On_PSC_q_k1;
  R1On_S1OnOff_PSC_s(n,:) = R1On_S1OnOff_PSC_s(n-1,:)+dt*R1On_S1OnOff_PSC_s_k1;
  R1On_S1OnOff_PSC_x(n,:) = R1On_S1OnOff_PSC_x(n-1,:)+dt*R1On_S1OnOff_PSC_x_k1;
  R1On_S1OnOff_PSC_F(n,:) = R1On_S1OnOff_PSC_F(n-1,:)+dt*R1On_S1OnOff_PSC_F_k1;
  R1On_S1OnOff_PSC_P(n,:) = R1On_S1OnOff_PSC_P(n-1,:)+dt*R1On_S1OnOff_PSC_P_k1;
  R1On_S1OnOff_PSC_q(n,:) = R1On_S1OnOff_PSC_q(n-1,:)+dt*R1On_S1OnOff_PSC_q_k1;
  R1Off_S1OnOff_PSC_s(n,:) = R1Off_S1OnOff_PSC_s(n-1,:)+dt*R1Off_S1OnOff_PSC_s_k1;
  R1Off_S1OnOff_PSC_x(n,:) = R1Off_S1OnOff_PSC_x(n-1,:)+dt*R1Off_S1OnOff_PSC_x_k1;
  R1Off_S1OnOff_PSC_F(n,:) = R1Off_S1OnOff_PSC_F(n-1,:)+dt*R1Off_S1OnOff_PSC_F_k1;
  R1Off_S1OnOff_PSC_P(n,:) = R1Off_S1OnOff_PSC_P(n-1,:)+dt*R1Off_S1OnOff_PSC_P_k1;
  R1Off_S1OnOff_PSC_q(n,:) = R1Off_S1OnOff_PSC_q(n-1,:)+dt*R1Off_S1OnOff_PSC_q_k1;
  R1Off_Off_PSC_s(n,:) = R1Off_Off_PSC_s(n-1,:)+dt*R1Off_Off_PSC_s_k1;
  R1Off_Off_PSC_x(n,:) = R1Off_Off_PSC_x(n-1,:)+dt*R1Off_Off_PSC_x_k1;
  R1Off_Off_PSC_F(n,:) = R1Off_Off_PSC_F(n-1,:)+dt*R1Off_Off_PSC_F_k1;
  R1Off_Off_PSC_P(n,:) = R1Off_Off_PSC_P(n-1,:)+dt*R1Off_Off_PSC_P_k1;
  R1Off_Off_PSC_q(n,:) = R1Off_Off_PSC_q(n-1,:)+dt*R1Off_Off_PSC_q_k1;
  S1OnOff_Off_PSC_s(n,:) = S1OnOff_Off_PSC_s(n-1,:)+dt*S1OnOff_Off_PSC_s_k1;
  S1OnOff_Off_PSC_x(n,:) = S1OnOff_Off_PSC_x(n-1,:)+dt*S1OnOff_Off_PSC_x_k1;
  S1OnOff_Off_PSC_F(n,:) = S1OnOff_Off_PSC_F(n-1,:)+dt*S1OnOff_Off_PSC_F_k1;
  S1OnOff_Off_PSC_P(n,:) = S1OnOff_Off_PSC_P(n-1,:)+dt*S1OnOff_Off_PSC_P_k1;
  S1OnOff_Off_PSC_q(n,:) = S1OnOff_Off_PSC_q(n-1,:)+dt*S1OnOff_Off_PSC_q_k1;
  R2On_R1On_PSC_s(n,:) = R2On_R1On_PSC_s(n-1,:)+dt*R2On_R1On_PSC_s_k1;
  R2On_R1On_PSC_x(n,:) = R2On_R1On_PSC_x(n-1,:)+dt*R2On_R1On_PSC_x_k1;
  R2On_R1On_PSC_F(n,:) = R2On_R1On_PSC_F(n-1,:)+dt*R2On_R1On_PSC_F_k1;
  R2On_R1On_PSC_P(n,:) = R2On_R1On_PSC_P(n-1,:)+dt*R2On_R1On_PSC_P_k1;
  R2On_R1On_PSC_q(n,:) = R2On_R1On_PSC_q(n-1,:)+dt*R2On_R1On_PSC_q_k1;
  S2OnOff_R1On_PSC_s(n,:) = S2OnOff_R1On_PSC_s(n-1,:)+dt*S2OnOff_R1On_PSC_s_k1;
  S2OnOff_R1On_PSC_x(n,:) = S2OnOff_R1On_PSC_x(n-1,:)+dt*S2OnOff_R1On_PSC_x_k1;
  S2OnOff_R1On_PSC_F(n,:) = S2OnOff_R1On_PSC_F(n-1,:)+dt*S2OnOff_R1On_PSC_F_k1;
  S2OnOff_R1On_PSC_P(n,:) = S2OnOff_R1On_PSC_P(n-1,:)+dt*S2OnOff_R1On_PSC_P_k1;
  S2OnOff_R1On_PSC_q(n,:) = S2OnOff_R1On_PSC_q(n-1,:)+dt*S2OnOff_R1On_PSC_q_k1;
  R2On_S2OnOff_PSC_s(n,:) = R2On_S2OnOff_PSC_s(n-1,:)+dt*R2On_S2OnOff_PSC_s_k1;
  R2On_S2OnOff_PSC_x(n,:) = R2On_S2OnOff_PSC_x(n-1,:)+dt*R2On_S2OnOff_PSC_x_k1;
  R2On_S2OnOff_PSC_F(n,:) = R2On_S2OnOff_PSC_F(n-1,:)+dt*R2On_S2OnOff_PSC_F_k1;
  R2On_S2OnOff_PSC_P(n,:) = R2On_S2OnOff_PSC_P(n-1,:)+dt*R2On_S2OnOff_PSC_P_k1;
  R2On_S2OnOff_PSC_q(n,:) = R2On_S2OnOff_PSC_q(n-1,:)+dt*R2On_S2OnOff_PSC_q_k1;
  R2Off_S2OnOff_PSC_s(n,:) = R2Off_S2OnOff_PSC_s(n-1,:)+dt*R2Off_S2OnOff_PSC_s_k1;
  R2Off_S2OnOff_PSC_x(n,:) = R2Off_S2OnOff_PSC_x(n-1,:)+dt*R2Off_S2OnOff_PSC_x_k1;
  R2Off_S2OnOff_PSC_F(n,:) = R2Off_S2OnOff_PSC_F(n-1,:)+dt*R2Off_S2OnOff_PSC_F_k1;
  R2Off_S2OnOff_PSC_P(n,:) = R2Off_S2OnOff_PSC_P(n-1,:)+dt*R2Off_S2OnOff_PSC_P_k1;
  R2Off_S2OnOff_PSC_q(n,:) = R2Off_S2OnOff_PSC_q(n-1,:)+dt*R2Off_S2OnOff_PSC_q_k1;
  R2Off_R1Off_PSC_s(n,:) = R2Off_R1Off_PSC_s(n-1,:)+dt*R2Off_R1Off_PSC_s_k1;
  R2Off_R1Off_PSC_x(n,:) = R2Off_R1Off_PSC_x(n-1,:)+dt*R2Off_R1Off_PSC_x_k1;
  R2Off_R1Off_PSC_F(n,:) = R2Off_R1Off_PSC_F(n-1,:)+dt*R2Off_R1Off_PSC_F_k1;
  R2Off_R1Off_PSC_P(n,:) = R2Off_R1Off_PSC_P(n-1,:)+dt*R2Off_R1Off_PSC_P_k1;
  R2Off_R1Off_PSC_q(n,:) = R2Off_R1Off_PSC_q(n-1,:)+dt*R2Off_R1Off_PSC_q_k1;
  S2OnOff_R1Off_PSC_s(n,:) = S2OnOff_R1Off_PSC_s(n-1,:)+dt*S2OnOff_R1Off_PSC_s_k1;
  S2OnOff_R1Off_PSC_x(n,:) = S2OnOff_R1Off_PSC_x(n-1,:)+dt*S2OnOff_R1Off_PSC_x_k1;
  S2OnOff_R1Off_PSC_F(n,:) = S2OnOff_R1Off_PSC_F(n-1,:)+dt*S2OnOff_R1Off_PSC_F_k1;
  S2OnOff_R1Off_PSC_P(n,:) = S2OnOff_R1Off_PSC_P(n-1,:)+dt*S2OnOff_R1Off_PSC_P_k1;
  S2OnOff_R1Off_PSC_q(n,:) = S2OnOff_R1Off_PSC_q(n-1,:)+dt*S2OnOff_R1Off_PSC_q_k1;
  R2On_R2On_iNoise_V3_sn(n,:) = R2On_R2On_iNoise_V3_sn(n-1,:)+dt*R2On_R2On_iNoise_V3_sn_k1;
  R2On_R2On_iNoise_V3_xn(n,:) = R2On_R2On_iNoise_V3_xn(n-1,:)+dt*R2On_R2On_iNoise_V3_xn_k1;

  % ------------------------------------------------------------
  % Conditional actions:
  % ------------------------------------------------------------
  conditional_test=any(S2OnOff_V(n,:)>=S2OnOff_V_thresh&S2OnOff_V(n-1,:)<S2OnOff_V_thresh);
  conditional_indx=(S2OnOff_V(n,:)>=S2OnOff_V_thresh&S2OnOff_V(n-1,:)<S2OnOff_V_thresh);
  if conditional_test, S2OnOff_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); S2OnOff_tspike(S2OnOff_buffer_index(i),i)=t; S2OnOff_buffer_index(i)=mod(-1+(S2OnOff_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(R2Off_V(n,:)>=R2Off_V_thresh&R2Off_V(n-1,:)<R2Off_V_thresh);
  conditional_indx=(R2Off_V(n,:)>=R2Off_V_thresh&R2Off_V(n-1,:)<R2Off_V_thresh);
  if conditional_test, R2Off_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); R2Off_tspike(R2Off_buffer_index(i),i)=t; R2Off_buffer_index(i)=mod(-1+(R2Off_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(R2On_V(n,:)>=R2On_V_thresh&R2On_V(n-1,:)<R2On_V_thresh);
  conditional_indx=(R2On_V(n,:)>=R2On_V_thresh&R2On_V(n-1,:)<R2On_V_thresh);
  if conditional_test, R2On_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); R2On_tspike(R2On_buffer_index(i),i)=t; R2On_buffer_index(i)=mod(-1+(R2On_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(S1OnOff_V(n,:)>=S1OnOff_V_thresh&S1OnOff_V(n-1,:)<S1OnOff_V_thresh);
  conditional_indx=(S1OnOff_V(n,:)>=S1OnOff_V_thresh&S1OnOff_V(n-1,:)<S1OnOff_V_thresh);
  if conditional_test, S1OnOff_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); S1OnOff_tspike(S1OnOff_buffer_index(i),i)=t; S1OnOff_buffer_index(i)=mod(-1+(S1OnOff_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(R1Off_V(n,:)>=R1Off_V_thresh&R1Off_V(n-1,:)<R1Off_V_thresh);
  conditional_indx=(R1Off_V(n,:)>=R1Off_V_thresh&R1Off_V(n-1,:)<R1Off_V_thresh);
  if conditional_test, R1Off_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); R1Off_tspike(R1Off_buffer_index(i),i)=t; R1Off_buffer_index(i)=mod(-1+(R1Off_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(R1On_V(n,:)>=R1On_V_thresh&R1On_V(n-1,:)<R1On_V_thresh);
  conditional_indx=(R1On_V(n,:)>=R1On_V_thresh&R1On_V(n-1,:)<R1On_V_thresh);
  if conditional_test, R1On_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); R1On_tspike(R1On_buffer_index(i),i)=t; R1On_buffer_index(i)=mod(-1+(R1On_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(Off_V(n,:)>=Off_V_thresh&Off_V(n-1,:)<Off_V_thresh);
  conditional_indx=(Off_V(n,:)>=Off_V_thresh&Off_V(n-1,:)<Off_V_thresh);
  if conditional_test, Off_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); Off_tspike(Off_buffer_index(i),i)=t; Off_buffer_index(i)=mod(-1+(Off_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(On_V(n,:)>=On_V_thresh&On_V(n-1,:)<On_V_thresh);
  conditional_indx=(On_V(n,:)>=On_V_thresh&On_V(n-1,:)<On_V_thresh);
  if conditional_test, On_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); On_tspike(On_buffer_index(i),i)=t; On_buffer_index(i)=mod(-1+(On_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(On_V(n,:) > On_V_thresh);
  conditional_indx=(On_V(n,:) > On_V_thresh);
  if conditional_test, On_V(n,conditional_indx) = On_V_reset; On_g_ad(n,conditional_indx) = On_g_ad(n,conditional_indx) + On_g_inc; end
  conditional_test=any(any(t<=On_tspike+On_t_ref,1));
  conditional_indx=(any(t<=On_tspike+On_t_ref,1));
  if conditional_test, On_V(n,conditional_indx) = On_V_reset; end
  conditional_test=any(Off_V(n,:) > Off_V_thresh);
  conditional_indx=(Off_V(n,:) > Off_V_thresh);
  if conditional_test, Off_V(n,conditional_indx) = Off_V_reset; Off_g_ad(n,conditional_indx) = Off_g_ad(n,conditional_indx) + Off_g_inc; end
  conditional_test=any(any(t<=Off_tspike+Off_t_ref,1));
  conditional_indx=(any(t<=Off_tspike+Off_t_ref,1));
  if conditional_test, Off_V(n,conditional_indx) = Off_V_reset; end
  conditional_test=any(R1On_V(n,:) > R1On_V_thresh);
  conditional_indx=(R1On_V(n,:) > R1On_V_thresh);
  if conditional_test, R1On_V(n,conditional_indx) = R1On_V_reset; R1On_g_ad(n,conditional_indx) = R1On_g_ad(n,conditional_indx) + R1On_g_inc; end
  conditional_test=any(any(t<=R1On_tspike+R1On_t_ref,1));
  conditional_indx=(any(t<=R1On_tspike+R1On_t_ref,1));
  if conditional_test, R1On_V(n,conditional_indx) = R1On_V_reset; end
  conditional_test=any(R1Off_V(n,:) > R1Off_V_thresh);
  conditional_indx=(R1Off_V(n,:) > R1Off_V_thresh);
  if conditional_test, R1Off_V(n,conditional_indx) = R1Off_V_reset; R1Off_g_ad(n,conditional_indx) = R1Off_g_ad(n,conditional_indx) + R1Off_g_inc; end
  conditional_test=any(any(t<=R1Off_tspike+R1Off_t_ref,1));
  conditional_indx=(any(t<=R1Off_tspike+R1Off_t_ref,1));
  if conditional_test, R1Off_V(n,conditional_indx) = R1Off_V_reset; end
  conditional_test=any(S1OnOff_V(n,:) > S1OnOff_V_thresh);
  conditional_indx=(S1OnOff_V(n,:) > S1OnOff_V_thresh);
  if conditional_test, S1OnOff_V(n,conditional_indx) = S1OnOff_V_reset; S1OnOff_g_ad(n,conditional_indx) = S1OnOff_g_ad(n,conditional_indx) + S1OnOff_g_inc; end
  conditional_test=any(any(t<=S1OnOff_tspike+S1OnOff_t_ref,1));
  conditional_indx=(any(t<=S1OnOff_tspike+S1OnOff_t_ref,1));
  if conditional_test, S1OnOff_V(n,conditional_indx) = S1OnOff_V_reset; end
  conditional_test=any(R2On_V(n,:) > R2On_V_thresh);
  conditional_indx=(R2On_V(n,:) > R2On_V_thresh);
  if conditional_test, R2On_V(n,conditional_indx) = R2On_V_reset; R2On_g_ad(n,conditional_indx) = R2On_g_ad(n,conditional_indx) + R2On_g_inc; end
  conditional_test=any(any(t<=R2On_tspike+R2On_t_ref,1));
  conditional_indx=(any(t<=R2On_tspike+R2On_t_ref,1));
  if conditional_test, R2On_V(n,conditional_indx) = R2On_V_reset; end
  conditional_test=any(R2Off_V(n,:) > R2Off_V_thresh);
  conditional_indx=(R2Off_V(n,:) > R2Off_V_thresh);
  if conditional_test, R2Off_V(n,conditional_indx) = R2Off_V_reset; R2Off_g_ad(n,conditional_indx) = R2Off_g_ad(n,conditional_indx) + R2Off_g_inc; end
  conditional_test=any(any(t<=R2Off_tspike+R2Off_t_ref,1));
  conditional_indx=(any(t<=R2Off_tspike+R2Off_t_ref,1));
  if conditional_test, R2Off_V(n,conditional_indx) = R2Off_V_reset; end
  conditional_test=any(S2OnOff_V(n,:) > S2OnOff_V_thresh);
  conditional_indx=(S2OnOff_V(n,:) > S2OnOff_V_thresh);
  if conditional_test, S2OnOff_V(n,conditional_indx) = S2OnOff_V_reset; S2OnOff_g_ad(n,conditional_indx) = S2OnOff_g_ad(n,conditional_indx) + S2OnOff_g_inc; end
  conditional_test=any(any(t<=S2OnOff_tspike+S2OnOff_t_ref,1));
  conditional_indx=(any(t<=S2OnOff_tspike+S2OnOff_t_ref,1));
  if conditional_test, S2OnOff_V(n,conditional_indx) = S2OnOff_V_reset; end
  conditional_test=any(any(t == On_tspike+R1On_On_PSC_delay,1));
  conditional_indx=(any(t == On_tspike+R1On_On_PSC_delay,1));
  if conditional_test, R1On_On_PSC_x(n,conditional_indx) = R1On_On_PSC_x(n,conditional_indx) + R1On_On_PSC_q(n,conditional_indx);R1On_On_PSC_q(n,conditional_indx) = R1On_On_PSC_F(n,conditional_indx).*R1On_On_PSC_P(n,conditional_indx);R1On_On_PSC_F(n,conditional_indx) = R1On_On_PSC_F(n,conditional_indx) + R1On_On_PSC_fF*(R1On_On_PSC_maxF-R1On_On_PSC_F(n,conditional_indx)); R1On_On_PSC_P(n,conditional_indx) = R1On_On_PSC_P(n,conditional_indx)*(1 - R1On_On_PSC_fP); end
  conditional_test=any(any(t == On_tspike+S1OnOff_On_PSC_delay,1));
  conditional_indx=(any(t == On_tspike+S1OnOff_On_PSC_delay,1));
  if conditional_test, S1OnOff_On_PSC_x(n,conditional_indx) = S1OnOff_On_PSC_x(n,conditional_indx) + S1OnOff_On_PSC_q(n,conditional_indx);S1OnOff_On_PSC_q(n,conditional_indx) = S1OnOff_On_PSC_F(n,conditional_indx).*S1OnOff_On_PSC_P(n,conditional_indx);S1OnOff_On_PSC_F(n,conditional_indx) = S1OnOff_On_PSC_F(n,conditional_indx) + S1OnOff_On_PSC_fF*(S1OnOff_On_PSC_maxF-S1OnOff_On_PSC_F(n,conditional_indx)); S1OnOff_On_PSC_P(n,conditional_indx) = S1OnOff_On_PSC_P(n,conditional_indx)*(1 - S1OnOff_On_PSC_fP); end
  conditional_test=any(any(t == S1OnOff_tspike+R1On_S1OnOff_PSC_delay,1));
  conditional_indx=(any(t == S1OnOff_tspike+R1On_S1OnOff_PSC_delay,1));
  if conditional_test, R1On_S1OnOff_PSC_x(n,conditional_indx) = R1On_S1OnOff_PSC_x(n,conditional_indx) + R1On_S1OnOff_PSC_q(n,conditional_indx);R1On_S1OnOff_PSC_q(n,conditional_indx) = R1On_S1OnOff_PSC_F(n,conditional_indx).*R1On_S1OnOff_PSC_P(n,conditional_indx);R1On_S1OnOff_PSC_F(n,conditional_indx) = R1On_S1OnOff_PSC_F(n,conditional_indx) + R1On_S1OnOff_PSC_fF*(R1On_S1OnOff_PSC_maxF-R1On_S1OnOff_PSC_F(n,conditional_indx)); R1On_S1OnOff_PSC_P(n,conditional_indx) = R1On_S1OnOff_PSC_P(n,conditional_indx)*(1 - R1On_S1OnOff_PSC_fP); end
  conditional_test=any(any(t == S1OnOff_tspike+R1Off_S1OnOff_PSC_delay,1));
  conditional_indx=(any(t == S1OnOff_tspike+R1Off_S1OnOff_PSC_delay,1));
  if conditional_test, R1Off_S1OnOff_PSC_x(n,conditional_indx) = R1Off_S1OnOff_PSC_x(n,conditional_indx) + R1Off_S1OnOff_PSC_q(n,conditional_indx);R1Off_S1OnOff_PSC_q(n,conditional_indx) = R1Off_S1OnOff_PSC_F(n,conditional_indx).*R1Off_S1OnOff_PSC_P(n,conditional_indx);R1Off_S1OnOff_PSC_F(n,conditional_indx) = R1Off_S1OnOff_PSC_F(n,conditional_indx) + R1Off_S1OnOff_PSC_fF*(R1Off_S1OnOff_PSC_maxF-R1Off_S1OnOff_PSC_F(n,conditional_indx)); R1Off_S1OnOff_PSC_P(n,conditional_indx) = R1Off_S1OnOff_PSC_P(n,conditional_indx)*(1 - R1Off_S1OnOff_PSC_fP); end
  conditional_test=any(any(t == Off_tspike+R1Off_Off_PSC_delay,1));
  conditional_indx=(any(t == Off_tspike+R1Off_Off_PSC_delay,1));
  if conditional_test, R1Off_Off_PSC_x(n,conditional_indx) = R1Off_Off_PSC_x(n,conditional_indx) + R1Off_Off_PSC_q(n,conditional_indx);R1Off_Off_PSC_q(n,conditional_indx) = R1Off_Off_PSC_F(n,conditional_indx).*R1Off_Off_PSC_P(n,conditional_indx);R1Off_Off_PSC_F(n,conditional_indx) = R1Off_Off_PSC_F(n,conditional_indx) + R1Off_Off_PSC_fF*(R1Off_Off_PSC_maxF-R1Off_Off_PSC_F(n,conditional_indx)); R1Off_Off_PSC_P(n,conditional_indx) = R1Off_Off_PSC_P(n,conditional_indx)*(1 - R1Off_Off_PSC_fP); end
  conditional_test=any(any(t == Off_tspike+S1OnOff_Off_PSC_delay,1));
  conditional_indx=(any(t == Off_tspike+S1OnOff_Off_PSC_delay,1));
  if conditional_test, S1OnOff_Off_PSC_x(n,conditional_indx) = S1OnOff_Off_PSC_x(n,conditional_indx) + S1OnOff_Off_PSC_q(n,conditional_indx);S1OnOff_Off_PSC_q(n,conditional_indx) = S1OnOff_Off_PSC_F(n,conditional_indx).*S1OnOff_Off_PSC_P(n,conditional_indx);S1OnOff_Off_PSC_F(n,conditional_indx) = S1OnOff_Off_PSC_F(n,conditional_indx) + S1OnOff_Off_PSC_fF*(S1OnOff_Off_PSC_maxF-S1OnOff_Off_PSC_F(n,conditional_indx)); S1OnOff_Off_PSC_P(n,conditional_indx) = S1OnOff_Off_PSC_P(n,conditional_indx)*(1 - S1OnOff_Off_PSC_fP); end
  conditional_test=any(any(t == R1On_tspike+R2On_R1On_PSC_delay,1));
  conditional_indx=(any(t == R1On_tspike+R2On_R1On_PSC_delay,1));
  if conditional_test, R2On_R1On_PSC_x(n,conditional_indx) = R2On_R1On_PSC_x(n,conditional_indx) + R2On_R1On_PSC_q(n,conditional_indx);R2On_R1On_PSC_q(n,conditional_indx) = R2On_R1On_PSC_F(n,conditional_indx).*R2On_R1On_PSC_P(n,conditional_indx);R2On_R1On_PSC_F(n,conditional_indx) = R2On_R1On_PSC_F(n,conditional_indx) + R2On_R1On_PSC_fF*(R2On_R1On_PSC_maxF-R2On_R1On_PSC_F(n,conditional_indx)); R2On_R1On_PSC_P(n,conditional_indx) = R2On_R1On_PSC_P(n,conditional_indx)*(1 - R2On_R1On_PSC_fP); end
  conditional_test=any(any(t == R1On_tspike+S2OnOff_R1On_PSC_delay,1));
  conditional_indx=(any(t == R1On_tspike+S2OnOff_R1On_PSC_delay,1));
  if conditional_test, S2OnOff_R1On_PSC_x(n,conditional_indx) = S2OnOff_R1On_PSC_x(n,conditional_indx) + S2OnOff_R1On_PSC_q(n,conditional_indx);S2OnOff_R1On_PSC_q(n,conditional_indx) = S2OnOff_R1On_PSC_F(n,conditional_indx).*S2OnOff_R1On_PSC_P(n,conditional_indx);S2OnOff_R1On_PSC_F(n,conditional_indx) = S2OnOff_R1On_PSC_F(n,conditional_indx) + S2OnOff_R1On_PSC_fF*(S2OnOff_R1On_PSC_maxF-S2OnOff_R1On_PSC_F(n,conditional_indx)); S2OnOff_R1On_PSC_P(n,conditional_indx) = S2OnOff_R1On_PSC_P(n,conditional_indx)*(1 - S2OnOff_R1On_PSC_fP); end
  conditional_test=any(any(t == S2OnOff_tspike+R2On_S2OnOff_PSC_delay,1));
  conditional_indx=(any(t == S2OnOff_tspike+R2On_S2OnOff_PSC_delay,1));
  if conditional_test, R2On_S2OnOff_PSC_x(n,conditional_indx) = R2On_S2OnOff_PSC_x(n,conditional_indx) + R2On_S2OnOff_PSC_q(n,conditional_indx);R2On_S2OnOff_PSC_q(n,conditional_indx) = R2On_S2OnOff_PSC_F(n,conditional_indx).*R2On_S2OnOff_PSC_P(n,conditional_indx);R2On_S2OnOff_PSC_F(n,conditional_indx) = R2On_S2OnOff_PSC_F(n,conditional_indx) + R2On_S2OnOff_PSC_fF*(R2On_S2OnOff_PSC_maxF-R2On_S2OnOff_PSC_F(n,conditional_indx)); R2On_S2OnOff_PSC_P(n,conditional_indx) = R2On_S2OnOff_PSC_P(n,conditional_indx)*(1 - R2On_S2OnOff_PSC_fP); end
  conditional_test=any(any(t == S2OnOff_tspike+R2Off_S2OnOff_PSC_delay,1));
  conditional_indx=(any(t == S2OnOff_tspike+R2Off_S2OnOff_PSC_delay,1));
  if conditional_test, R2Off_S2OnOff_PSC_x(n,conditional_indx) = R2Off_S2OnOff_PSC_x(n,conditional_indx) + R2Off_S2OnOff_PSC_q(n,conditional_indx);R2Off_S2OnOff_PSC_q(n,conditional_indx) = R2Off_S2OnOff_PSC_F(n,conditional_indx).*R2Off_S2OnOff_PSC_P(n,conditional_indx);R2Off_S2OnOff_PSC_F(n,conditional_indx) = R2Off_S2OnOff_PSC_F(n,conditional_indx) + R2Off_S2OnOff_PSC_fF*(R2Off_S2OnOff_PSC_maxF-R2Off_S2OnOff_PSC_F(n,conditional_indx)); R2Off_S2OnOff_PSC_P(n,conditional_indx) = R2Off_S2OnOff_PSC_P(n,conditional_indx)*(1 - R2Off_S2OnOff_PSC_fP); end
  conditional_test=any(any(t == R1Off_tspike+R2Off_R1Off_PSC_delay,1));
  conditional_indx=(any(t == R1Off_tspike+R2Off_R1Off_PSC_delay,1));
  if conditional_test, R2Off_R1Off_PSC_x(n,conditional_indx) = R2Off_R1Off_PSC_x(n,conditional_indx) + R2Off_R1Off_PSC_q(n,conditional_indx);R2Off_R1Off_PSC_q(n,conditional_indx) = R2Off_R1Off_PSC_F(n,conditional_indx).*R2Off_R1Off_PSC_P(n,conditional_indx);R2Off_R1Off_PSC_F(n,conditional_indx) = R2Off_R1Off_PSC_F(n,conditional_indx) + R2Off_R1Off_PSC_fF*(R2Off_R1Off_PSC_maxF-R2Off_R1Off_PSC_F(n,conditional_indx)); R2Off_R1Off_PSC_P(n,conditional_indx) = R2Off_R1Off_PSC_P(n,conditional_indx)*(1 - R2Off_R1Off_PSC_fP); end
  conditional_test=any(any(t == R1Off_tspike+S2OnOff_R1Off_PSC_delay,1));
  conditional_indx=(any(t == R1Off_tspike+S2OnOff_R1Off_PSC_delay,1));
  if conditional_test, S2OnOff_R1Off_PSC_x(n,conditional_indx) = S2OnOff_R1Off_PSC_x(n,conditional_indx) + S2OnOff_R1Off_PSC_q(n,conditional_indx);S2OnOff_R1Off_PSC_q(n,conditional_indx) = S2OnOff_R1Off_PSC_F(n,conditional_indx).*S2OnOff_R1Off_PSC_P(n,conditional_indx);S2OnOff_R1Off_PSC_F(n,conditional_indx) = S2OnOff_R1Off_PSC_F(n,conditional_indx) + S2OnOff_R1Off_PSC_fF*(S2OnOff_R1Off_PSC_maxF-S2OnOff_R1Off_PSC_F(n,conditional_indx)); S2OnOff_R1Off_PSC_P(n,conditional_indx) = S2OnOff_R1Off_PSC_P(n,conditional_indx)*(1 - S2OnOff_R1Off_PSC_fP); end

  % ------------------------------------------------------------
  % Update monitors:
  % ------------------------------------------------------------
  On_On_IC_iIC(n,:)=On_On_IC_g_postIC*(On_On_IC_input(k,:)*On_On_IC_netcon).*(On_V(n,:)-On_On_IC_E_exc);
  Off_Off_IC_iIC(n,:)=Off_Off_IC_g_postIC*(Off_Off_IC_input(k,:)*Off_Off_IC_netcon).*(Off_V(n,:)-Off_Off_IC_E_exc);
  R1On_On_PSC_syn(n,:)=R1On_On_PSC_gSYN.*(R1On_On_PSC_s(n,:)*R1On_On_PSC_netcon).*(R1On_V(n,:)-R1On_On_PSC_ESYN);
  S1OnOff_On_PSC_syn(n,:)=S1OnOff_On_PSC_gSYN.*(S1OnOff_On_PSC_s(n,:)*S1OnOff_On_PSC_netcon).*(S1OnOff_V(n,:)-S1OnOff_On_PSC_ESYN);
  R1On_S1OnOff_PSC_syn(n,:)=R1On_S1OnOff_PSC_gSYN.*(R1On_S1OnOff_PSC_s(n,:)*R1On_S1OnOff_PSC_netcon).*(R1On_V(n,:)-R1On_S1OnOff_PSC_ESYN);
  R1Off_S1OnOff_PSC_syn(n,:)=R1Off_S1OnOff_PSC_gSYN.*(R1Off_S1OnOff_PSC_s(n,:)*R1Off_S1OnOff_PSC_netcon).*(R1Off_V(n,:)-R1Off_S1OnOff_PSC_ESYN);
  R1Off_Off_PSC_syn(n,:)=R1Off_Off_PSC_gSYN.*(R1Off_Off_PSC_s(n,:)*R1Off_Off_PSC_netcon).*(R1Off_V(n,:)-R1Off_Off_PSC_ESYN);
  S1OnOff_Off_PSC_syn(n,:)=S1OnOff_Off_PSC_gSYN.*(S1OnOff_Off_PSC_s(n,:)*S1OnOff_Off_PSC_netcon).*(S1OnOff_V(n,:)-S1OnOff_Off_PSC_ESYN);
  R2On_R1On_PSC_syn(n,:)=R2On_R1On_PSC_gSYN.*(R2On_R1On_PSC_s(n,:)*R2On_R1On_PSC_netcon).*(R2On_V(n,:)-R2On_R1On_PSC_ESYN);
  S2OnOff_R1On_PSC_syn(n,:)=S2OnOff_R1On_PSC_gSYN.*(S2OnOff_R1On_PSC_s(n,:)*S2OnOff_R1On_PSC_netcon).*(S2OnOff_V(n,:)-S2OnOff_R1On_PSC_ESYN);
  R2On_S2OnOff_PSC_syn(n,:)=R2On_S2OnOff_PSC_gSYN.*(R2On_S2OnOff_PSC_s(n,:)*R2On_S2OnOff_PSC_netcon).*(R2On_V(n,:)-R2On_S2OnOff_PSC_ESYN);
  R2Off_S2OnOff_PSC_syn(n,:)=R2Off_S2OnOff_PSC_gSYN.*(R2Off_S2OnOff_PSC_s(n,:)*R2Off_S2OnOff_PSC_netcon).*(R2Off_V(n,:)-R2Off_S2OnOff_PSC_ESYN);
  R2Off_R1Off_PSC_syn(n,:)=R2Off_R1Off_PSC_gSYN.*(R2Off_R1Off_PSC_s(n,:)*R2Off_R1Off_PSC_netcon).*(R2Off_V(n,:)-R2Off_R1Off_PSC_ESYN);
  S2OnOff_R1Off_PSC_syn(n,:)=S2OnOff_R1Off_PSC_gSYN.*(S2OnOff_R1Off_PSC_s(n,:)*S2OnOff_R1Off_PSC_netcon).*(S2OnOff_V(n,:)-S2OnOff_R1Off_PSC_ESYN);
  n=n+1;
end

T=T(1:downsample_factor:ntime);

end
