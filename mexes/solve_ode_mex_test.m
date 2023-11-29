function [T,IC_V,IC_g_ad,S1_V,S1_g_ad,R1_V,R1_g_ad,S2_V,S2_g_ad,R2_V,R2_g_ad,X1_V,X1_g_ad,X2_V,X2_g_ad,C_V,C_g_ad,R1_IC_PSC_s,R1_IC_PSC_x,R1_IC_PSC_F,R1_IC_PSC_P,S1_IC_PSC_s,S1_IC_PSC_x,S1_IC_PSC_F,S1_IC_PSC_P,X1_R1_PSC_s,X1_R1_PSC_x,X1_R1_PSC_F,X1_R1_PSC_P,R1_X1_PSC_s,R1_X1_PSC_x,R1_X1_PSC_F,R1_X1_PSC_P,R1_S1_PSC_s,R1_S1_PSC_x,R1_S1_PSC_F,R1_S1_PSC_P,S1_R1_PSC_s,S1_R1_PSC_x,S1_R1_PSC_F,S1_R1_PSC_P,R2_R1_PSC_s,R2_R1_PSC_x,R2_R1_PSC_F,R2_R1_PSC_P,S2_R1_PSC_s,S2_R1_PSC_x,S2_R1_PSC_F,S2_R1_PSC_P,R2_S2_PSC_s,R2_S2_PSC_x,R2_S2_PSC_F,R2_S2_PSC_P,S2_R2_PSC_s,S2_R2_PSC_x,S2_R2_PSC_F,S2_R2_PSC_P,X2_R2_PSC_s,X2_R2_PSC_x,X2_R2_PSC_F,X2_R2_PSC_P,R2_X2_PSC_s,R2_X2_PSC_x,R2_X2_PSC_F,R2_X2_PSC_P,R2_R2_iNoise_V3_sn,R2_R2_iNoise_V3_xn,C_R2_PSC_s,C_R2_PSC_x,C_R2_PSC_F,C_R2_PSC_P,C_S2_PSC_s,C_S2_PSC_x,C_S2_PSC_F,C_S2_PSC_P,IC_V_spikes,S1_V_spikes,R1_V_spikes,S2_V_spikes,R2_V_spikes,X1_V_spikes,X2_V_spikes,C_V_spikes,IC_IC_IC_iIC,IC_R,IC_tau,IC_Imask,S1_R,S1_tau,S1_Imask,R1_R,R1_tau,R1_Imask,S2_R,S2_tau,S2_Imask,R2_R,R2_tau,R2_Imask,X1_R,X1_tau,X1_Imask,X2_R,X2_tau,X2_Imask,C_R,C_tau,C_Imask,IC_IC_IC_netcon,IC_IC_IC_input,R1_IC_PSC_netcon,R1_IC_PSC_scale,S1_IC_PSC_netcon,S1_IC_PSC_scale,X1_R1_PSC_netcon,X1_R1_PSC_scale,R1_X1_PSC_netcon,R1_X1_PSC_scale,R1_S1_PSC_netcon,R1_S1_PSC_scale,S1_R1_PSC_netcon,S1_R1_PSC_scale,R2_R1_PSC_netcon,R2_R1_PSC_scale,S2_R1_PSC_netcon,S2_R1_PSC_scale,R2_S2_PSC_netcon,R2_S2_PSC_scale,S2_R2_PSC_netcon,S2_R2_PSC_scale,X2_R2_PSC_netcon,X2_R2_PSC_scale,R2_X2_PSC_netcon,R2_X2_PSC_scale,R2_R2_iNoise_V3_netcon,R2_R2_iNoise_V3_token,R2_R2_iNoise_V3_scale,C_R2_PSC_netcon,C_R2_PSC_scale,C_S2_PSC_netcon,C_S2_PSC_scale]=solve_ode

% ------------------------------------------------------------
% Parameters:
% ------------------------------------------------------------
params = load('params.mat','p');
p = params.p;
downsample_factor=p.downsample_factor;
dt=p.dt;
T=(p.tspan(1):dt:p.tspan(2))';
ntime=length(T);
nsamp=length(1:downsample_factor:ntime);

% seed the random number generator
rng(p.random_seed);

% ------------------------------------------------------------
% Fixed variables:
% ------------------------------------------------------------
IC_R = 1/p.IC_g_L;
IC_tau = p.IC_C*IC_R;
IC_Imask = ones(1,p.IC_Npop);
S1_R = 1/p.S1_g_L;
S1_tau = p.S1_C*S1_R;
S1_Imask = ones(1,p.S1_Npop);
R1_R = 1/p.R1_g_L;
R1_tau = p.R1_C*R1_R;
R1_Imask = ones(1,p.R1_Npop);
S2_R = 1/p.S2_g_L;
S2_tau = p.S2_C*S2_R;
S2_Imask = ones(1,p.S2_Npop);
R2_R = 1/p.R2_g_L;
R2_tau = p.R2_C*R2_R;
R2_Imask = ones(1,p.R2_Npop);
X1_R = 1/p.X1_g_L;
X1_tau = p.X1_C*X1_R;
X1_Imask = ones(1,p.X1_Npop);
X2_R = 1/p.X2_g_L;
X2_tau = p.X2_C*X2_R;
X2_Imask = ones(1,p.X2_Npop);
C_R = 1/p.C_g_L;
C_tau = p.C_C*C_R;
C_Imask = ones(1,p.C_Npop);
IC_IC_IC_netcon = [+1.000000000000000e+00   +0.000000000000000e+00; +0.000000000000000e+00   +1.000000000000000e+00];
IC_IC_IC_input = genICSpks(p.IC_IC_IC_trial,p.IC_IC_IC_locNum);
R1_IC_PSC_netcon = eye(p.IC_Npop,p.R1_Npop);
R1_IC_PSC_scale = (p.R1_IC_PSC_tauD/p.R1_IC_PSC_tauR)^(p.R1_IC_PSC_tauR/(p.R1_IC_PSC_tauD-p.R1_IC_PSC_tauR));
S1_IC_PSC_netcon = eye(p.IC_Npop,p.S1_Npop);
S1_IC_PSC_scale = (p.S1_IC_PSC_tauD/p.S1_IC_PSC_tauR)^(p.S1_IC_PSC_tauR/(p.S1_IC_PSC_tauD-p.S1_IC_PSC_tauR));
X1_R1_PSC_netcon = eye(p.R1_Npop,p.X1_Npop);
X1_R1_PSC_scale = (p.X1_R1_PSC_tauD/p.X1_R1_PSC_tauR)^(p.X1_R1_PSC_tauR/(p.X1_R1_PSC_tauD-p.X1_R1_PSC_tauR));
R1_X1_PSC_netcon = [+0.000000000000000e+00   +0.000000000000000e+00; +1.000000000000000e+00   +0.000000000000000e+00];
R1_X1_PSC_scale = (p.R1_X1_PSC_tauD/p.R1_X1_PSC_tauR)^(p.R1_X1_PSC_tauR/(p.R1_X1_PSC_tauD-p.R1_X1_PSC_tauR));
R1_S1_PSC_netcon = eye(p.S1_Npop,p.R1_Npop);
R1_S1_PSC_scale = (p.R1_S1_PSC_tauD/p.R1_S1_PSC_tauR)^(p.R1_S1_PSC_tauR/(p.R1_S1_PSC_tauD-p.R1_S1_PSC_tauR));
S1_R1_PSC_netcon = eye(p.R1_Npop,p.S1_Npop);
S1_R1_PSC_scale = (p.S1_R1_PSC_tauD/p.S1_R1_PSC_tauR)^(p.S1_R1_PSC_tauR/(p.S1_R1_PSC_tauD-p.S1_R1_PSC_tauR));
R2_R1_PSC_netcon = eye(p.R1_Npop,p.R2_Npop);
R2_R1_PSC_scale = (p.R2_R1_PSC_tauD/p.R2_R1_PSC_tauR)^(p.R2_R1_PSC_tauR/(p.R2_R1_PSC_tauD-p.R2_R1_PSC_tauR));
S2_R1_PSC_netcon = eye(p.R1_Npop,p.S2_Npop);
S2_R1_PSC_scale = (p.S2_R1_PSC_tauD/p.S2_R1_PSC_tauR)^(p.S2_R1_PSC_tauR/(p.S2_R1_PSC_tauD-p.S2_R1_PSC_tauR));
R2_S2_PSC_netcon = eye(p.S2_Npop,p.R2_Npop);
R2_S2_PSC_scale = (p.R2_S2_PSC_tauD/p.R2_S2_PSC_tauR)^(p.R2_S2_PSC_tauR/(p.R2_S2_PSC_tauD-p.R2_S2_PSC_tauR));
S2_R2_PSC_netcon = eye(p.R2_Npop,p.S2_Npop);
S2_R2_PSC_scale = (p.S2_R2_PSC_tauD/p.S2_R2_PSC_tauR)^(p.S2_R2_PSC_tauR/(p.S2_R2_PSC_tauD-p.S2_R2_PSC_tauR));
X2_R2_PSC_netcon = eye(p.R2_Npop,p.X2_Npop);
X2_R2_PSC_scale = (p.X2_R2_PSC_tauD/p.X2_R2_PSC_tauR)^(p.X2_R2_PSC_tauR/(p.X2_R2_PSC_tauD-p.X2_R2_PSC_tauR));
R2_X2_PSC_netcon = [+0.000000000000000e+00   +0.000000000000000e+00; +1.000000000000000e+00   +0.000000000000000e+00];
R2_X2_PSC_scale = (p.R2_X2_PSC_tauD/p.R2_X2_PSC_tauR)^(p.R2_X2_PSC_tauR/(p.R2_X2_PSC_tauD-p.R2_X2_PSC_tauR));
R2_R2_iNoise_V3_netcon =  eye(p.R2_Npop,p.R2_Npop);
R2_R2_iNoise_V3_token = genPoissonTimes(p.R2_Npop,p.R2_R2_iNoise_V3_dt,p.R2_R2_iNoise_V3_FR,p.R2_R2_iNoise_V3_sigma,p.R2_R2_iNoise_V3_locNum);
R2_R2_iNoise_V3_scale =  (p.R2_R2_iNoise_V3_tauD_N/p.R2_R2_iNoise_V3_tauR_N)^(p.R2_R2_iNoise_V3_tauR_N/(p.R2_R2_iNoise_V3_tauD_N-p.R2_R2_iNoise_V3_tauR_N));
C_R2_PSC_netcon = [+1.000000000000000e+00; +1.000000000000000e+00];
C_R2_PSC_scale = (p.C_R2_PSC_tauD/p.C_R2_PSC_tauR)^(p.C_R2_PSC_tauR/(p.C_R2_PSC_tauD-p.C_R2_PSC_tauR));
C_S2_PSC_netcon = [+1.000000000000000e+00; +1.000000000000000e+00];
C_S2_PSC_scale = (p.C_S2_PSC_tauD/p.C_S2_PSC_tauR)^(p.C_S2_PSC_tauR/(p.C_S2_PSC_tauD-p.C_S2_PSC_tauR));

% ------------------------------------------------------------
% Initial conditions:
% ------------------------------------------------------------
t=0; k=1;

% STATE_VARIABLES:
IC_V = zeros(nsamp,p.IC_Npop);
IC_V(1,:) =  p.IC_E_L*ones(1,p.IC_Npop);
IC_g_ad = zeros(nsamp,p.IC_Npop);
IC_g_ad(1,:) =  zeros(1,p.IC_Npop);
S1_V = zeros(nsamp,p.S1_Npop);
S1_V(1,:) =  p.S1_E_L*ones(1,p.S1_Npop);
S1_g_ad = zeros(nsamp,p.S1_Npop);
S1_g_ad(1,:) =  zeros(1,p.S1_Npop);
R1_V = zeros(nsamp,p.R1_Npop);
R1_V(1,:) =  p.R1_E_L*ones(1,p.R1_Npop);
R1_g_ad = zeros(nsamp,p.R1_Npop);
R1_g_ad(1,:) =  zeros(1,p.R1_Npop);
S2_V = zeros(nsamp,p.S2_Npop);
S2_V(1,:) =  p.S2_E_L*ones(1,p.S2_Npop);
S2_g_ad = zeros(nsamp,p.S2_Npop);
S2_g_ad(1,:) =  zeros(1,p.S2_Npop);
R2_V = zeros(nsamp,p.R2_Npop);
R2_V(1,:) =  p.R2_E_L*ones(1,p.R2_Npop);
R2_g_ad = zeros(nsamp,p.R2_Npop);
R2_g_ad(1,:) =  zeros(1,p.R2_Npop);
X1_V = zeros(nsamp,p.X1_Npop);
X1_V(1,:) =  p.X1_E_L*ones(1,p.X1_Npop);
X1_g_ad = zeros(nsamp,p.X1_Npop);
X1_g_ad(1,:) =  zeros(1,p.X1_Npop);
X2_V = zeros(nsamp,p.X2_Npop);
X2_V(1,:) =  p.X2_E_L*ones(1,p.X2_Npop);
X2_g_ad = zeros(nsamp,p.X2_Npop);
X2_g_ad(1,:) =  zeros(1,p.X2_Npop);
C_V = zeros(nsamp,p.C_Npop);
C_V(1,:) =  p.C_E_L*ones(1,p.C_Npop);
C_g_ad = zeros(nsamp,p.C_Npop);
C_g_ad(1,:) =  zeros(1,p.C_Npop);
R1_IC_PSC_s = zeros(nsamp,p.IC_Npop);
R1_IC_PSC_s(1,:) =  zeros(1,p.IC_Npop);
R1_IC_PSC_x = zeros(nsamp,p.IC_Npop);
R1_IC_PSC_x(1,:) =  zeros(1,p.IC_Npop);
R1_IC_PSC_F = zeros(nsamp,p.IC_Npop);
R1_IC_PSC_F(1,:) =  ones(1,p.IC_Npop);
R1_IC_PSC_P = zeros(nsamp,p.IC_Npop);
R1_IC_PSC_P(1,:) =  ones(1,p.IC_Npop);
S1_IC_PSC_s = zeros(nsamp,p.IC_Npop);
S1_IC_PSC_s(1,:) =  zeros(1,p.IC_Npop);
S1_IC_PSC_x = zeros(nsamp,p.IC_Npop);
S1_IC_PSC_x(1,:) =  zeros(1,p.IC_Npop);
S1_IC_PSC_F = zeros(nsamp,p.IC_Npop);
S1_IC_PSC_F(1,:) =  ones(1,p.IC_Npop);
S1_IC_PSC_P = zeros(nsamp,p.IC_Npop);
S1_IC_PSC_P(1,:) =  ones(1,p.IC_Npop);
X1_R1_PSC_s = zeros(nsamp,p.R1_Npop);
X1_R1_PSC_s(1,:) =  zeros(1,p.R1_Npop);
X1_R1_PSC_x = zeros(nsamp,p.R1_Npop);
X1_R1_PSC_x(1,:) =  zeros(1,p.R1_Npop);
X1_R1_PSC_F = zeros(nsamp,p.R1_Npop);
X1_R1_PSC_F(1,:) =  ones(1,p.R1_Npop);
X1_R1_PSC_P = zeros(nsamp,p.R1_Npop);
X1_R1_PSC_P(1,:) =  ones(1,p.R1_Npop);
R1_X1_PSC_s = zeros(nsamp,p.X1_Npop);
R1_X1_PSC_s(1,:) =  zeros(1,p.X1_Npop);
R1_X1_PSC_x = zeros(nsamp,p.X1_Npop);
R1_X1_PSC_x(1,:) =  zeros(1,p.X1_Npop);
R1_X1_PSC_F = zeros(nsamp,p.X1_Npop);
R1_X1_PSC_F(1,:) =  ones(1,p.X1_Npop);
R1_X1_PSC_P = zeros(nsamp,p.X1_Npop);
R1_X1_PSC_P(1,:) =  ones(1,p.X1_Npop);
R1_S1_PSC_s = zeros(nsamp,p.S1_Npop);
R1_S1_PSC_s(1,:) =  zeros(1,p.S1_Npop);
R1_S1_PSC_x = zeros(nsamp,p.S1_Npop);
R1_S1_PSC_x(1,:) =  zeros(1,p.S1_Npop);
R1_S1_PSC_F = zeros(nsamp,p.S1_Npop);
R1_S1_PSC_F(1,:) =  ones(1,p.S1_Npop);
R1_S1_PSC_P = zeros(nsamp,p.S1_Npop);
R1_S1_PSC_P(1,:) =  ones(1,p.S1_Npop);
S1_R1_PSC_s = zeros(nsamp,p.R1_Npop);
S1_R1_PSC_s(1,:) =  zeros(1,p.R1_Npop);
S1_R1_PSC_x = zeros(nsamp,p.R1_Npop);
S1_R1_PSC_x(1,:) =  zeros(1,p.R1_Npop);
S1_R1_PSC_F = zeros(nsamp,p.R1_Npop);
S1_R1_PSC_F(1,:) =  ones(1,p.R1_Npop);
S1_R1_PSC_P = zeros(nsamp,p.R1_Npop);
S1_R1_PSC_P(1,:) =  ones(1,p.R1_Npop);
R2_R1_PSC_s = zeros(nsamp,p.R1_Npop);
R2_R1_PSC_s(1,:) =  zeros(1,p.R1_Npop);
R2_R1_PSC_x = zeros(nsamp,p.R1_Npop);
R2_R1_PSC_x(1,:) =  zeros(1,p.R1_Npop);
R2_R1_PSC_F = zeros(nsamp,p.R1_Npop);
R2_R1_PSC_F(1,:) =  ones(1,p.R1_Npop);
R2_R1_PSC_P = zeros(nsamp,p.R1_Npop);
R2_R1_PSC_P(1,:) =  ones(1,p.R1_Npop);
S2_R1_PSC_s = zeros(nsamp,p.R1_Npop);
S2_R1_PSC_s(1,:) =  zeros(1,p.R1_Npop);
S2_R1_PSC_x = zeros(nsamp,p.R1_Npop);
S2_R1_PSC_x(1,:) =  zeros(1,p.R1_Npop);
S2_R1_PSC_F = zeros(nsamp,p.R1_Npop);
S2_R1_PSC_F(1,:) =  ones(1,p.R1_Npop);
S2_R1_PSC_P = zeros(nsamp,p.R1_Npop);
S2_R1_PSC_P(1,:) =  ones(1,p.R1_Npop);
R2_S2_PSC_s = zeros(nsamp,p.S2_Npop);
R2_S2_PSC_s(1,:) =  zeros(1,p.S2_Npop);
R2_S2_PSC_x = zeros(nsamp,p.S2_Npop);
R2_S2_PSC_x(1,:) =  zeros(1,p.S2_Npop);
R2_S2_PSC_F = zeros(nsamp,p.S2_Npop);
R2_S2_PSC_F(1,:) =  ones(1,p.S2_Npop);
R2_S2_PSC_P = zeros(nsamp,p.S2_Npop);
R2_S2_PSC_P(1,:) =  ones(1,p.S2_Npop);
S2_R2_PSC_s = zeros(nsamp,p.R2_Npop);
S2_R2_PSC_s(1,:) =  zeros(1,p.R2_Npop);
S2_R2_PSC_x = zeros(nsamp,p.R2_Npop);
S2_R2_PSC_x(1,:) =  zeros(1,p.R2_Npop);
S2_R2_PSC_F = zeros(nsamp,p.R2_Npop);
S2_R2_PSC_F(1,:) =  ones(1,p.R2_Npop);
S2_R2_PSC_P = zeros(nsamp,p.R2_Npop);
S2_R2_PSC_P(1,:) =  ones(1,p.R2_Npop);
X2_R2_PSC_s = zeros(nsamp,p.R2_Npop);
X2_R2_PSC_s(1,:) =  zeros(1,p.R2_Npop);
X2_R2_PSC_x = zeros(nsamp,p.R2_Npop);
X2_R2_PSC_x(1,:) =  zeros(1,p.R2_Npop);
X2_R2_PSC_F = zeros(nsamp,p.R2_Npop);
X2_R2_PSC_F(1,:) =  ones(1,p.R2_Npop);
X2_R2_PSC_P = zeros(nsamp,p.R2_Npop);
X2_R2_PSC_P(1,:) =  ones(1,p.R2_Npop);
R2_X2_PSC_s = zeros(nsamp,p.X2_Npop);
R2_X2_PSC_s(1,:) =  zeros(1,p.X2_Npop);
R2_X2_PSC_x = zeros(nsamp,p.X2_Npop);
R2_X2_PSC_x(1,:) =  zeros(1,p.X2_Npop);
R2_X2_PSC_F = zeros(nsamp,p.X2_Npop);
R2_X2_PSC_F(1,:) =  ones(1,p.X2_Npop);
R2_X2_PSC_P = zeros(nsamp,p.X2_Npop);
R2_X2_PSC_P(1,:) =  ones(1,p.X2_Npop);
R2_R2_iNoise_V3_sn = zeros(nsamp,p.R2_Npop);
R2_R2_iNoise_V3_sn(1,:) =  0 * ones(1,p.R2_Npop);
R2_R2_iNoise_V3_xn = zeros(nsamp,p.R2_Npop);
R2_R2_iNoise_V3_xn(1,:) =  0 * ones(1,p.R2_Npop);
C_R2_PSC_s = zeros(nsamp,p.R2_Npop);
C_R2_PSC_s(1,:) =  zeros(1,p.R2_Npop);
C_R2_PSC_x = zeros(nsamp,p.R2_Npop);
C_R2_PSC_x(1,:) =  zeros(1,p.R2_Npop);
C_R2_PSC_F = zeros(nsamp,p.R2_Npop);
C_R2_PSC_F(1,:) =  ones(1,p.R2_Npop);
C_R2_PSC_P = zeros(nsamp,p.R2_Npop);
C_R2_PSC_P(1,:) =  ones(1,p.R2_Npop);
C_S2_PSC_s = zeros(nsamp,p.S2_Npop);
C_S2_PSC_s(1,:) =  zeros(1,p.S2_Npop);
C_S2_PSC_x = zeros(nsamp,p.S2_Npop);
C_S2_PSC_x(1,:) =  zeros(1,p.S2_Npop);
C_S2_PSC_F = zeros(nsamp,p.S2_Npop);
C_S2_PSC_F(1,:) =  ones(1,p.S2_Npop);
C_S2_PSC_P = zeros(nsamp,p.S2_Npop);
C_S2_PSC_P(1,:) =  ones(1,p.S2_Npop);

% MONITORS:
IC_tspike = -1e32*ones(5,p.IC_Npop);
IC_buffer_index = ones(1,p.IC_Npop);
IC_V_spikes = zeros(nsamp,p.IC_Npop);
S1_tspike = -1e32*ones(5,p.S1_Npop);
S1_buffer_index = ones(1,p.S1_Npop);
S1_V_spikes = zeros(nsamp,p.S1_Npop);
R1_tspike = -1e32*ones(5,p.R1_Npop);
R1_buffer_index = ones(1,p.R1_Npop);
R1_V_spikes = zeros(nsamp,p.R1_Npop);
S2_tspike = -1e32*ones(5,p.S2_Npop);
S2_buffer_index = ones(1,p.S2_Npop);
S2_V_spikes = zeros(nsamp,p.S2_Npop);
R2_tspike = -1e32*ones(5,p.R2_Npop);
R2_buffer_index = ones(1,p.R2_Npop);
R2_V_spikes = zeros(nsamp,p.R2_Npop);
X1_tspike = -1e32*ones(5,p.X1_Npop);
X1_buffer_index = ones(1,p.X1_Npop);
X1_V_spikes = zeros(nsamp,p.X1_Npop);
X2_tspike = -1e32*ones(5,p.X2_Npop);
X2_buffer_index = ones(1,p.X2_Npop);
X2_V_spikes = zeros(nsamp,p.X2_Npop);
C_tspike = -1e32*ones(5,p.C_Npop);
C_buffer_index = ones(1,p.C_Npop);
C_V_spikes = zeros(nsamp,p.C_Npop);
IC_IC_IC_iIC = zeros(nsamp,p.IC_Npop);
IC_IC_IC_iIC(1,:)=p.IC_IC_IC_g_postIC*(IC_IC_IC_input(k,:)*IC_IC_IC_netcon).*(IC_V(1,:)-p.IC_IC_IC_E_exc);

% ###########################################################
% Numerical integration:
% ###########################################################
n=2;
for k=2:ntime
  t=T(k-1);
  IC_V_k1 = ( (p.IC_E_L-IC_V(n-1,:)) - IC_R*IC_g_ad(n-1,:).*(IC_V(n-1,:)-p.IC_E_k) - IC_R*((((p.IC_IC_IC_g_postIC*(IC_IC_IC_input(k,:)*IC_IC_IC_netcon).*(IC_V(n-1,:)-p.IC_IC_IC_E_exc))))) + IC_R*p.IC_Itonic.*IC_Imask + IC_R*p.IC_noise.*randn(1,p.IC_Npop) ) / IC_tau;
  IC_g_ad_k1 = -IC_g_ad(n-1,:) / p.IC_tau_ad;
  S1_V_k1 = ( (p.S1_E_L-S1_V(n-1,:)) - S1_R*S1_g_ad(n-1,:).*(S1_V(n-1,:)-p.S1_E_k) - S1_R*((((p.S1_IC_PSC_gSYN.*((S1_IC_PSC_s(n-1,:).*S1_IC_PSC_F(n-1,:).*S1_IC_PSC_P(n-1,:))*S1_IC_PSC_netcon).*(S1_V(n-1,:)-p.S1_IC_PSC_ESYN))))+((((p.S1_R1_PSC_gSYN.*((S1_R1_PSC_s(n-1,:).*S1_R1_PSC_F(n-1,:).*S1_R1_PSC_P(n-1,:))*S1_R1_PSC_netcon).*(S1_V(n-1,:)-p.S1_R1_PSC_ESYN)))))) + S1_R*p.S1_Itonic.*S1_Imask + S1_R*p.S1_noise.*randn(1,p.S1_Npop) ) / S1_tau;
  S1_g_ad_k1 = -S1_g_ad(n-1,:) / p.S1_tau_ad;
  R1_V_k1 = ( (p.R1_E_L-R1_V(n-1,:)) - R1_R*R1_g_ad(n-1,:).*(R1_V(n-1,:)-p.R1_E_k) - R1_R*((((p.R1_IC_PSC_gSYN.*((R1_IC_PSC_s(n-1,:).*R1_IC_PSC_F(n-1,:).*R1_IC_PSC_P(n-1,:))*R1_IC_PSC_netcon).*(R1_V(n-1,:)-p.R1_IC_PSC_ESYN))))+((((p.R1_X1_PSC_gSYN.*((R1_X1_PSC_s(n-1,:).*R1_X1_PSC_F(n-1,:).*R1_X1_PSC_P(n-1,:))*R1_X1_PSC_netcon).*(R1_V(n-1,:)-p.R1_X1_PSC_ESYN))))+((((p.R1_S1_PSC_gSYN.*((R1_S1_PSC_s(n-1,:).*R1_S1_PSC_F(n-1,:).*R1_S1_PSC_P(n-1,:))*R1_S1_PSC_netcon).*(R1_V(n-1,:)-p.R1_S1_PSC_ESYN))))))) + R1_R*p.R1_Itonic.*R1_Imask + R1_R*p.R1_noise.*randn(1,p.R1_Npop) ) / R1_tau;
  R1_g_ad_k1 = -R1_g_ad(n-1,:) / p.R1_tau_ad;
  S2_V_k1 = ( (p.S2_E_L-S2_V(n-1,:)) - S2_R*S2_g_ad(n-1,:).*(S2_V(n-1,:)-p.S2_E_k) - S2_R*((((p.S2_R1_PSC_gSYN.*((S2_R1_PSC_s(n-1,:).*S2_R1_PSC_F(n-1,:).*S2_R1_PSC_P(n-1,:))*S2_R1_PSC_netcon).*(S2_V(n-1,:)-p.S2_R1_PSC_ESYN))))+((((p.S2_R2_PSC_gSYN.*((S2_R2_PSC_s(n-1,:).*S2_R2_PSC_F(n-1,:).*S2_R2_PSC_P(n-1,:))*S2_R2_PSC_netcon).*(S2_V(n-1,:)-p.S2_R2_PSC_ESYN)))))) + S2_R*p.S2_Itonic.*S2_Imask + S2_R*p.S2_noise.*randn(1,p.S2_Npop) ) / S2_tau;
  S2_g_ad_k1 = -S2_g_ad(n-1,:) / p.S2_tau_ad;
  R2_V_k1 = ( (p.R2_E_L-R2_V(n-1,:)) - R2_R*R2_g_ad(n-1,:).*(R2_V(n-1,:)-p.R2_E_k) - R2_R*((((p.R2_R1_PSC_gSYN.*((R2_R1_PSC_s(n-1,:).*R2_R1_PSC_F(n-1,:).*R2_R1_PSC_P(n-1,:))*R2_R1_PSC_netcon).*(R2_V(n-1,:)-p.R2_R1_PSC_ESYN))))+((((p.R2_S2_PSC_gSYN.*((R2_S2_PSC_s(n-1,:).*R2_S2_PSC_F(n-1,:).*R2_S2_PSC_P(n-1,:))*R2_S2_PSC_netcon).*(R2_V(n-1,:)-p.R2_S2_PSC_ESYN))))+((((p.R2_X2_PSC_gSYN.*((R2_X2_PSC_s(n-1,:).*R2_X2_PSC_F(n-1,:).*R2_X2_PSC_P(n-1,:))*R2_X2_PSC_netcon).*(R2_V(n-1,:)-p.R2_X2_PSC_ESYN))))+((((p.R2_R2_iNoise_V3_nSYN.*(R2_R2_iNoise_V3_sn(n-1,:)*R2_R2_iNoise_V3_netcon).*(R2_V(n-1,:)-p.R2_R2_iNoise_V3_E_exc)))))))) + R2_R*p.R2_Itonic.*R2_Imask + R2_R*p.R2_noise.*randn(1,p.R2_Npop) ) / R2_tau;
  R2_g_ad_k1 = -R2_g_ad(n-1,:) / p.R2_tau_ad;
  X1_V_k1 = ( (p.X1_E_L-X1_V(n-1,:)) - X1_R*X1_g_ad(n-1,:).*(X1_V(n-1,:)-p.X1_E_k) - X1_R*((((p.X1_R1_PSC_gSYN.*((X1_R1_PSC_s(n-1,:).*X1_R1_PSC_F(n-1,:).*X1_R1_PSC_P(n-1,:))*X1_R1_PSC_netcon).*(X1_V(n-1,:)-p.X1_R1_PSC_ESYN))))) + X1_R*p.X1_Itonic.*X1_Imask + X1_R*p.X1_noise.*randn(1,p.X1_Npop) ) / X1_tau;
  X1_g_ad_k1 = -X1_g_ad(n-1,:) / p.X1_tau_ad;
  X2_V_k1 = ( (p.X2_E_L-X2_V(n-1,:)) - X2_R*X2_g_ad(n-1,:).*(X2_V(n-1,:)-p.X2_E_k) - X2_R*((((p.X2_R2_PSC_gSYN.*((X2_R2_PSC_s(n-1,:).*X2_R2_PSC_F(n-1,:).*X2_R2_PSC_P(n-1,:))*X2_R2_PSC_netcon).*(X2_V(n-1,:)-p.X2_R2_PSC_ESYN))))) + X2_R*p.X2_Itonic.*X2_Imask + X2_R*p.X2_noise.*randn(1,p.X2_Npop) ) / X2_tau;
  X2_g_ad_k1 = -X2_g_ad(n-1,:) / p.X2_tau_ad;
  C_V_k1 = ( (p.C_E_L-C_V(n-1,:)) - C_R*C_g_ad(n-1,:).*(C_V(n-1,:)-p.C_E_k) - C_R*((((p.C_R2_PSC_gSYN.*((C_R2_PSC_s(n-1,:).*C_R2_PSC_F(n-1,:).*C_R2_PSC_P(n-1,:))*C_R2_PSC_netcon).*(C_V(n-1,:)-p.C_R2_PSC_ESYN))))+((((p.C_S2_PSC_gSYN.*((C_S2_PSC_s(n-1,:).*C_S2_PSC_F(n-1,:).*C_S2_PSC_P(n-1,:))*C_S2_PSC_netcon).*(C_V(n-1,:)-p.C_S2_PSC_ESYN)))))) + C_R*p.C_Itonic.*C_Imask + C_R*p.C_noise.*randn(1,p.C_Npop) ) / C_tau;
  C_g_ad_k1 = -C_g_ad(n-1,:) / p.C_tau_ad;
  R1_IC_PSC_s_k1 = ( R1_IC_PSC_scale * R1_IC_PSC_x(n-1,:) - R1_IC_PSC_s(n-1,:) )/p.R1_IC_PSC_tauR;
  R1_IC_PSC_x_k1 = -R1_IC_PSC_x(n-1,:)/p.R1_IC_PSC_tauD;
  R1_IC_PSC_F_k1 = (1 - R1_IC_PSC_F(n-1,:))/p.R1_IC_PSC_tauF;
  R1_IC_PSC_P_k1 = (1 - R1_IC_PSC_P(n-1,:))/p.R1_IC_PSC_tauP;
  S1_IC_PSC_s_k1 = ( S1_IC_PSC_scale * S1_IC_PSC_x(n-1,:) - S1_IC_PSC_s(n-1,:) )/p.S1_IC_PSC_tauR;
  S1_IC_PSC_x_k1 = -S1_IC_PSC_x(n-1,:)/p.S1_IC_PSC_tauD;
  S1_IC_PSC_F_k1 = (1 - S1_IC_PSC_F(n-1,:))/p.S1_IC_PSC_tauF;
  S1_IC_PSC_P_k1 = (1 - S1_IC_PSC_P(n-1,:))/p.S1_IC_PSC_tauP;
  X1_R1_PSC_s_k1 = ( X1_R1_PSC_scale * X1_R1_PSC_x(n-1,:) - X1_R1_PSC_s(n-1,:) )/p.X1_R1_PSC_tauR;
  X1_R1_PSC_x_k1 = -X1_R1_PSC_x(n-1,:)/p.X1_R1_PSC_tauD;
  X1_R1_PSC_F_k1 = (1 - X1_R1_PSC_F(n-1,:))/p.X1_R1_PSC_tauF;
  X1_R1_PSC_P_k1 = (1 - X1_R1_PSC_P(n-1,:))/p.X1_R1_PSC_tauP;
  R1_X1_PSC_s_k1 = ( R1_X1_PSC_scale * R1_X1_PSC_x(n-1,:) - R1_X1_PSC_s(n-1,:) )/p.R1_X1_PSC_tauR;
  R1_X1_PSC_x_k1 = -R1_X1_PSC_x(n-1,:)/p.R1_X1_PSC_tauD;
  R1_X1_PSC_F_k1 = (1 - R1_X1_PSC_F(n-1,:))/p.R1_X1_PSC_tauF;
  R1_X1_PSC_P_k1 = (1 - R1_X1_PSC_P(n-1,:))/p.R1_X1_PSC_tauP;
  R1_S1_PSC_s_k1 = ( R1_S1_PSC_scale * R1_S1_PSC_x(n-1,:) - R1_S1_PSC_s(n-1,:) )/p.R1_S1_PSC_tauR;
  R1_S1_PSC_x_k1 = -R1_S1_PSC_x(n-1,:)/p.R1_S1_PSC_tauD;
  R1_S1_PSC_F_k1 = (1 - R1_S1_PSC_F(n-1,:))/p.R1_S1_PSC_tauF;
  R1_S1_PSC_P_k1 = (1 - R1_S1_PSC_P(n-1,:))/p.R1_S1_PSC_tauP;
  S1_R1_PSC_s_k1 = ( S1_R1_PSC_scale * S1_R1_PSC_x(n-1,:) - S1_R1_PSC_s(n-1,:) )/p.S1_R1_PSC_tauR;
  S1_R1_PSC_x_k1 = -S1_R1_PSC_x(n-1,:)/p.S1_R1_PSC_tauD;
  S1_R1_PSC_F_k1 = (1 - S1_R1_PSC_F(n-1,:))/p.S1_R1_PSC_tauF;
  S1_R1_PSC_P_k1 = (1 - S1_R1_PSC_P(n-1,:))/p.S1_R1_PSC_tauP;
  R2_R1_PSC_s_k1 = ( R2_R1_PSC_scale * R2_R1_PSC_x(n-1,:) - R2_R1_PSC_s(n-1,:) )/p.R2_R1_PSC_tauR;
  R2_R1_PSC_x_k1 = -R2_R1_PSC_x(n-1,:)/p.R2_R1_PSC_tauD;
  R2_R1_PSC_F_k1 = (1 - R2_R1_PSC_F(n-1,:))/p.R2_R1_PSC_tauF;
  R2_R1_PSC_P_k1 = (1 - R2_R1_PSC_P(n-1,:))/p.R2_R1_PSC_tauP;
  S2_R1_PSC_s_k1 = ( S2_R1_PSC_scale * S2_R1_PSC_x(n-1,:) - S2_R1_PSC_s(n-1,:) )/p.S2_R1_PSC_tauR;
  S2_R1_PSC_x_k1 = -S2_R1_PSC_x(n-1,:)/p.S2_R1_PSC_tauD;
  S2_R1_PSC_F_k1 = (1 - S2_R1_PSC_F(n-1,:))/p.S2_R1_PSC_tauF;
  S2_R1_PSC_P_k1 = (1 - S2_R1_PSC_P(n-1,:))/p.S2_R1_PSC_tauP;
  R2_S2_PSC_s_k1 = ( R2_S2_PSC_scale * R2_S2_PSC_x(n-1,:) - R2_S2_PSC_s(n-1,:) )/p.R2_S2_PSC_tauR;
  R2_S2_PSC_x_k1 = -R2_S2_PSC_x(n-1,:)/p.R2_S2_PSC_tauD;
  R2_S2_PSC_F_k1 = (1 - R2_S2_PSC_F(n-1,:))/p.R2_S2_PSC_tauF;
  R2_S2_PSC_P_k1 = (1 - R2_S2_PSC_P(n-1,:))/p.R2_S2_PSC_tauP;
  S2_R2_PSC_s_k1 = ( S2_R2_PSC_scale * S2_R2_PSC_x(n-1,:) - S2_R2_PSC_s(n-1,:) )/p.S2_R2_PSC_tauR;
  S2_R2_PSC_x_k1 = -S2_R2_PSC_x(n-1,:)/p.S2_R2_PSC_tauD;
  S2_R2_PSC_F_k1 = (1 - S2_R2_PSC_F(n-1,:))/p.S2_R2_PSC_tauF;
  S2_R2_PSC_P_k1 = (1 - S2_R2_PSC_P(n-1,:))/p.S2_R2_PSC_tauP;
  X2_R2_PSC_s_k1 = ( X2_R2_PSC_scale * X2_R2_PSC_x(n-1,:) - X2_R2_PSC_s(n-1,:) )/p.X2_R2_PSC_tauR;
  X2_R2_PSC_x_k1 = -X2_R2_PSC_x(n-1,:)/p.X2_R2_PSC_tauD;
  X2_R2_PSC_F_k1 = (1 - X2_R2_PSC_F(n-1,:))/p.X2_R2_PSC_tauF;
  X2_R2_PSC_P_k1 = (1 - X2_R2_PSC_P(n-1,:))/p.X2_R2_PSC_tauP;
  R2_X2_PSC_s_k1 = ( R2_X2_PSC_scale * R2_X2_PSC_x(n-1,:) - R2_X2_PSC_s(n-1,:) )/p.R2_X2_PSC_tauR;
  R2_X2_PSC_x_k1 = -R2_X2_PSC_x(n-1,:)/p.R2_X2_PSC_tauD;
  R2_X2_PSC_F_k1 = (1 - R2_X2_PSC_F(n-1,:))/p.R2_X2_PSC_tauF;
  R2_X2_PSC_P_k1 = (1 - R2_X2_PSC_P(n-1,:))/p.R2_X2_PSC_tauP;
  R2_R2_iNoise_V3_sn_k1 = ( R2_R2_iNoise_V3_scale * R2_R2_iNoise_V3_xn(n-1,:) - R2_R2_iNoise_V3_sn(n-1,:) )/p.R2_R2_iNoise_V3_tauR_N;
  R2_R2_iNoise_V3_xn_k1 = -R2_R2_iNoise_V3_xn(n-1,:)/p.R2_R2_iNoise_V3_tauD_N + R2_R2_iNoise_V3_token(k,:)/p.R2_R2_iNoise_V3_dt;
  C_R2_PSC_s_k1 = ( C_R2_PSC_scale * C_R2_PSC_x(n-1,:) - C_R2_PSC_s(n-1,:) )/p.C_R2_PSC_tauR;
  C_R2_PSC_x_k1 = -C_R2_PSC_x(n-1,:)/p.C_R2_PSC_tauD;
  C_R2_PSC_F_k1 = (1 - C_R2_PSC_F(n-1,:))/p.C_R2_PSC_tauF;
  C_R2_PSC_P_k1 = (1 - C_R2_PSC_P(n-1,:))/p.C_R2_PSC_tauP;
  C_S2_PSC_s_k1 = ( C_S2_PSC_scale * C_S2_PSC_x(n-1,:) - C_S2_PSC_s(n-1,:) )/p.C_S2_PSC_tauR;
  C_S2_PSC_x_k1 = -C_S2_PSC_x(n-1,:)/p.C_S2_PSC_tauD;
  C_S2_PSC_F_k1 = (1 - C_S2_PSC_F(n-1,:))/p.C_S2_PSC_tauF;
  C_S2_PSC_P_k1 = (1 - C_S2_PSC_P(n-1,:))/p.C_S2_PSC_tauP;

  % ------------------------------------------------------------
  % Update state variables:
  % ------------------------------------------------------------
  IC_V(n,:) = IC_V(n-1,:)+dt*IC_V_k1;
  IC_g_ad(n,:) = IC_g_ad(n-1,:)+dt*IC_g_ad_k1;
  S1_V(n,:) = S1_V(n-1,:)+dt*S1_V_k1;
  S1_g_ad(n,:) = S1_g_ad(n-1,:)+dt*S1_g_ad_k1;
  R1_V(n,:) = R1_V(n-1,:)+dt*R1_V_k1;
  R1_g_ad(n,:) = R1_g_ad(n-1,:)+dt*R1_g_ad_k1;
  S2_V(n,:) = S2_V(n-1,:)+dt*S2_V_k1;
  S2_g_ad(n,:) = S2_g_ad(n-1,:)+dt*S2_g_ad_k1;
  R2_V(n,:) = R2_V(n-1,:)+dt*R2_V_k1;
  R2_g_ad(n,:) = R2_g_ad(n-1,:)+dt*R2_g_ad_k1;
  X1_V(n,:) = X1_V(n-1,:)+dt*X1_V_k1;
  X1_g_ad(n,:) = X1_g_ad(n-1,:)+dt*X1_g_ad_k1;
  X2_V(n,:) = X2_V(n-1,:)+dt*X2_V_k1;
  X2_g_ad(n,:) = X2_g_ad(n-1,:)+dt*X2_g_ad_k1;
  C_V(n,:) = C_V(n-1,:)+dt*C_V_k1;
  C_g_ad(n,:) = C_g_ad(n-1,:)+dt*C_g_ad_k1;
  R1_IC_PSC_s(n,:) = R1_IC_PSC_s(n-1,:)+dt*R1_IC_PSC_s_k1;
  R1_IC_PSC_x(n,:) = R1_IC_PSC_x(n-1,:)+dt*R1_IC_PSC_x_k1;
  R1_IC_PSC_F(n,:) = R1_IC_PSC_F(n-1,:)+dt*R1_IC_PSC_F_k1;
  R1_IC_PSC_P(n,:) = R1_IC_PSC_P(n-1,:)+dt*R1_IC_PSC_P_k1;
  S1_IC_PSC_s(n,:) = S1_IC_PSC_s(n-1,:)+dt*S1_IC_PSC_s_k1;
  S1_IC_PSC_x(n,:) = S1_IC_PSC_x(n-1,:)+dt*S1_IC_PSC_x_k1;
  S1_IC_PSC_F(n,:) = S1_IC_PSC_F(n-1,:)+dt*S1_IC_PSC_F_k1;
  S1_IC_PSC_P(n,:) = S1_IC_PSC_P(n-1,:)+dt*S1_IC_PSC_P_k1;
  X1_R1_PSC_s(n,:) = X1_R1_PSC_s(n-1,:)+dt*X1_R1_PSC_s_k1;
  X1_R1_PSC_x(n,:) = X1_R1_PSC_x(n-1,:)+dt*X1_R1_PSC_x_k1;
  X1_R1_PSC_F(n,:) = X1_R1_PSC_F(n-1,:)+dt*X1_R1_PSC_F_k1;
  X1_R1_PSC_P(n,:) = X1_R1_PSC_P(n-1,:)+dt*X1_R1_PSC_P_k1;
  R1_X1_PSC_s(n,:) = R1_X1_PSC_s(n-1,:)+dt*R1_X1_PSC_s_k1;
  R1_X1_PSC_x(n,:) = R1_X1_PSC_x(n-1,:)+dt*R1_X1_PSC_x_k1;
  R1_X1_PSC_F(n,:) = R1_X1_PSC_F(n-1,:)+dt*R1_X1_PSC_F_k1;
  R1_X1_PSC_P(n,:) = R1_X1_PSC_P(n-1,:)+dt*R1_X1_PSC_P_k1;
  R1_S1_PSC_s(n,:) = R1_S1_PSC_s(n-1,:)+dt*R1_S1_PSC_s_k1;
  R1_S1_PSC_x(n,:) = R1_S1_PSC_x(n-1,:)+dt*R1_S1_PSC_x_k1;
  R1_S1_PSC_F(n,:) = R1_S1_PSC_F(n-1,:)+dt*R1_S1_PSC_F_k1;
  R1_S1_PSC_P(n,:) = R1_S1_PSC_P(n-1,:)+dt*R1_S1_PSC_P_k1;
  S1_R1_PSC_s(n,:) = S1_R1_PSC_s(n-1,:)+dt*S1_R1_PSC_s_k1;
  S1_R1_PSC_x(n,:) = S1_R1_PSC_x(n-1,:)+dt*S1_R1_PSC_x_k1;
  S1_R1_PSC_F(n,:) = S1_R1_PSC_F(n-1,:)+dt*S1_R1_PSC_F_k1;
  S1_R1_PSC_P(n,:) = S1_R1_PSC_P(n-1,:)+dt*S1_R1_PSC_P_k1;
  R2_R1_PSC_s(n,:) = R2_R1_PSC_s(n-1,:)+dt*R2_R1_PSC_s_k1;
  R2_R1_PSC_x(n,:) = R2_R1_PSC_x(n-1,:)+dt*R2_R1_PSC_x_k1;
  R2_R1_PSC_F(n,:) = R2_R1_PSC_F(n-1,:)+dt*R2_R1_PSC_F_k1;
  R2_R1_PSC_P(n,:) = R2_R1_PSC_P(n-1,:)+dt*R2_R1_PSC_P_k1;
  S2_R1_PSC_s(n,:) = S2_R1_PSC_s(n-1,:)+dt*S2_R1_PSC_s_k1;
  S2_R1_PSC_x(n,:) = S2_R1_PSC_x(n-1,:)+dt*S2_R1_PSC_x_k1;
  S2_R1_PSC_F(n,:) = S2_R1_PSC_F(n-1,:)+dt*S2_R1_PSC_F_k1;
  S2_R1_PSC_P(n,:) = S2_R1_PSC_P(n-1,:)+dt*S2_R1_PSC_P_k1;
  R2_S2_PSC_s(n,:) = R2_S2_PSC_s(n-1,:)+dt*R2_S2_PSC_s_k1;
  R2_S2_PSC_x(n,:) = R2_S2_PSC_x(n-1,:)+dt*R2_S2_PSC_x_k1;
  R2_S2_PSC_F(n,:) = R2_S2_PSC_F(n-1,:)+dt*R2_S2_PSC_F_k1;
  R2_S2_PSC_P(n,:) = R2_S2_PSC_P(n-1,:)+dt*R2_S2_PSC_P_k1;
  S2_R2_PSC_s(n,:) = S2_R2_PSC_s(n-1,:)+dt*S2_R2_PSC_s_k1;
  S2_R2_PSC_x(n,:) = S2_R2_PSC_x(n-1,:)+dt*S2_R2_PSC_x_k1;
  S2_R2_PSC_F(n,:) = S2_R2_PSC_F(n-1,:)+dt*S2_R2_PSC_F_k1;
  S2_R2_PSC_P(n,:) = S2_R2_PSC_P(n-1,:)+dt*S2_R2_PSC_P_k1;
  X2_R2_PSC_s(n,:) = X2_R2_PSC_s(n-1,:)+dt*X2_R2_PSC_s_k1;
  X2_R2_PSC_x(n,:) = X2_R2_PSC_x(n-1,:)+dt*X2_R2_PSC_x_k1;
  X2_R2_PSC_F(n,:) = X2_R2_PSC_F(n-1,:)+dt*X2_R2_PSC_F_k1;
  X2_R2_PSC_P(n,:) = X2_R2_PSC_P(n-1,:)+dt*X2_R2_PSC_P_k1;
  R2_X2_PSC_s(n,:) = R2_X2_PSC_s(n-1,:)+dt*R2_X2_PSC_s_k1;
  R2_X2_PSC_x(n,:) = R2_X2_PSC_x(n-1,:)+dt*R2_X2_PSC_x_k1;
  R2_X2_PSC_F(n,:) = R2_X2_PSC_F(n-1,:)+dt*R2_X2_PSC_F_k1;
  R2_X2_PSC_P(n,:) = R2_X2_PSC_P(n-1,:)+dt*R2_X2_PSC_P_k1;
  R2_R2_iNoise_V3_sn(n,:) = R2_R2_iNoise_V3_sn(n-1,:)+dt*R2_R2_iNoise_V3_sn_k1;
  R2_R2_iNoise_V3_xn(n,:) = R2_R2_iNoise_V3_xn(n-1,:)+dt*R2_R2_iNoise_V3_xn_k1;
  C_R2_PSC_s(n,:) = C_R2_PSC_s(n-1,:)+dt*C_R2_PSC_s_k1;
  C_R2_PSC_x(n,:) = C_R2_PSC_x(n-1,:)+dt*C_R2_PSC_x_k1;
  C_R2_PSC_F(n,:) = C_R2_PSC_F(n-1,:)+dt*C_R2_PSC_F_k1;
  C_R2_PSC_P(n,:) = C_R2_PSC_P(n-1,:)+dt*C_R2_PSC_P_k1;
  C_S2_PSC_s(n,:) = C_S2_PSC_s(n-1,:)+dt*C_S2_PSC_s_k1;
  C_S2_PSC_x(n,:) = C_S2_PSC_x(n-1,:)+dt*C_S2_PSC_x_k1;
  C_S2_PSC_F(n,:) = C_S2_PSC_F(n-1,:)+dt*C_S2_PSC_F_k1;
  C_S2_PSC_P(n,:) = C_S2_PSC_P(n-1,:)+dt*C_S2_PSC_P_k1;

  % ------------------------------------------------------------
  % Conditional actions:
  % ------------------------------------------------------------
  conditional_test=any(C_V(n,:)>=p.C_V_thresh&C_V(n-1,:)<p.C_V_thresh);
  conditional_indx=(C_V(n,:)>=p.C_V_thresh&C_V(n-1,:)<p.C_V_thresh);
  if conditional_test, C_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); C_tspike(C_buffer_index(i),i)=t; C_buffer_index(i)=mod(-1+(C_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(X2_V(n,:)>=p.X2_V_thresh&X2_V(n-1,:)<p.X2_V_thresh);
  conditional_indx=(X2_V(n,:)>=p.X2_V_thresh&X2_V(n-1,:)<p.X2_V_thresh);
  if conditional_test, X2_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); X2_tspike(X2_buffer_index(i),i)=t; X2_buffer_index(i)=mod(-1+(X2_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(X1_V(n,:)>=p.X1_V_thresh&X1_V(n-1,:)<p.X1_V_thresh);
  conditional_indx=(X1_V(n,:)>=p.X1_V_thresh&X1_V(n-1,:)<p.X1_V_thresh);
  if conditional_test, X1_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); X1_tspike(X1_buffer_index(i),i)=t; X1_buffer_index(i)=mod(-1+(X1_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(R2_V(n,:)>=p.R2_V_thresh&R2_V(n-1,:)<p.R2_V_thresh);
  conditional_indx=(R2_V(n,:)>=p.R2_V_thresh&R2_V(n-1,:)<p.R2_V_thresh);
  if conditional_test, R2_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); R2_tspike(R2_buffer_index(i),i)=t; R2_buffer_index(i)=mod(-1+(R2_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(S2_V(n,:)>=p.S2_V_thresh&S2_V(n-1,:)<p.S2_V_thresh);
  conditional_indx=(S2_V(n,:)>=p.S2_V_thresh&S2_V(n-1,:)<p.S2_V_thresh);
  if conditional_test, S2_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); S2_tspike(S2_buffer_index(i),i)=t; S2_buffer_index(i)=mod(-1+(S2_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(R1_V(n,:)>=p.R1_V_thresh&R1_V(n-1,:)<p.R1_V_thresh);
  conditional_indx=(R1_V(n,:)>=p.R1_V_thresh&R1_V(n-1,:)<p.R1_V_thresh);
  if conditional_test, R1_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); R1_tspike(R1_buffer_index(i),i)=t; R1_buffer_index(i)=mod(-1+(R1_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(S1_V(n,:)>=p.S1_V_thresh&S1_V(n-1,:)<p.S1_V_thresh);
  conditional_indx=(S1_V(n,:)>=p.S1_V_thresh&S1_V(n-1,:)<p.S1_V_thresh);
  if conditional_test, S1_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); S1_tspike(S1_buffer_index(i),i)=t; S1_buffer_index(i)=mod(-1+(S1_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(IC_V(n,:)>=p.IC_V_thresh&IC_V(n-1,:)<p.IC_V_thresh);
  conditional_indx=(IC_V(n,:)>=p.IC_V_thresh&IC_V(n-1,:)<p.IC_V_thresh);
  if conditional_test, IC_V_spikes(n,conditional_indx)=1;inds=find(conditional_indx); for j=1:length(inds), i=inds(j); IC_tspike(IC_buffer_index(i),i)=t; IC_buffer_index(i)=mod(-1+(IC_buffer_index(i)+1),5)+1; end; end
  conditional_test=any(any(IC_V(n,:) > p.IC_V_thresh,1));
  conditional_indx=(any(IC_V(n,:) > p.IC_V_thresh,1));
  if conditional_test, IC_V(n,conditional_indx) = p.IC_V_reset; IC_g_ad(n,conditional_indx) = IC_g_ad(n,conditional_indx) + p.IC_g_inc; end
  conditional_test=any(any(t<=IC_tspike+p.IC_t_ref,1));
  conditional_indx=(any(t<=IC_tspike+p.IC_t_ref,1));
  if conditional_test, IC_V(n,conditional_indx) = p.IC_V_reset; end
  conditional_test=any(any(S1_V(n,:) > p.S1_V_thresh,1));
  conditional_indx=(any(S1_V(n,:) > p.S1_V_thresh,1));
  if conditional_test, S1_V(n,conditional_indx) = p.S1_V_reset; S1_g_ad(n,conditional_indx) = S1_g_ad(n,conditional_indx) + p.S1_g_inc; end
  conditional_test=any(any(t<=S1_tspike+p.S1_t_ref,1));
  conditional_indx=(any(t<=S1_tspike+p.S1_t_ref,1));
  if conditional_test, S1_V(n,conditional_indx) = p.S1_V_reset; end
  conditional_test=any(any(R1_V(n,:) > p.R1_V_thresh,1));
  conditional_indx=(any(R1_V(n,:) > p.R1_V_thresh,1));
  if conditional_test, R1_V(n,conditional_indx) = p.R1_V_reset; R1_g_ad(n,conditional_indx) = R1_g_ad(n,conditional_indx) + p.R1_g_inc; end
  conditional_test=any(any(t<=R1_tspike+p.R1_t_ref,1));
  conditional_indx=(any(t<=R1_tspike+p.R1_t_ref,1));
  if conditional_test, R1_V(n,conditional_indx) = p.R1_V_reset; end
  conditional_test=any(any(S2_V(n,:) > p.S2_V_thresh,1));
  conditional_indx=(any(S2_V(n,:) > p.S2_V_thresh,1));
  if conditional_test, S2_V(n,conditional_indx) = p.S2_V_reset; S2_g_ad(n,conditional_indx) = S2_g_ad(n,conditional_indx) + p.S2_g_inc; end
  conditional_test=any(any(t<=S2_tspike+p.S2_t_ref,1));
  conditional_indx=(any(t<=S2_tspike+p.S2_t_ref,1));
  if conditional_test, S2_V(n,conditional_indx) = p.S2_V_reset; end
  conditional_test=any(any(R2_V(n,:) > p.R2_V_thresh,1));
  conditional_indx=(any(R2_V(n,:) > p.R2_V_thresh,1));
  if conditional_test, R2_V(n,conditional_indx) = p.R2_V_reset; R2_g_ad(n,conditional_indx) = R2_g_ad(n,conditional_indx) + p.R2_g_inc; end
  conditional_test=any(any(t<=R2_tspike+p.R2_t_ref,1));
  conditional_indx=(any(t<=R2_tspike+p.R2_t_ref,1));
  if conditional_test, R2_V(n,conditional_indx) = p.R2_V_reset; end
  conditional_test=any(any(X1_V(n,:) > p.X1_V_thresh,1));
  conditional_indx=(any(X1_V(n,:) > p.X1_V_thresh,1));
  if conditional_test, X1_V(n,conditional_indx) = p.X1_V_reset; X1_g_ad(n,conditional_indx) = X1_g_ad(n,conditional_indx) + p.X1_g_inc; end
  conditional_test=any(any(t<=X1_tspike+p.X1_t_ref,1));
  conditional_indx=(any(t<=X1_tspike+p.X1_t_ref,1));
  if conditional_test, X1_V(n,conditional_indx) = p.X1_V_reset; end
  conditional_test=any(any(X2_V(n,:) > p.X2_V_thresh,1));
  conditional_indx=(any(X2_V(n,:) > p.X2_V_thresh,1));
  if conditional_test, X2_V(n,conditional_indx) = p.X2_V_reset; X2_g_ad(n,conditional_indx) = X2_g_ad(n,conditional_indx) + p.X2_g_inc; end
  conditional_test=any(any(t<=X2_tspike+p.X2_t_ref,1));
  conditional_indx=(any(t<=X2_tspike+p.X2_t_ref,1));
  if conditional_test, X2_V(n,conditional_indx) = p.X2_V_reset; end
  conditional_test=any(any(C_V(n,:) > p.C_V_thresh,1));
  conditional_indx=(any(C_V(n,:) > p.C_V_thresh,1));
  if conditional_test, C_V(n,conditional_indx) = p.C_V_reset; C_g_ad(n,conditional_indx) = C_g_ad(n,conditional_indx) + p.C_g_inc; end
  conditional_test=any(any(t<=C_tspike+p.C_t_ref,1));
  conditional_indx=(any(t<=C_tspike+p.C_t_ref,1));
  if conditional_test, C_V(n,conditional_indx) = p.C_V_reset; end
  conditional_test=any(any(t == IC_tspike+p.R1_IC_PSC_delay,1));
  conditional_indx=(any(t == IC_tspike+p.R1_IC_PSC_delay,1));
  if conditional_test, R1_IC_PSC_x(n,conditional_indx) = R1_IC_PSC_x(n,conditional_indx) + 1;R1_IC_PSC_F(n,conditional_indx) = R1_IC_PSC_F(n,conditional_indx) + p.R1_IC_PSC_fF*(p.R1_IC_PSC_maxF-R1_IC_PSC_F(n,conditional_indx)); R1_IC_PSC_P(n,conditional_indx) = R1_IC_PSC_P(n,conditional_indx)*(1 - p.R1_IC_PSC_fP); end
  conditional_test=any(any(t == IC_tspike+p.S1_IC_PSC_delay,1));
  conditional_indx=(any(t == IC_tspike+p.S1_IC_PSC_delay,1));
  if conditional_test, S1_IC_PSC_x(n,conditional_indx) = S1_IC_PSC_x(n,conditional_indx) + 1;S1_IC_PSC_F(n,conditional_indx) = S1_IC_PSC_F(n,conditional_indx) + p.S1_IC_PSC_fF*(p.S1_IC_PSC_maxF-S1_IC_PSC_F(n,conditional_indx)); S1_IC_PSC_P(n,conditional_indx) = S1_IC_PSC_P(n,conditional_indx)*(1 - p.S1_IC_PSC_fP); end
  conditional_test=any(any(t == R1_tspike+p.X1_R1_PSC_delay,1));
  conditional_indx=(any(t == R1_tspike+p.X1_R1_PSC_delay,1));
  if conditional_test, X1_R1_PSC_x(n,conditional_indx) = X1_R1_PSC_x(n,conditional_indx) + 1;X1_R1_PSC_F(n,conditional_indx) = X1_R1_PSC_F(n,conditional_indx) + p.X1_R1_PSC_fF*(p.X1_R1_PSC_maxF-X1_R1_PSC_F(n,conditional_indx)); X1_R1_PSC_P(n,conditional_indx) = X1_R1_PSC_P(n,conditional_indx)*(1 - p.X1_R1_PSC_fP); end
  conditional_test=any(any(t == X1_tspike+p.R1_X1_PSC_delay,1));
  conditional_indx=(any(t == X1_tspike+p.R1_X1_PSC_delay,1));
  if conditional_test, R1_X1_PSC_x(n,conditional_indx) = R1_X1_PSC_x(n,conditional_indx) + 1;R1_X1_PSC_F(n,conditional_indx) = R1_X1_PSC_F(n,conditional_indx) + p.R1_X1_PSC_fF*(p.R1_X1_PSC_maxF-R1_X1_PSC_F(n,conditional_indx)); R1_X1_PSC_P(n,conditional_indx) = R1_X1_PSC_P(n,conditional_indx)*(1 - p.R1_X1_PSC_fP); end
  conditional_test=any(any(t == S1_tspike+p.R1_S1_PSC_delay,1));
  conditional_indx=(any(t == S1_tspike+p.R1_S1_PSC_delay,1));
  if conditional_test, R1_S1_PSC_x(n,conditional_indx) = R1_S1_PSC_x(n,conditional_indx) + 1;R1_S1_PSC_F(n,conditional_indx) = R1_S1_PSC_F(n,conditional_indx) + p.R1_S1_PSC_fF*(p.R1_S1_PSC_maxF-R1_S1_PSC_F(n,conditional_indx)); R1_S1_PSC_P(n,conditional_indx) = R1_S1_PSC_P(n,conditional_indx)*(1 - p.R1_S1_PSC_fP); end
  conditional_test=any(any(t == R1_tspike+p.S1_R1_PSC_delay,1));
  conditional_indx=(any(t == R1_tspike+p.S1_R1_PSC_delay,1));
  if conditional_test, S1_R1_PSC_x(n,conditional_indx) = S1_R1_PSC_x(n,conditional_indx) + 1;S1_R1_PSC_F(n,conditional_indx) = S1_R1_PSC_F(n,conditional_indx) + p.S1_R1_PSC_fF*(p.S1_R1_PSC_maxF-S1_R1_PSC_F(n,conditional_indx)); S1_R1_PSC_P(n,conditional_indx) = S1_R1_PSC_P(n,conditional_indx)*(1 - p.S1_R1_PSC_fP); end
  conditional_test=any(any(t == R1_tspike+p.R2_R1_PSC_delay,1));
  conditional_indx=(any(t == R1_tspike+p.R2_R1_PSC_delay,1));
  if conditional_test, R2_R1_PSC_x(n,conditional_indx) = R2_R1_PSC_x(n,conditional_indx) + 1;R2_R1_PSC_F(n,conditional_indx) = R2_R1_PSC_F(n,conditional_indx) + p.R2_R1_PSC_fF*(p.R2_R1_PSC_maxF-R2_R1_PSC_F(n,conditional_indx)); R2_R1_PSC_P(n,conditional_indx) = R2_R1_PSC_P(n,conditional_indx)*(1 - p.R2_R1_PSC_fP); end
  conditional_test=any(any(t == R1_tspike+p.S2_R1_PSC_delay,1));
  conditional_indx=(any(t == R1_tspike+p.S2_R1_PSC_delay,1));
  if conditional_test, S2_R1_PSC_x(n,conditional_indx) = S2_R1_PSC_x(n,conditional_indx) + 1;S2_R1_PSC_F(n,conditional_indx) = S2_R1_PSC_F(n,conditional_indx) + p.S2_R1_PSC_fF*(p.S2_R1_PSC_maxF-S2_R1_PSC_F(n,conditional_indx)); S2_R1_PSC_P(n,conditional_indx) = S2_R1_PSC_P(n,conditional_indx)*(1 - p.S2_R1_PSC_fP); end
  conditional_test=any(any(t == S2_tspike+p.R2_S2_PSC_delay,1));
  conditional_indx=(any(t == S2_tspike+p.R2_S2_PSC_delay,1));
  if conditional_test, R2_S2_PSC_x(n,conditional_indx) = R2_S2_PSC_x(n,conditional_indx) + 1;R2_S2_PSC_F(n,conditional_indx) = R2_S2_PSC_F(n,conditional_indx) + p.R2_S2_PSC_fF*(p.R2_S2_PSC_maxF-R2_S2_PSC_F(n,conditional_indx)); R2_S2_PSC_P(n,conditional_indx) = R2_S2_PSC_P(n,conditional_indx)*(1 - p.R2_S2_PSC_fP); end
  conditional_test=any(any(t == R2_tspike+p.S2_R2_PSC_delay,1));
  conditional_indx=(any(t == R2_tspike+p.S2_R2_PSC_delay,1));
  if conditional_test, S2_R2_PSC_x(n,conditional_indx) = S2_R2_PSC_x(n,conditional_indx) + 1;S2_R2_PSC_F(n,conditional_indx) = S2_R2_PSC_F(n,conditional_indx) + p.S2_R2_PSC_fF*(p.S2_R2_PSC_maxF-S2_R2_PSC_F(n,conditional_indx)); S2_R2_PSC_P(n,conditional_indx) = S2_R2_PSC_P(n,conditional_indx)*(1 - p.S2_R2_PSC_fP); end
  conditional_test=any(any(t == R2_tspike+p.X2_R2_PSC_delay,1));
  conditional_indx=(any(t == R2_tspike+p.X2_R2_PSC_delay,1));
  if conditional_test, X2_R2_PSC_x(n,conditional_indx) = X2_R2_PSC_x(n,conditional_indx) + 1;X2_R2_PSC_F(n,conditional_indx) = X2_R2_PSC_F(n,conditional_indx) + p.X2_R2_PSC_fF*(p.X2_R2_PSC_maxF-X2_R2_PSC_F(n,conditional_indx)); X2_R2_PSC_P(n,conditional_indx) = X2_R2_PSC_P(n,conditional_indx)*(1 - p.X2_R2_PSC_fP); end
  conditional_test=any(any(t == X2_tspike+p.R2_X2_PSC_delay,1));
  conditional_indx=(any(t == X2_tspike+p.R2_X2_PSC_delay,1));
  if conditional_test, R2_X2_PSC_x(n,conditional_indx) = R2_X2_PSC_x(n,conditional_indx) + 1;R2_X2_PSC_F(n,conditional_indx) = R2_X2_PSC_F(n,conditional_indx) + p.R2_X2_PSC_fF*(p.R2_X2_PSC_maxF-R2_X2_PSC_F(n,conditional_indx)); R2_X2_PSC_P(n,conditional_indx) = R2_X2_PSC_P(n,conditional_indx)*(1 - p.R2_X2_PSC_fP); end
  conditional_test=any(any(t == R2_tspike+p.C_R2_PSC_delay,1));
  conditional_indx=(any(t == R2_tspike+p.C_R2_PSC_delay,1));
  if conditional_test, C_R2_PSC_x(n,conditional_indx) = C_R2_PSC_x(n,conditional_indx) + 1;C_R2_PSC_F(n,conditional_indx) = C_R2_PSC_F(n,conditional_indx) + p.C_R2_PSC_fF*(p.C_R2_PSC_maxF-C_R2_PSC_F(n,conditional_indx)); C_R2_PSC_P(n,conditional_indx) = C_R2_PSC_P(n,conditional_indx)*(1 - p.C_R2_PSC_fP); end
  conditional_test=any(any(t == S2_tspike+p.C_S2_PSC_delay,1));
  conditional_indx=(any(t == S2_tspike+p.C_S2_PSC_delay,1));
  if conditional_test, C_S2_PSC_x(n,conditional_indx) = C_S2_PSC_x(n,conditional_indx) + 1;C_S2_PSC_F(n,conditional_indx) = C_S2_PSC_F(n,conditional_indx) + p.C_S2_PSC_fF*(p.C_S2_PSC_maxF-C_S2_PSC_F(n,conditional_indx)); C_S2_PSC_P(n,conditional_indx) = C_S2_PSC_P(n,conditional_indx)*(1 - p.C_S2_PSC_fP); end

  % ------------------------------------------------------------
  % Update monitors:
  % ------------------------------------------------------------
  IC_IC_IC_iIC(n,:)=p.IC_IC_IC_g_postIC*(IC_IC_IC_input(k,:)*IC_IC_IC_netcon).*(IC_V(n,:)-p.IC_IC_IC_E_exc);
  n=n+1;
end

T=T(1:downsample_factor:ntime);

end
