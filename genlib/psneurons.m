% function out = psneurons(rmgsPsein,rmgsPsiin,arp,dt,Ese,Esi,rmgse,rmgsi,
%                          taum,Vap,Vre,Vth,El,noiselev,taue,delay,Type)
%   [OUT,COUNT] = PSNEURON(...) calculates the output of NN leaky integrate
%   and fire neurons in response to postsynaptic input column vectors 
%   RMGSPSEIN and RMGSPSIIN (typically scaled spike trains convolved with 
%   exponentials) of length NS samples, outputting a 1xNN vector COUNT of 
%   spikes from each neuron; and as OUT either the corresponding spike 
%   train (TYPE>=0, default) or the voltage trace V (TYPE=1).
%
%   There is no directly injected current in the model, so the differential
%   equations are integrated using Euler's method with time step dt is
%
%      taum dVdt = rmgsPse*(Ese - V) + rmgsPsi*(Esi - V) + (El - V) + N.
%
%   See psneuron.m for a description of this simple model. Parameters are
%   as follows (all units are in the same time units):
%   
%   01: RMGSPSEIN (NS x NN real double matrix)
%   02: RMGSPSIIN (NS x NN real double matrix)
%       The product of the membrane resistance, the excitatory (inhibitory)
%       synaptic conductance, and the initial scaled excit. (inhib.) 
%       spiking input input to each neuron (usually a sequence of scaled 
%       1's and 0's, *not* a spike train convolved with an exponential!).
%   03: ARP (NN real double vector or scalar) 
%       The absolute refractory period for each or all neurons.
%   04: DT (real double scalar)
%       Integration time step.
%   05: ESE (NN real double vector or scalar)
%   06: ESI (NN real double vector or scalar)
%       Excit. (inhib) synaptic potentials for each or all neurons.
%   07: RMGSE (NN x NN real double matrix)
%   08: RMGSI (NN x NN real double matrix)
%       The strength of the excit. (inhib.) connection from neuron Ni to Nj
%       is stored in entry RMGSE(i,j) (RMGSI(i,j)).
%   09: TAUM (NN real double vector or scalar)
%       Membrane time constants for each or all neurons.
%   10: VAP (NN real double vector or scalar)
%       Action potential voltage for each or all neurons.
%   11: VRE (NN real double vector or scalar) 
%       Reset voltage for each or all neurons.
%   12: VTH (NN real double vector or scalar)
%       Spiking threshold for each or all neurons.
%   13: EL (NN real double vector or scalar)
%       The leak or steady-state potential for each or all neurons.
%   14: NOISELEV (NN real double vector or scalar)
%       N is additive white Gaussian noise with mean zero and standard 
%       deviation NOISELEV for each or all neurons.
%   15: TAUE (NN real double vector or scalar)
%       Synaptic time constants for each or all neurons.
%   16: DELAY (Nn x Nn, scalar, or empty)
%       Synaptic time delay for each or all connections between neurons. 
%       (Empty brackets imply no delay.)
%   17: OUTTYPE (1)
%       An optional input. PSNEURON changes OUT depending on Type:
%        - Type=0 (or is omitted) => OUT is a sequence of 1's and 0's,
%          equivalent to OUT = double(OUT_type1 == Vap)
%        - Type=1 => OUT is the output voltage trace of the cell, or OUT = V
%       Note that in both cases the number of spikes is returned in COUNT.
%
%   For a demo, try the following code, where the high noise (50mV SD) and
%   heavy connectivity (usually) cause rapid firing:
%   
%     Nn = 4;
%     Ns = 100;
%     re = ones(Nn)-eye(Nn);
%     ri = eye(Nn);
%     z = zeros(Ns,Nn);
%     z1 = zeros(Nn,1);
%     plot(psneurons(z,z,2,1,0,-90,re,ri,20,0,-80,-55,-70,50,10,10,50,1))
%   
%  07/17/08, Eric Larson edlarson@bu.edu
%  