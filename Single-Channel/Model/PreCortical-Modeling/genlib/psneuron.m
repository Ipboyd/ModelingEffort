% function out = psneuron(Pse,Psi,arp,dt,Ese,Esi,rmgse,rmgsi,taum,Vap,...
%                         Vre,Vth,El,noiselev,Type)
%   [OUT,COUNT] = PSNEURON(...) calculates the output of a leaky integrate
%   and fire neuron in response to postsynaptic input vectors Pse and Psi
%   (typically spike trains convolved with exponentials), outputting COUNT 
%   the resulting number of spikes; and as OUT either 0) the corresponding 
%   spike train (Type=0, default if ommitted), 1) the voltage trace V 
%   (Type=1), or 2) a list of spike indices (Type=2).
%
%   There is no directly injected current in the model, so the differential
%   equation integrated using Euler's method with time step dt is
%
%      taum dVdt = rmgse*Pse*(Ese - V) + rmgsi*Psi*(Esi - V) + (El - V) + N
%
%   where RMGSE (RMGSI) is the product of the membrane resistance and the
%   excitatory (inhibitory) synapse; PSE (PSI) is the excit. (inhib.)
%   postsynaptic input; ESE (ESI) is the excit. (inhib) synaptic potential;
%   EL is the leak or steady-state potential; and N is additive white
%   Gaussian noise with mean zero and standard deviation noiselev. Whenever
%   V exceeds the threshold voltage VTH, the V is set to the action 
%   potential voltage VAP, and on the next time step V is set to the reset 
%   threshold VRE. Subsequent action potentials cannot occur until the 
%   absolute refractory period arp has passed. Note that V is hard-limited
%   to stay above ESI (to protect against some issues with numerical
%   integration).
%
%   When PSI=0 (scalar) is passed to PSNEURON, the equation is integrated
%   without the rmgsi*Psi*(Esi - V) term, saving computation time.
%
%   Type is an optional input. PSNEURON changes OUT depending on Type:
%    - Type=0 (or is omitted) => OUT is a sequence of 1's and 0's,
%      equivalent to OUT = double(OUT_type1 == Vap)
%    - Type=1 => OUT is the output voltage trace of the cell, or OUT = V
%    - Type=2 => OUT is a list of indices of spikes, equivalent to OUT =
%      find(OUT_type1 == Vap)
%   Note that, in all three cases, the number of generated spikes is
%   returned in COUNT.
%   