% function out = ineuron(rmI,arp,dt,taum,Vap,Vre,Vth,El,noiselev,Type)
%   [OUT,COUNT] = INEURON(...) calculates the output of a leaky integrate
%   and fire neuron in response to membrane-resistance*current input vector
%   rmI, outputting COUNT the resulting number of spikes; and as OUT either
%   the corresponding spike train (Type=0, default if ommitted), the voltage trace V 
%   (Type=1), or a list of spike indices (Type=2).
%
%   The differential equation integrated using Euler's method with time 
%   step dt is
%
%      taum dVdt = (El - V) + N + rmI
%
%   where El is the leak or steady-state potential; and N is additive white
%   Gaussian noise with mean zero and standard deviation noiselev. Whenever
%   V exceeds the threshold voltage Vth, the V is set to the action 
%   potential voltage Vap, and on the next time step V is set to the reset 
%   threshold Vre. Subsequent action potentials cannot occur until the 
%   absolute refractory period arp has passed.
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
%   To get a sense of the function, try:
%     plot(ineuron(zeros(100,1),0.002,0.001,0.005,0,-80,-55,-70,50))
%     plot(ineuron(zeros(100,1),0.002,0.001,0.005,0,-80,-55,-70,50,1))
%     ineuron(zeros(100,1),0.002,0.001,0.005,0,-80,-55,-70,50,2)
%   