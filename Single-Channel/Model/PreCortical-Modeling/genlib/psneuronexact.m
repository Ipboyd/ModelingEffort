% function out = psneuronexact(spikes,weights,arp,Ese,Esi,taum,taue,...
%                              Vre,Vth,El,duration,maxOut)
%   [OUT] = PSNEURONEXACT(...) calculates the output of a leaky integrate
%   and fire neuron in response to presynaptic input spikes at times given
%   by the monotonically nondecreasing vector SPIKES with positive 
%   (excitatory) or negative (inhibitory) weights given by the vector
%   WEIGHTS, outputting the corresponding list of evoked spike times.
%
%   There is no directly injected current or noise in the model. It is
%   evaluated using the "Exact" method described by Brette (2005),
%   equivalent to numerically integrating the standard IF equation with an
%   absolute refractory period ARP (see PSNEURON). The scalar values in the
%   vector WEIGHTS correspond to the values of RMGSE and RMGSI which would
%   be passed to PSNEURON, for example.
%
%   In this model, TAUM is the membrane time constant; TAUE is the synaptic
%   time constant; ESE (ESI) is the excit. (inhib) synaptic potential; and
%   EL is the leak or steady-state potential. Whenever the cell voltage
%   exceeds the threshold voltage Vth, an action potential is evoked, and
%   the voltage is set to the reset threshold VRE. Subsequent action
%   potentials cannot occur until the absolute refractory period arp has
%   passed.
%
% 06/09/2009 Eric Larson edlarson@bu.edu
% 