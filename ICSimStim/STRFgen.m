function strf=STRFgen(paramH,paramG,f,dt,nIn,outputNL, freqDom)
%% Generate STRFs with specified parameters
% Create STRF of [f] frequecy channels, and time delays of 40 dts
% Parameters from Amin et al., 2010, J Neurophysiol
%
% First part incoporated from STRFlab

% Takes the number of inputs and outputs for a generalized linear model, together
% with a string [outfunc] which specifies the output unit activation function, and
% returns a strf structure [strf]. The weights are drawn from a zero mean,
% isotropic Gaussian, with variance scaled by the fan-in of the output
% units. This makes use of the Matlab function RANDN and so the seed for
% the random weight initialization can be  set using RANDN('STATE', S)
% where S is the seed value.
%
% INPUT:
% 		[nin] = number of inputs in one time slice of the strf
%    [delays] = row vector of integers, describing latencies for the strf.
%		        (For example [0 1 2] is 3 time lags including no lag)
%	[outputNL] = string indicating the output function and regression type:
%				'linear'     (For use with squared error)
%				'logistic'   (For use with logistic regression)
%				'exponential'(For use with poisson regression)
%   [freqDom] = toggles between options for convolution of stimulus with model weight.  For models with long delays
%				(e.g., auditory data), try freqDom = 1.  For models with shorter delays,
%				freqDom = 0 will usually produce faster results.
%
% OUTPUT:
%	[strf] = a STRF structure containing fields:
%	   .type  = 'glm'
%	  	.nin  = number of inputs in one time slice of the strf (see above)
%  	 .delays  = delay vector input (see above)
%	   .nWts  = total number of weights and biases
%	  .actfn  = string describing the output unit activation function: 'linear',
%			    'logistic', or 'exponential'
% .freqDomain = 1 for convolution in frequency domain.
%	   	  .w1 = nin x 1 vector of weights
%	      .b1 = scalar bias term
%
%
%
%	SEE ALSO
%	linPak, linUnpak, linFwd, linErr, linGrad
%
%(Some code modified from NETLAB)


% Set default option values
% --------------------
if nargin<5
    nIn=1;
end
if nargin<6
    outputNL='linear';
end
if nargin<7
    freqDom=0;
end

strf.type = 'lin';
strf.nIn = nIn;
strf.t =0:dt:39*dt;
strf.delays =0:39;
strf.nWts = (nIn*length(strf.delays) + 1);

% strf.w1=zeros(nIn,length(delays));
strf.b1=0;

nlSet={'linear','logistic','softmax','exponential'};
if ismember(outputNL,nlSet)
    strf.outputNL = outputNL;
else
    error('linInit >< Unknown Output Nonlinearity!');
end
strf.freqDomain = freqDom;

strf.internal.compFwd=1;
strf.internal.prevResp = [];
strf.internal.prevLinResp = [];
strf.internal.dataHash = NaN;

if ~isfield(paramH,'phase')
    paramH.phase=pi/2;
end
%% Generate STRFs with specified parameters
% Create STRF of [f] frequecy channels, and time delays of 40 dts
% Parameters from Amin et al., 2010, J Neurophysiol
if length(fieldnames(paramH))==4
    strf.exp=exp(-.5*((strf.t-paramH.t0)/paramH.BW).^2);
    strf.cos=cos(2*pi*paramH.BTM*(strf.t-paramH.t0)+paramH.phase);
    strf.H=strf.exp.*strf.cos;
elseif length(fieldnames(paramH))==5   %add inhibitory component
    temp=paramH.sti*exp(-.5*((strf.t-paramH.ti)/paramH.BW).^2).*cos(2*pi*paramH.BTM*(strf.t-paramH.ti));
    strf.H=exp(-.5*((strf.t-paramH.t0)/paramH.BW).^2).*cos(2*pi*paramH.BTM*(strf.t-paramH.t0))-temp;
else
    disp('Wrong number of fields in param.H')
    return
end
strf.G=exp(-.5*((f-paramG.f0)/paramG.BW).^2).*cos(2*pi*paramG.BSM*(f-paramG.f0));
strf.w1=strf.G'*strf.H;
strf.f=f;
%%
% subplot(3, 2, 1);
% plot(f,strf.G)
% view(-90,90)
% subplot(3, 2, 2);
% plot(strf.t,strf.H)
% set(gca,'XTickLabel','','YTickLabel','')
