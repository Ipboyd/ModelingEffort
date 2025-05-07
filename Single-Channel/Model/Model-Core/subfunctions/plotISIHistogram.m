function plotISIHistogram(varargin)

pop = varargin{1};
snn_out = varargin{2};

if nargin == 3
    vary = varargin{3};
else
    vary = 1;
end

dt = 0.1;

ISI_lim = 100;

numVaries = length(snn_out)/20;

for n = vary:numVaries:20
    ISIs{n} = diff(find(snn_out(n).([pop '_V_spikes']))*dt);
end

figure('unit','inches','position',[5 2.5 6 4.5]); histogram(vertcat(ISIs{:}),0:0.1:ISI_lim); xlim([0 ISI_lim]);
xlabel('ISI (ms)'); title([pop ' ISI histogram']);

end