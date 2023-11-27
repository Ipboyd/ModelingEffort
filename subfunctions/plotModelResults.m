function plotModelResults(varargin)

snn_out = varargin{1};

if
    info.pops = 'R';
else
    info.pops = 'S';
end

'syn';
'Pde';
'Fac';
'g_ad';
'V';
'V_spikes';

if strcmp(layer,'L2/3')
    info.layer = '2';
else
    info.layer = '1';
end

label = findlabel(info,snn_out(1).labels);

if contains(label,'spikes')
    offset_base = 1; 
else
    offset_base = 20;
end

t_vec = snn_out(1).time;
nTrials = numel(snn_out)/20;

figure; hold on;
for t = 1:nTrials
    plot(t_vec,snn_out(t).(label) - offset_base);
    offset = offset + offset_base*t;
end
xlabel('Time (ms)');

end

function label = findlabel(info,labels)

label = labels(contains(labels,info.layer) & contains(labels,info.pops) & contains(labels,info.layer));


end