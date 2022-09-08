function plotsimPSTH(varargin)

pop = varargin{1};
snn_out = varargin{2};

if nargin == 3
    vary = varargin{3};
else
    vary = 1;
end

if nargin == 4
    lineFlag = varargin{4};
else
    lineFlag = 0;
end

jump = length(snn_out)/20;

trials = vary:jump:length(snn_out);

t1 = trials(1:10); t2 = trials(11:20);

t_vec = 0:20:3500; % ms

temp = find([snn_out(t1).([pop '_V_spikes'])]);

if isempty(temp)
    snn_out = fetchSpks(pop,snn_out);
end

[inds,~] = ind2sub([35000 10],temp);

psth = histcounts(inds/10,t_vec);
psth(end+1) = 0;
figure('unit','inches','position',[2 2 2.75 3]);

subplot(2,1,1);
if lineFlag
plot(t_vec/1000-0.3,psth,'linewidth',1); hold on;
else
hold off; bar(t_vec/1000-0.3,psth,'facecolor','k');
end

xlabel('Time (ms)'); title('Target 1'); ylim([0 60]);

temp = find([snn_out(t2).([pop '_V_spikes'])]);
[inds,~] = ind2sub([35000 10],temp);

psth = histcounts(inds/10,t_vec);
psth(end+1) = 0;

subplot(2,1,2);
if lineFlag
plot(t_vec/1000-0.3,psth,'linewidth',1); hold on;
else
hold off; bar(t_vec/1000-0.3,psth,'facecolor','k');
end

xlabel('Time (ms)'); title('Target 2'); ylim([0 60]);

end

function snn_out = fetchSpks(pop,snn_out)

V_field = ([pop '_V']);
spk_field = ([pop '_V_spikes']);

if ~contains(pop,'S2')
    V_reset = -52;
else
    V_reset = -50;
end

for n = 1:length(snn_out)
    for ch = 1:size(snn_out(n).(V_field),2)
        a = (snn_out(n).(V_field)(2:end,ch) == V_reset);
        b = diff(snn_out(n).(V_field)(:,ch)) < 0;
        spks = (a == 1 & b == 1); spks(end+1) = 0;
        snn_out(n).(spk_field) = spks;
    end
end

end