function plotsimRaster(varargin)

pop = varargin{1};
snn_out = varargin{2};

if nargin > 2
    vary = varargin{3};
else
    vary = 1;
end

jump = length(snn_out)/20;
trials = vary:jump:length(snn_out);

t1 = trials(1:10); t2 = trials(11:20);

raster = horzcat(snn_out(t1).([pop '_V_spikes']))';

if isempty(raster)
    snn_out = fetchSpks(pop,snn_out);
    raster = horzcat(snn_out(t1).([pop '_V_spikes']))';
end

figure('unit','inches','position',[2 2 2.75 3]);
subplot(2,1,1);
plotSpikeRasterFs(flipud(logical(raster)), 'PlotType','vertline','Fs',10000);
xlabel('Time (ms)'); title('Target 1'); xlim([0 3500]);
ylim([0.5 10.5]); 
set(gca,'ytick',[ ],'fontsize',8);

raster = horzcat(snn_out(t2).([pop '_V_spikes']))';

subplot(2,1,2);
plotSpikeRasterFs(flipud(logical(raster)), 'PlotType','vertline','Fs',10000);
xlabel('Time (ms)'); title('Target 2'); xlim([0 3500]);
ylim([0.5 10.5]);
set(gca,'ytick',[ ],'fontsize',8);

sgtitle([pop ' rasters']);

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