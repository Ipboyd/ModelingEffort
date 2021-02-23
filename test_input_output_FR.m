% explore input-output relationship between firing rates!
dynasimPath = 'C:\Users\Kenny\Desktop\GitHub\DynaSim';
addpath(genpath(dynasimPath))
addpath('mechs')

s=[];
nPops=1;

itonic = 0.35;
itonic_range = 0.05:0.01:0.5;
varied_range = itonic_range;

% itonic_range = [0.3];
gSYN_range = [0.05:0.05:0.9];
vary = {'Y','Itonic',itonic_range};


s.pops(1).name='X';
s.pops(1).size= nPops;
s.pops(1).equations='chouLIF';
s.pops(1).parameters={'Itonic',itonic};

s.pops(end+1).name='Y';
s.pops(end).size= nPops;
s.pops(end).equations='chouLIF';
s.pops(end).parameters={'Itonic',itonic};

s.pops(end+1).name='R';
s.pops(end).size= 1;
s.pops(end).equations='chouLIF';

% s.pops(end+1).name='C';
% s.pops(end).size=1;
% s.pops(end).equations='chouLIF';

s.connections(1).direction='X->R';
s.connections(1).mechanism_list={'synDoubleExp'};
s.connections(end).parameters={'gSYN',0.2, 'tauR',0.4, 'tauD',2, 'netcon', ones(nPops,1)}; 

s.connections(end+1).direction='Y->R';
s.connections(end).mechanism_list={'synDoubleExp'};
s.connections(end).parameters={'gSYN',0.25, 'tauR',0.4, 'tauD',20, 'netcon', ones(nPops,1),'ESYN',-80,'delay',2}; 

% s.connections(end+1).direction='R->C';
% s.connections(end).mechanism_list={'synDoubleExp'};
% s.connections(end).parameters={'gSYN',0.18, 'tauR',0.4, 'tauD',2, 'netcon', ones(nPops,1)}; 

% simulation
data=dsSimulate(s,'time_limits',[0 500],'solver','rk1','dt',0.1,'vary',vary,'parfor_flag',1);
% dsPlot(data,'plot_type','rastergram');

clear FR_X FR_Y FR_R

for i = 1:length(varied_range )
    FR_X(i) = (sum(data(i).X_V_spikes))/0.5;
    FR_Y(i) = (sum(data(i).Y_V_spikes))/0.5;
    FR_R(i) = sum(data(i).R_V_spikes)/0.5;
%     FR_C(i) = sum(data(i).C_V_spikes)/0.5;
end
% figure; scatter(varied_range ,FR_R);
% xlabel('I_{tonic}')
% ylabel('Postsynaptic neuron FR')
% 
% figure; scatter(varied_range ,FR_X)
% xlabel('I_{tonic}')
% ylabel('Presynaptic exc neuron FR')
% 
% figure; scatter(varied_range ,FR_Y)
% xlabel('I_{tonic}')
% ylabel('Presynaptic inh neuron FR')

figure;
subplot(3,1,1);
plot(data(7).time,data(7).R_V); ylabel('out')
subplot(3,1,2);
plot(data(7).time,data(7).X_V); ylabel('exc')
subplot(3,1,3);
plot(data(7).time,data(7).Y_V); ylabel('inh')

figure; 
plot(varied_range ,FR_R, 'o-'); hold on;
plot(varied_range ,FR_X, 'o-');
plot(varied_range ,FR_Y, 'o-');
xlabel('I_{tonic}')
ylabel('firing rate')
legend('Postsynaptic neuron FR','Presynaptic exc neuron FR','Presynaptic inh neuron FR')

figure;
plot(FR_Y ,FR_R, 'o-'); hold on;
ylabel('output neuron firing rate')
xlabel('inhibitory neuron firing rate')

