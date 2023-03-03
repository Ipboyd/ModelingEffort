
sims = dir('simData');

sims(~contains({sims.name},'02-07-2023 masked, varying noise and strf gain ')) = [];

noises = 6:4:18;

for s = 1:length(sims)

% get strf gain factor from folder name

pattern='(\d+(\.\d+))';
out = regexp(sims(s).name,pattern,'match');
strfgain(s) = str2num(out{1});

% get 
load(fullfile('simData',sims(s).name,'perf_fr_R2On.mat'));

pcs(s,:) = pc.SPIKE;
frs(s,:) = fr;

end

%% plot

figure; surf(noises,strfgain,pcs,'facecolor','b','facealpha',0.5);

xlabel('Output noise (FR)'); ylabel('STRF gain'); xlim([6 18]); ylim([0.09 0.11]); 
zlabel('Performance'); ztickformat('percentage');zlim([50 80]);
title('Output noise and input STRF gain vs. performance')

figure; surf(noises,strfgain,frs,'facecolor','y','facealpha',0.5);

xlabel('Output noise (FR)'); ylabel('STRF gain'); xlim([6 18]); ylim([0.09 0.11]); 
zlabel('FR (Hz)'); zlim([0 80]);
title('Output noise and input STRF gain vs. performance')