load('Penikis2023_vocodedspeech_amplitudeenvelopes.mat');

% for raw stim
figure;
for n = 1:size(VS_stim_full,1)

    [allTS,env,peakDrv,peakEnv] = find_peakRate(VS_stim_full(n,:)',fs,'amp_linear');
    plot(env,'linewidth',1); hold on; scatter(find(peakDrv),env(peakDrv ~= 0),'filled');
    title(sprintf('VS token %d',n))
    pause; clf;
end