
% need to run SpikingNetwork_AMStim with params_AM_varyingStrengths first
% before creating the plots

fp = [0.3 0.4 0.5 0.6];
tau = [60 80 100 120];

[m,n] = meshgrid(fp,tau);
z = [m(:),n(:)];

freqs = [2 4 8 16 32];
freq_RM = [];

for nV = 1:(length(fp)*length(tau))
    tempRM = [];
    for a = 1:5
        tempRM = cat(1,tempRM,abs(RM(nV,a).R2On));
    end

    % interpolate AM frequency at 50% rate modulation
    ind = find(tempRM < 0.5,1);

    % tempfreqs = logspace(log10(freqs(ind-1)),log10(freqs(ind)),400);
    tempfreqs = linspace(freqs(ind-1),freqs(ind),400);
    
    m = (tempRM(ind)-tempRM(ind-1))/(freqs(ind)-freqs(ind-1));
    freq_RM(nV) = (0.5 - tempRM(ind-1))/m + freqs(ind-1);

end

figure('unit','inches','position',[4 4 2.5 2.5]);

surf(fp,tau,reshape(freq_RM,[length(fp) length(tau)]),'facecolor','b','facealpha',0.5);
axis square; set(gca,'fontsize',8,'ydir','normal','xtick',fp,'ytick',tau); 
xlabel('f_P'); ylabel('\tau_P (ms)');
xlabel('Onset g_{SYN}'); ylabel('Offset g_{SYN}');
zlabel('Estimated AM frequency at 0.5 rate modulation (Hz)');
savefig(gcf,fullfile(simDataDir,'AM_vs_vals_envelope.fig'));

figure('unit','inches','position',[4 4 2.5 2.5]);
imagesc(fp,tau,reshape(freq_RM,[length(fp) length(tau)]));
axis square; set(gca,'fontsize',8,'ydir','normal','xtick',fp,'ytick',tau); 
xlabel('f_P'); ylabel('\tau_P (ms)');
xlabel('Onset g_{SYN}'); ylabel('Offset g_{SYN}');
c=colorbar;
%clim([8 12]);
c.Label.String = 'Estimated AM frequency at 50% rate modulation (Hz)';
savefig(gcf,fullfile(simDataDir,'AM_vs_vals_image.fig'));
