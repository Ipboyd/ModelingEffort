
% need to run SpikingNetwork_AMStim with params_AM_varyingStrengths first
% before creating the plots

fp = [0.3 0.4 0.5 0.6];
tau = [60 80 100 120];

[m,n] = meshgrid(fp,tau);
z = [m(:),n(:)];

freqs = [2 4 8 16 32];

for x = 1:4

    figure('unit','inches','position',[4 4 1.8 1.4]);
    for nV = (0:4:15) + x  % (1:4) + (x-1)*4% 
        tempRM = [];
        for a = 1:5
            tempRM = cat(1,tempRM,abs(RM(nV,a).R2On));
        end
        plot([2 4 8 16 32],tempRM,'linewidth',1,'displayname',sprintf('f_P = %.1f, tau_P = %i',z(nV,1),z(nV,2)));
        hold on
    end
    % set(gca,'xtick',1:5,'xticklabel',[8 16 32],'fontsize',8,'ytick',[0.1 0.4 0.7]);
    xlabel('AM frequency (Hz)');
    % ylim([0.1 0.7]); xlim([3 5])
    ylabel('Rate modulation');

    %legend;
end
