
files = dir(cd);
inds = contains({files.name},'R2On_AMresponse');
files(~inds) = [];

for f = 1:length(files)
    openfig(files(f).name);
    
    for s = 1:5
        subplot(5,1,s);
        lines = findobj(gca,'type','line');
        PSTH = lines(end).YData;
        meanFR = mean(PSTH(t_vec >= 250 & t_vec < t_stim*1000+250)) * (1000/t_bin) / TrialsPerStim;
        title_str = sprintf('Mean FR during AM stim: %.0f Hz',meanFR);
        title(title_str,'fontweight','normal');
        if s < 5
            set(gca,'xticklabels',[])
        end
    end
    
    savefig(gcf,files(f).name);
    close;
end



