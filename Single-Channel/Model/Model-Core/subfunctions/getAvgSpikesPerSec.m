holder = [];
for i = 1:3000
    holder = [holder, results(i).S2OnOff_V_spikes];
end

holder2 = sum(holder);
%Firing rate in avg spikes/s
fr = sum(holder2)


%Normal
%716
%732

%30_trials
%216709

%Corrected gain and onset gsyn
%avg of 5 
%673 -- Normal

%763.4 -- gsyn = 20
%596.6 -- gsyn = 18
%681.5 -- gsyn = 19


%avg of 30
%679.11 -- Normal

%681.15 -- Offset only




%Offset only
%No changes gsyn = 15 364
%gsyn = 32  1190
%gsyn = 25  880
%gsyn = 23  787
%gsyn = 22  748
%gsyn = 21  691

%30_trials
%gsyn = 21  264664

%Need to rerun this???

