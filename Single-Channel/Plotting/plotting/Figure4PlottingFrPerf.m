fignum = 1;

figlabels = {'E \rightarrow E$','$E \rightarrow PV$','$PV \rightarrow E$'};

figure('Position',[300,300,300,300])

subplot('Position',[0.15,0.45,0.8,0.5])

plot(0:0.1:1,[data(15).perf.R2On.channel1(1,:)]); hold on
plot(0:0.1:1,[data(15).perf.R2On.channel1(2,:)]); hold on 
plot(0:0.1:1,[data(15).perf.R2On.channel1(3,:)]);


ylim([35 100])
ylabel('Performance')
xticklabels('')

subplot('Position',[0.15,0.15,0.8,0.25])
plot(0:0.1:1,[data(15).fr.R2On.channel1],'k')
xticks([0:0.2:1])
xticklabels([0:0.2:1])
xlabel(['$f_p   :' figlabels{fignum}], 'Interpreter', 'latex')
ylabel(sprintf('Firing Rate (Hz)\n'))
ylim([0 50])  



