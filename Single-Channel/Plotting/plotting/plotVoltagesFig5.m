fignum = 1;

%figure(1);
%plot(snn_out(1).R2On_V); hold on
%figure(2);
%plot(snn_out(1).R2On_V_spikes); hold on

indexs = find(snn_out(fignum).R2On_V_spikes == 1);
indexs2 = find(snn_out(fignum).S2OnOff_V_spikes == 1);

holder = snn_out(fignum).R2On_V;
holder(indexs) = 0;

holder2 = snn_out(fignum).S2OnOff_V;
holder2(indexs2) = 0;

figure(5);
plot(holder(9000:15000),'k')

figure(6);
plot(holder2(9000:15000),'r')

%plot(snn_out(1).On_On_IC_iIC(4500:5500))
%figure(2);
%plot(snn_out(1).R2On_R1On_PSC_syn)

%Also can try doing 0.015 abd 0.025 or something instead of 0.025 and 0.035