% use correlation to find delay between two PSTHs

% define kernel
t = (0:dt:500)/1000; % 0-500ms
tau = 0.01; %10 ms
kernel = t.*exp(-t/tau);

% input and output spikes
input_spks = squeeze(spks(:,1,:));
Cspks = [temp(1:numVaried:end).C_V_spikes];


smoothed_in = conv2(input_spks',kernel','same');
smoothed_out = conv2(Cspks,kernel','same');
clear c lags
for i = 1:20
    [c(i,:),lags] = xcorr(smoothed_in(:,i),smoothed_out(:,i));
end
 
figure; plot(lags,c)
xlim([-2000 2000])
xlabel('lags')
ylabel('CC coef')

figure; imagesc(lags,1:20,c);
xlabel('lags')
ylabel('trial')

% output mean delay
[val,idx]=max(c,[],2);
round(mean(lags(idx)))
%% double check the delay is 0 after accounting for it
NumDelayTaps = 51;
input_spks = squeeze(spks(:,1,1:end-NumDelayTaps)); %account for delay
Cspks = [temp(1:numVaried:end).C_V_spikes];
Cspks = Cspks(1+NumDelayTaps:end,:); %account for delay

smoothed_in = conv2(input_spks',kernel','same');
smoothed_out = conv2(Cspks,kernel','same');
clear c lags
for i = 1:20
    [c(i,:),lags] = xcorr(smoothed_in(:,i),smoothed_out(:,i));
end
 
[val,idx]=max(c,[],2);
round(mean(lags(idx)))