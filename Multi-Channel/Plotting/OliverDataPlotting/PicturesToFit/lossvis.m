%m = matfile('run_2025-10-03_17-26-18.mat');

loss = mean(losses(:,2,:),3);
min_loss = min(squeeze(losses(:,2,:))');

figure;
subplot(3,1,1)
plot(loss(50:end)); hold on
plot(movmean(loss(50:end),10))
xlim([-50,250])
subplot(3,1,2)
plot(min_loss);hold on
plot(movmean(min_loss,10))
subplot(3,1,3)
plot(movmean(min_loss,10))