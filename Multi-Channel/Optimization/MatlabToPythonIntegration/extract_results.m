figure(1);
subplot(2,1,1)
plot(param_tracker)
subplot(2,1,2)
plot(losses(:,1))

figure(2);
spy(output)