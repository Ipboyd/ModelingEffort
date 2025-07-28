epochs = 100;
num_params = 10;

%Extract Losses
losses = [];
a = x{1};
for k = 1:epochs
    %print(float(a{k}))
    losses = [losses, double(a{k})];
end

parameter_vals = [];
b = x{3};
for k = 1:epochs
    parameter_vals = [parameter_vals, double(b{k})];
end

parameter_vals = reshape(parameter_vals, [num_params,epochs]);


%Compensating for erroneous placement of p.append (fixed now)
%parameter_vals = [0.005, parameter_vals(1:49)];


figure;
subplot(2,1,1)
plot(1:epochs, parameter_vals)
xlabel('epochs')
ylabel('R1 to R2 Strength')
subplot(2,1,2)
plot(1:epochs, losses)
xlabel('epochs')
ylabel('loss')