epochs = 20;
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
    
    if k>1
        
        thing = [];
        for l = 1:10
            c = b{k}(l);
         
            thing = [thing, double(c{1})];
        end
       parameter_vals = [parameter_vals, thing];
    else
        parameter_vals = [parameter_vals, double(b{k})];
    end


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
plot(1:epochs, losses(1:2:2*epochs))
xlabel('epochs')
ylabel('loss')