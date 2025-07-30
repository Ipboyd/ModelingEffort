vr_ex = x{end};


vals = [];
for k = 1:length(vr_ex)
    %print(float(a{k}))
    vals = [vals, double(vr_ex{k})];
end
figure
plot(vals)