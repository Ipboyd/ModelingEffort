
m = x{2};

holder = [];
for z = 1:10
    holder = [holder;double(m{z})];
end
figure;
spy(holder);

