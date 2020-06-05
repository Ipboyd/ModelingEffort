% data from Panniello et al., 2018; fig 3C
x = [-90 -61 -10 0 18 61 90];
y1 = [18 20 23 40 55 60 100]; %ipsi preferred
y2 = [30 40 50 100 45 25 20]; %center preferred
y3 = [100 65 50 35 25 18 20]; %contra preferred

cb = [56,83,163]/255;
cg = [105,189,69]/255;
cr = [238,31,35]/255;

% define functions to use
f1 = @(b,x) b(1).*exp(b(2).*(x-b(3)))+b(4); % exponential Function
s = @(b,x) b(1)./(1+exp(b(2)*x+b(3)))+b(4); % sigmoidal function
g = @(b,x) b(4)*exp(-(x-b(1)).^2/(2*b(2).^2))+b(3); % gaussian
u = @(b,x) -b(4)*exp(-(x-b(1)).^2/(2*b(2).^2))+b(3); % gaussian

% 1 - ipsi-preferred neurons
options = optimset('MaxFunEvals',50);
B1 = fminsearch(@(b) norm(y1 - f1(b,x)), [1; -1; 0; 15]); % Estimate Parameters

figure
plot(x, y1, 'o','color',cr,'markersize',8,'markerfacecolor',cr); hold on
plot(x, f1(B1,x), '-','color',cr,'linewidth',2);

% 2 - center-preferred neurons
options = optimset('MaxFunEvals',500);
B3 = fminsearch(@(b) norm(y2(1:4) - f1(b,x(1:4))), [100; 0.25; 0; 20],options); % piece-wise
B4 = fminsearch(@(b) norm(y2(4:end) - f1(b,x(4:end))), [5; -0.25; 60; 20],options); % exponentials
B5 = fminsearch(@(b) norm(y2 - g(b,x)), [0; 20; 90; 100],options); % try gaussian fit

plot(x,y2,'o','color',cg,'markersize',8,'markerfacecolor',cg);
plot(x(1:4),f1(B3,x(1:4)), '-','color',cg,'linewidth',2);
plot(x(4:end),f1(B4,x(4:end)), '-','color',cg,'linewidth',2);
plot(x,g(B5,x), '--','color',cg-0.25,'linewidth',2);

% 3 - contra-preferred neurons
B2 = fminsearch(@(b) norm(y3 - f1(b,x)), [1; 0.5; 0; 15]); % Estimate Parameters

plot(x, y3, 'o','color',cb,'markersize',8,'markerfacecolor',cb); hold on
plot(x, f1(B2,x), '-','color',cb,'linewidth',2);
xlabel('\leftarrow contralateral side; azimuths (degrees).')
ylabel('f(x)')

grid;
xlabel('\leftarrow contralateral side; azimuths (degrees).')
ylabel('normalized responses (%\DeltaF/F)')
title('fit to Panniello et al., 2018')

%% data from Ono and Oliver 2014
x = [-90 -61 -10 0 18 61 90];
y1I = [95 85 85 80 65 50 48]; % fig 4B
y1E = [78 79 48 50 48 35 35];
y2I = [90 47 42 10 20 23 22]; % fig 4C
y2E = [57 50 45 20 18 25 35];
y3I = [95 72 75 50 40 20 10]; % fig 4D
y3E = [40 62 68 85 88 40 18];

figure;
subplot(1,3,1);
plot(x,y1I,'^-','color',cb,'markersize',8,'markerfacecolor',cb); hold on;
plot(x,y1E,'^-','color',cr,'markersize',8,'markerfacecolor',cr)
title('sigmoidal fits; contra preferred')
subplot(1,3,2);
plot(x,y2I,'v-','color',cb,'markersize',8,'markerfacecolor',cb); hold on;
plot(x,y2E,'v-','color',cr,'markersize',8,'markerfacecolor',cr)
title('u fits; still contra preferred')
subplot(1,3,3);
plot(x,y3I,'v-','color',cb,'markersize',8,'markerfacecolor',cb); hold on;
plot(x,y3E,'v-','color',cr,'markersize',8,'markerfacecolor',cr)
title('gaussian fits;')

subplot(1,3,1);
By1I = fminsearch(@(b) norm(y1I - s(b,x)), [50; 1; 1; 50]);
plot(x, s(By1I,x), '-','color',cb,'linewidth',2);
By1E = fminsearch(@(b) norm(y1E - s(b,x)), [50; 1; 1; 50]);
plot(x, s(By1E,x), '-','color',cr,'linewidth',2);
legend({'data','','fit',''})

subplot(1,3,2);
By2I = fminsearch(@(b) norm(y2I - u(b,x)), [0; 20; 90; 100]); 
plot(x, u(By2I,x), '-','color',cb,'linewidth',2);
By2E = fminsearch(@(b) norm(y2E - u(b,x)), [0; 20; 90; 100]);
plot(x, u(By2E,x), '-','color',cr,'linewidth',2);
legend({'data','','fit',''});

subplot(1,3,3);
By3I = fminsearch(@(b) norm(y3I - s(b,x)), [0; 20; 90; 100]); 
plot(x, s(By3I,x), '-','color',cb,'linewidth',2);
By3E = fminsearch(@(b) norm(y3E - g(b,x)), [0; 20; 90; 100]);
plot(x, g(By3E,x), '-','color',cr,'linewidth',2);
legend({'data','','fit',''});
xlabel('azimuths')