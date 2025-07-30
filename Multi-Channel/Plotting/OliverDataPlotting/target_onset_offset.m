
clear
close all

load('/Users/lbowman/Desktop/Research/sound_files.mat','sampleRate','target1','target2');

% figure;
% plot(target1)

t1 = abs(target1);

% figure;
% plot(t1)

st1 = smoothdata(t1,'gaussian',20000);

x = max(t1);
y = max(st1);

fac = x/y;
st1 = st1 .* fac;

figure;
plot(st1)

figure;
hold on
plot(t1)
plot(st1)
hold off
% 
% tx = 0:(length(st1)-1);
% tx = tx';

% figure;
% plot(tx,st1)

dt1 = diff(st1);
sdt1 = smoothdata(dt1,"movmedian",500);

figure;
plot(sdt1)
yline(0)

onoff = sdt1 > 0;
fonoff = onoff .* max(st1);

figure;
hold on
plot(fonoff,'r')
% plot(t1,'b')
hold off

% counts = [];
% 
% for i = 1:length(sdt1)
% 
%     if dt1(i) > 0
%         if dt1(i+500) > 0
%             counts(i:(i+5000),1) = 1;
%             i = i+1000;
%         end
%     else 
%         counts(i,1) = 0;
%     end
% end
% 
% figure;
% plot(counts)
% ylim([-0.5 1.5])


onidx = find(onoff);
off = 1 - onoff;
offidx = find(off);

figure;
plot(onidx)