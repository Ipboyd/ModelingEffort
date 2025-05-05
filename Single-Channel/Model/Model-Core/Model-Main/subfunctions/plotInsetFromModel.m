% control: 5, laser: 2

for f = 1:2
    figure(f);
    a = findobj(gcf,'type','axes');
    for i = 1:length(a)
        if strcmp(a(i).Title.String{1},'R2On')
            l = findobj(a(i),'type','line');
            for j = 1:2
                x = l(j).XData;
                y(f,j,:) = l(j).YData;
            end
        end
    end
end

x = x/10000;

inds = x >= 0.1+0.3 & x <= 0.4+0.3;
figure; plot(x(inds)-0.3,squeeze(y(1,1,inds)),'k','linewidth',1);
hold on; plot(x(inds)-0.3,squeeze(y(2,1,inds)),'r','linewidth',1);
xlabel('Time (s)'); ylabel('Spike count');