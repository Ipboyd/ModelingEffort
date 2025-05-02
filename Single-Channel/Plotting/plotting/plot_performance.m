function plot_performance(bg,bfg,name,k)
        
        %TODOS -- Add labels

        %set_start = ((ceil(k/2)-1)*14) + abs(mod(k,2)-2) + 2;
        %number_set = set_start+(0:4)*2;

        %h = subplot(20,2,number_set(2:end));
        h = subplot(2,1,2);
        plotPerfGrid(flipud(bg(2:5,:)), flipud(bfg(2:5,:)), []); hold on

        xticks(1:4);
        xticklabels({'90','45','0','-90'});
        yticks(1:4);
        yticklabels(fliplr({'90','45','0','-90'}));
        xlabel('target location');
        ylabel('masker location');

        %pos = get(h,'Position');
        %pos(4) = pos(4) - 0.12;
        %pos(2) = pos(2) + 0.18;

        pos = get(h,'Position');
        pos(4) = pos(4) - 0.25;
        pos(2) = pos(2) + 0.35;

        subplot('Position',pos)
        plotPerfGrid(bg(1,:), bfg(1,:), []);
        title(name)

        %yticks(1:1);
        %yticklabels('0');
        ylabel('Clean');
        

end