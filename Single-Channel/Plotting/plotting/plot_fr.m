function plot_fr(datas, target, names, k)
    subplot(20,2,[29, 40])

    colors = {'r','g','b','k'};
    
    for m = 1:length(datas)
        plot(datas{m}(1,:),Tag=strcat(names{m}," Clean"),Color=colors{m}); hold on;
        plot(datas{m}(2,:),'--',Tag=strcat(names{m}," Mixed"),Color=colors{m}); hold on;
    end

    plot(target(1,:),Tag="Target Clean",LineWidth=3,Color=colors{m+1}); hold on; 
    plot(target(2,:),'--',Tag="Target Mixed",LineWidth=3,Color=colors{m+1}); hold on; 
    title("Firing Rate Comparison")

    lines = findobj(gca, 'Type', 'line');
    tags = get(lines, 'Tag');
    legend(flipud(tags));
    
end