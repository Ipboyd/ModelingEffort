function noticks(h)

if nargin == 0
    set(gca,'xtick',[],'ytick',[])
else
    set(h,'xtick',[],'ytick',[])
end