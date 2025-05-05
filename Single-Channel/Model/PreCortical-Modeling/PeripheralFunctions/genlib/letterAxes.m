function [hax txtax] =letterAxes(labelAxes, varargin)
%letterAxes(labelAxes, leftshift, upshift, varargin)
%labelAxes - vector of axis to use
%leftshift - OPTIONAL, left shift (in inches) from top left corner of axis, can be
%           scalar (gets applied to each axis) or a vector that of the same
%           length as the number of axis.  Default to 0.3.
%upshift - OPTIONAL, up shift from top left corner of axis, can be
%           scalar (gets applied to each axis) or a vector that of the same
%           length as the number of axis.  Default to 0.05.
%varargin - arguments that get passed to the text function, eg fontsize
%
%Output
%hax - new set of axis that the text is plotted on
%txtax - vector of handles to the text objects
numaxis = length(labelAxes);
nvar = length(varargin);
txtflag = 0;
if (nvar > 2)
    txtflag=1;
    textchar = varargin(3:end);
    if isempty(varargin{2})
        upshift = repmat(0.05, numaxis, 1);
    else
        if isscalar(varargin{2})
            upshift = repmat(varargin{2}, numaxis, 1);
        else
            upshift = varargin{2};
        end
    end
    
    if isempty(varargin{1})
        leftshift = repmat(0.3, numaxis, 1);
    else
        if isscalar(varargin{1})
            leftshift = repmat(varargin{1}, numaxis, 1);
        else
            leftshift = varargin{1};
        end
    end
    
elseif (nvar == 2)
    if isempty(varargin{2})
        upshift = repmat(0.05, numaxis, 1);
    else
        if isscalar(varargin{2})
            upshift = repmat(varargin{2}, numaxis, 1);
        else
            upshift = varargin{2};
        end
    end
    
    if isempty(varargin{1})
        leftshift = repmat(0.3, numaxis, 1);
    else
        if isscalar(varargin{1})
            leftshift = repmat(varargin{1}, numaxis, 1);
        else
            leftshift = varargin{1};
        end
    end
    
elseif (nvar == 1)
        upshift = repmat(0.05, numaxis, 1);

    if isempty(varargin{1})
        leftshift = repmat(0.3, numaxis, 1);
    else
        if isscalar(varargin{1})
            leftshift = repmat(varargin{1}, numaxis, 1);
        else
            leftshift = varargin{1};
        end
    end
    
else %no optional arguments
   upshift = repmat(0.05, numaxis, 1);
   leftshift = repmat(0.3, numaxis, 1);
end
        
hax = axes('position', [0 0 1 1]);
axis off; box off;
uns1 = get(gcf, 'Units');
set(gcf, 'Units', 'Inches');
posit = get(gcf, 'Position');
figSize = posit(3:4);
txtax = zeros(numaxis,1);
for ii = 1:numaxis
    uns2 = get(labelAxes(ii), 'Units');
    set(labelAxes(ii),'Units','Inches');
    pos = get(labelAxes(ii), 'position');
    if txtflag
        txtax(ii) = text((pos(1)-leftshift(ii))/figSize(1),(pos(2)+pos(4)+upshift(ii))/figSize(2), upper(alphanum(ii)),'Fontweight','Bold','VerticalAlignment','Top','HorizontalAlignment','Right','FontSize',9,'FontWeight','Bold', textchar{:});
    else
        txtax(ii) = text((pos(1)-leftshift(ii))/figSize(1),(pos(2)+pos(4)+upshift(ii))/figSize(2), upper(alphanum(ii)),'Fontweight','Bold','VerticalAlignment','Top','HorizontalAlignment','Right','FontSize',9,'FontWeight','Bold');
    end
    set(labelAxes(ii),'Units', uns2);
end
set(gcf, 'Units', uns1);