function breakAx = breakaxis(loc,ax,squigAmp,squigLen,gapWidth)
% function breakAx = breakaxis(loc,ax,squigAmp,squigLen,gapWidth)
%   BREAKAX = BREAKAXIS(LOC) inserts break marks on the X-axis at the 
%   specified location LOC using a new set of axes BREAKAX that is labeled 
%   with the tag 'Axis Break'.
%   
%   BREAXAX = BREAKAXIS(LOC,AX,SQUIGAMP,SQUIGLEN,GAPWIDTH) inserts 
%   break marks on the coordinate axis specified by AX ('x' or 'y') at the 
%   specified location LOC---with squiggles of amplitude SQUIGAMP, length
%   SQUIGLEN, and a gap of GAPWIDTH between them---using a new set of 
%   axes BREAKAX that is labeled with the tag 'Axis Break'. Each input
%   parameter after LOC is optional, taking on default values when the
%   appropriate input values are empty [].
%
% REQUIRES: axescoord2figurecoord (MATLAB File Exchange)
% 
% Created: Eric Larson 06/26/08
% 

if nargin < 1 || ~isnumeric(loc)
    error('breakaxis requires a valid break location as its first input')
else
    if nargin < 2 || isempty(ax)
        ax = 'x';
    end
    if nargin < 3 || isempty(squigAmp)
        squigAmp = 0.005; % Width of each squiggle
    end
    if nargin < 4 || isempty(squigLen)
        squigLen = 0.05; % Proportional default
    end
    if nargin < 5
        gapWidth = 0.025;    % Proportional default
    end
end

% Check to make sure we have axes to add a break to
fig = get(0,'CurrentFigure');
if(isempty(fig))
    error('Must have a figure and axes already plotted to use breakaxis');
else
    oldAx = get(fig,'CurrentAxes');
    if(isempty(oldAx))
        error('Must have a figure a set of axes already plotted to use breakaxis');
    end
end

oldUnits = get(gcf,'Units');
set(gcf,'Units','normalized');
oldPos = get(oldAx,'Position');

% Calculate squiggles
sinVec = 0:0.05:1;
squigY1 = sinVec*squigLen;
squigY2 = squigY1;
squigX1 = (squigAmp/2)*sin(2*pi*sinVec);
squigX2 = squigX1 + gapWidth;
clear sinVec;

% Calculate dimensions, taking into account x- or y-ness
smallAxLims = [-squigAmp/2 gapWidth+squigAmp/2 0 squigLen];
axWidth = squigAmp + squigLen;
axHeight = squigLen;
if(ax(1) == 'y')
    axPosOrd = [2 1 4 3];
    axLimOrd = [3 4 1 2];
    squigY1 = squigX1;
    squigX1 = squigY2;
    squigY2 = squigX2;
    squigX2 = squigX1;
    [ytemp,xtemp] = axescoord2figurecoord(0,loc);
    ytemp = oldPos(1);
else
    axPosOrd = [1 2 3 4];
    axLimOrd = [1 2 3 4];
    [xtemp,ytemp] = axescoord2figurecoord(loc,0);
    ytemp = oldPos(2);
end
axX = xtemp - axWidth/2;
axY = ytemp - axHeight/2;
axPos = [axX axY axWidth axHeight];
clear xtemp ytemp;

% Place axes over current axes
breakAx = axes('Position',axPos(axPosOrd),'Tag','Axis Break');

% Plot break on break axes
fill([squigX1 fliplr(squigX2)],[squigY1 fliplr(squigY2)],'w','LineStyle','none'); hold on;
plot(squigX1,squigY1,'k','LineWidth',get(oldAx,'LineWidth'),'Clipping','Off');
plot(squigX2,squigY2,'k','LineWidth',get(oldAx,'LineWidth'),'Clipping','Off'); hold off; axis off; axis(smallAxLims(axLimOrd));
set(breakAx,'Clipping','Off');
set(gca,'DataAspectRatio',[1 1 1]);

% Return control to normal axes
set(fig,'CurrentAxes',oldAx,'Units',oldUnits);
