function h=subplot3(m,n,p,dy,dx)

%   h = subplot2(m,n,p,[dy],[dx]);
% Front end for subplot: make subplots with a specified amount of space
% between them. m is number of rows (vertical stack), n is number of
% columns (horizontal stack), p is which plot this is in matrix, dy & dx
% (optional) are spacing between adject vertical & horizontal plots. dy &
% dx can also be given as 3-element lists, dy=[dyi dyB dyT] and
% dx=[dxi dxL dxR], or 5-element lists, dy=[dyi dyB dyT nb nt] and
% dx=[dxi dxL dxR nl nr] specifying spacing at bottom, top, vertically
% between subplots, and nudging to the top and bottom location of the
% subplot; left, right, and horizontally between plots, and nudging to the
% left and right location of the subplot.
% Example 1:
%   clf; for n=1:3, h1=subplot2(1,3,n,[0.04 0.3 0.35], 0.0175);
%     [x,y]=meshgrid(3:7,1:4); pcolor2(x,y,rand(size(x))*10+30)
%     set(gca,'clim',[30 40],'xtick',[],'ytick',[]); end
%   h2=subplot2(1,1,1,[0 0.18 0.8],0.3); colorbar(h2,'peer',h1)
% Example 2:
%   clf; for n=1:4, h1=subplot2(2,2,n,[0.04 0.23 0.09],0.07);
%     [x,y]=meshgrid(3:7,1:4); pcolor2(x,y,rand(size(x))*10+30)
%     set(gca,'clim',[30 40],'xtick',[],'ytick',[]); end
%   h2=subplot2(1,1,1,[0 0.1 0.87],0.3); colorbar(h2,'peer',h1)
%
% Ian Eisenman, 2007

if(nargout > 0)
    h = 0; % in case of error; otherwise, it is assigned below
end
if nargin<1, disp('subplot2(m,n,p,[dyi dyB dyT],[dxi dxL dxR]); [default: dy=0.03, dx=0.03]'), return, end
if nargin<5, dy=0.03; end
if nargin<4, dx=0.03; end

if length(dy)==3
    dyi=dy(1); dyB=dy(2); dyT=dy(3); nb=0; nt=0;
elseif length(dy)==5
    dyi=dy(1); dyB=dy(2); dyT=dy(3); nb=dy(4); nt=dy(5);
elseif length(dy)==1
    dyB=dy; dyT=dy; dyi=dy; nb=0; nt=0;
else
    disp('Error: dy should be 1-, 3-, or 5-element number'), return
end

if length(dx)==3
    dxi=dx(1); dxL=dx(2); dxR=dx(3); nl=0; nr=0;
elseif length(dx)==5
    dxi=dx(1); dxL=dx(2); dxR=dx(3); nl=dx(4); nr=dx(5);
elseif length(dx)==1
    dxL=dx; dxR=dx; dxi=dx; nl=0; nr=0;
else
    disp('Error: dy should be 1-, 3-, or 5-element number'), return
end

if dyB+dyT+(m-1)*dyi>=1
    disp('Error: not enough vertical room (dyB+dyT+(m-1)*dyi>1)'), return
end

if dxL+dxR+(n-1)*dxi>=1
    disp('Error: not enough horizontal room (dxL+dxR+(m-1)*dxi>1)'), return
end

DX=(1-dxL-dxR-dxi*(n-1))/n; % width of subplot
DY=(1-dyB-dyT-dyi*(m-1))/m; % height of subplot

nx=rem(p-1,n)+1; % horizontal number for this subplot (from left)
ny=m-fix((p-1)/n); % vertical number for this subplot (from bottom)

h0=subplot('position',[dxL+(dxi+DX)*(nx-1)+nl dyB+(dyi+DY)*(ny-1)+nb DX-nl-nr DY-nb-nt]);

if(nargout > 0)
    h = h0;
end
