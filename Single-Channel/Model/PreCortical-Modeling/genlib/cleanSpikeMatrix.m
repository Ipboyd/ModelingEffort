function varargout = cleanSpikeMatrix(spikeMatrix,Ntpsout,confs,reRun)
% function [OUTMATRIX,CONFS] = CLEANSPIKEMATRIX(SPIKEMATRIX,NTPSOUT,CONFS)
%   Function to isolate movement artifacts by eye. It will eliminate as
%   many bad trails as possible while leaving NTPSOUT number of trials per
%   song to be outputted.
%
%   Clicking trials will cycle through five levels of confidences:
%     #) Color  - Movement
%     --------------------
%     1) Black  - none (default)
%     2) Blue   - possible
%     3) Green  - likely
%     4) Yellow - likelier
%     5) Red    - certain
%
% 2009/05/21 Eric Larson edlarson@bu.edu
% 

Ntpb = 50; % # of trials per block
useSpikeTimes = false;
outDims = size(spikeMatrix);
if(~iscell(spikeMatrix))
    Ns = outDims(1);
    outDims = outDims(2:end);
else
    useSpikeTimes = true;
end

if(nargin < 2 || isempty(Ntpsout))
    Ntpsout = outDims(1);
end
if(nargin < 4 || isempty(reRun))
    reRun = true;
end
if(Ntpsout > outDims(1))
    error('Ntpsout > Ntrials');
end
Ntot = prod(outDims);
mycolors = 'kbgyr';

% If no confidences matrix is inputted, generate confidences
% Or if it requests confidences out, re-do confidences
if(nargin < 3 || nargout > 1)
    Nblocks = ceil(Ntot/Ntpb);
    blims = unique([0 Ntpb:Ntpb:Ntot Ntot]);
    if(nargin < 3)
        confs = zeros(outDims);
    end
    for bi = 1:Nblocks
        Ntinds = blims(bi+1)-blims(bi);
        tinds = blims(bi)+1:blims(bi+1);
        if(useSpikeTimes)
            smallMatrix = spikeMatrix(tinds);
        else
            smallMatrix = spikeMatrix(:,tinds);
        end

        if(reRun)
            cla;
            h = rasterplot(smallMatrix);
            for ti = 1:Ntinds
                if(h(ti))
                    set(h(ti),'Color',mycolors(confs(tinds(ti))+1),'UserData',confs(tinds(ti)),'ButtonDownFcn',@button_down0);
                end
            end
            title(sprintf('Block %d of %d',bi,Nblocks));
            drawnow;

            % Let the user click to change some colors, store those values
            pause
            for ti = 1:Ntinds
                if(h(ti))
                    confs(tinds(ti)) = get(h(ti),'UserData');
                end
            end
        end
    end
end

temp = size(spikeMatrix);
if(~useSpikeTimes)
    temp = temp(2:end);
end
if(~isequal(temp,size(confs)))
    keyboard
    error('CONFS size must match the non-time dimensions of SPIKEMATRIX');
end

if(useSpikeTimes)
    outMatrix = cell(outDims);
else
    outMatrix = zeros([Ns outDims]);
end
for ci = 1:prod(outDims(2:end))
    [junk,indord] = sort(confs(:,ci),'ascend');
    if(useSpikeTimes)
        outMatrix(1:Ntpsout,ci) = spikeMatrix(indord(1:Ntpsout),ci);
    else
        outMatrix(1:Ns,1:Ntpsout,ci) = spikeMatrix(:,indord(1:Ntpsout),ci);
    end
end

if(nargout)
    varargout{1} = outMatrix;
    varargout{2} = confs;
end


function button_down0(src,evnt) %#ok<INUSD>
mycolors = 'kbgyr';
nextConf = mod(get(src,'UserData')+1,5);
set(src,'Color',mycolors(nextConf+1),'UserData',nextConf);
