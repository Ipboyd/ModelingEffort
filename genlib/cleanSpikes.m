function [confs,outSpikes] = cleanSpikes(spikes,Ntpsout,confs,showStats)
% function CONFS = CLEANSPIKES(SPIKES)
%   Function to isolate movement artifacts by eye. It will eliminate as
%   many bad trails as possible while leaving NTPSOUT number of trials per
%   song to be outputted.
%
%   Clicking trials will cycle through five levels of error confidences:
%     #) Color  - Movement
%     --------------------
%     1) Black  - no movement artifacts (default)
%     2) Blue   - possible "   "
%     3) Green  - likely   "   "
%     4) Yellow - likelier "   "
%     5) Red    - certain  "   "
%   Use the spacebar to advance to the next group of spikes.
%
%   [CONFS,OUTSPIKES] = CLEANSPIKES(NTPSOUT,SPIKES)
% 2009/05/21 Eric Larson edlarson@bu.edu
% 

if(nargin < 1)
    error('Need one argument.');
end
usingTimes = iscell(spikes);
if(nargin < 2 && nargout >=2)
    warning('Requested output spikes with no limit (second argument) to number of trials returned, Output will be the same as the input.'); %#ok<WNTAG>
end
if(nargin < 4)
    showStats = false;
end
Ntpb = 50; % # of trials per block
if(usingTimes)
    [Ntrials Nsongs Nfis] = size(spikes);
else
    [Ns Ntrials Nsongs Nfis] = size(spikes);
end
if(nargin < 2)
    Ntpsout = Ntrials;
end
if(Ntpsout > Ntrials)
    error('Ntpsout > Ntrials');
end
Ntot = Nsongs*Ntrials*Nfis;
mycolors = 'kbgyr';

% If no confidences matrix is inputted, generate confidences
% Or if it requests confidences out, re-do confidences
if(nargin < 3 || nargout > 1)
    Nblocks = ceil(Ntot/Ntpb);
    blims = unique([0 Ntpb:Ntpb:Ntot Ntot]);
    if(nargin < 3)
        confs = zeros(Ntrials,Nsongs,Nfis);
    end
    for bi = 1:Nblocks
        Ntinds = blims(bi+1)-blims(bi);
        tinds = blims(bi)+1:blims(bi+1);
        if(usingTimes)
            smallMatrix = spikes(tinds);
        else
            smallMatrix = spikes(:,tinds);
        end
        cla;
        h = rasterplot(smallMatrix);
        for ti = 1:Ntinds
            set(h(ti),'UserData',confs(tinds(ti)),'ButtonDownFcn',@button_down0,'color',mycolors(confs(tinds(ti))+1));
        end
        title(sprintf('Block %d of %d',bi,Nblocks));
        drawnow;

        % Let the user click to change some colors, store those values
        pause
        if(Ntinds > 1)
            confs(tinds) = cell2mat(get(h,'UserData'));
        else
            confs(tinds) = get(h,'UserData');
        end
    end
end

if(nargout >= 2)
    temp = [Ntrials Nsongs Nfis];
    [temp1 temp2 temp3] = size(confs);
    if(~isequal(temp,[temp1 temp2 temp3]))
        error('CONFS size must match the non-time dimensions of SPIKEMATRIX');
    end

    if(usingTimes)
        outSpikes = cell(Ntpsout,Nsongs,Nfis);
    else
        outSpikes = zeros(Ns,Ntpsout,Nsongs,Nfis);
    end
    for fi = 1:Nfis
        for ci = 1:Nsongs
            [junk,indord] = sort(confs(:,ci,fi),'ascend'); %#ok<ASGLU>
            if(usingTimes)
                outSpikes(1:Ntpsout,ci,fi) = spikes(indord(1:Ntpsout),ci,fi);
            else
                outSpikes(:,1:Ntpsout,ci,fi) = spikes(:,indord(1:Ntpsout),ci,fi);
            end
            if(showStats)
                temp = hist(confs(indord(1:Ntpsout),ci,fi),0:4);
                fprintf('  Song %02.0f inclusions: (%02.0f,%02.0f,%02.0f,%02.0f,%02.0f)\n',ci,temp);
            end
        end
    end
end


function button_down0(src,evnt) %#ok<INUSD>
mycolors = 'kbgyr';
nextConf = mod(get(src,'UserData')+1,5);
set(src,'Color',mycolors(nextConf+1),'UserData',nextConf);
