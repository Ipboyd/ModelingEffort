% vizNetwork
% pops = {data.model.specification.populations.name};
% sz = {data.model.specification.populations.size};
%
%
% 2019-03-06 added IC->I and IC->R connections

% construct cell array of names
clear names

i = 1;
for pop = 1:length(s.populations)
    for l = 1:s.populations(pop).size
        names{i} = sprintf('%s_%i',s.populations(pop).name,l);
        i = i+1;
    end
end


%% netcons
rcNetcon = ones(s.populations(4).size,s.populations(5).size);
iciNetcon = diag(ones(1,nCells));
icsNetcon = diag(ones(1,nCells));
icrNetcon = diag(ones(1,nCells));

%% combine connectivity matrices, in the order of IC,I2,I,S,R,C
icStart = 1;
icEnd = icStart+icSize-1;
iStart = icEnd+1;
iEnd = iStart+iSize-1;
sStart = iEnd+1;
sEnd = sStart+sSize-1;
rStart = sEnd+1;
rEnd = rStart+rSize-1;
cStart = rEnd+1;
cEnd = cStart+cSize-1;


adjMtx = zeros(sum([s.populations.size]));
adjMtx(iStart:iEnd,rStart:rEnd) = irNetcon;
adjMtx(rStart:rEnd,cStart:cEnd) = rcNetcon;
adjMtx(sStart:sEnd,rStart:rEnd) = srNetcon;
adjMtx(icStart:icEnd,sStart:sEnd) = icsNetcon;
adjMtx(icStart:icEnd,rStart:rEnd) = icrNetcon;
adjMtx(icStart:icEnd,iStart:iEnd) = iciNetcon;
%% graph
g = digraph(adjMtx,names);
figure; p = plot(g,'layout','layered','Marker','o',...
                   'MarkerSize',8,'NodeLabel',names);
set(p,'ArrowSize',12)
layout(p,'layered','direction','down',...
         'sources',[cStart:cEnd],'sinks',[icStart:icEnd]);
