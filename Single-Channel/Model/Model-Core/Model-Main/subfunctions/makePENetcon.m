function PEnetcon = makePENetcon(bestLocs,sigma)

% Returns PEnetcon, the connectivity matrix from PV neurons to excitatory
% cells in the same layer

% For now, let's model the connectivity between PV and E as a set of Gaussians

% Other studies have found that PV neurons show broader tonal tuning than
% excitatory cells (Li et al. Cereb Cortex 2015), so someone could
% implement this in the spatial domain (Kyweriga et al. J Neurophysiol,
% 2014)

% Inputs:
% bestLocs - the best location for each input channel
% sigma - width of Gaussian curve
%   as sigma approaches 0, PEnetcon becomes an identity matrix

% nCells = numel(bestLocs);
% 
% azi = -108:108;
% 
% azi_inds = find(ismember(azi,bestLocs));

PEnetcon = eye(1);
% for i = 1:nCells
%     curve = gaussmf(azi,[sigma bestLocs(i)]);
%     PEnetcon(i,:) = curve(azi_inds);
% end

end