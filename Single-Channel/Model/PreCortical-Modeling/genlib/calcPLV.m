function y = calcPLV(phase)
% Calculates the PLV along the rows of PHASES.

cosPhase = cos(phase);
sinPhase = sin(phase);
y = sqrt(mean(cosPhase,2).^2+mean(sinPhase,2).^2);
