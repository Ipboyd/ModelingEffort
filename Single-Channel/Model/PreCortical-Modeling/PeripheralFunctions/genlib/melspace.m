function out = melspace(a,b,n)
%MELSPACE(a,b,n)
%  Creates n frequencies between a and b equally spaced on the mel scale.

out = mel2freq(linspace(freq2mel(a),freq2mel(b),n));