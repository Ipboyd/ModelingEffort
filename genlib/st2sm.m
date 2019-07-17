function mat = st2sm(t, fs, dur)

if(nargin < 3)
    dur = max(vertcat(t{:}));
end

tSize = size(t);
outSize = [ceil(fs*dur) tSize];
mat = zeros(outSize);
for ii = 1:prod(tSize)
	mat(max(ceil(t{ii}(t{ii}<outSize(1)/fs)*fs),1),ii) = 1;
end
