function map = setluminance(map, lumSet)

c = [0.2989, 0.5870, 0.1140];
map = max(0, min(1, bsxfun(@plus, map, lumSet - sum(bsxfun(@times, map, c), 2))));