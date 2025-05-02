function rgb = hex2rgb(hex)

if ~iscell(hex)
	hex = {hex};
end

for ii = 1:length(hex)
	if lower(hex{ii}(1)) == 'x' || hex{ii}(1) == '#'
		hex{ii} = hex{ii}(2:end);
	elseif lower(hex{ii}(2)) == 'x'
		hex{ii} = hex{ii}(3:end);
	end
	
	dec = hex2dec(hex{ii});
	rgb(ii,1) = floor(dec/16^4);
	rgb(ii,2) = floor(dec/16^2) - rgb(ii,1)*16^2;
	rgb(ii,3) = floor(dec) - sum(rgb(ii,1:2).*[16^4 16^2]);
	rgb(ii,:) = rgb(ii,:)/255;
end
