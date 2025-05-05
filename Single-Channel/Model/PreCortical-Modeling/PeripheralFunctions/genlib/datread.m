function out = datread(fn)

if ~any(fn == '.')
	fn = [fn '.dat'];
end
fid = fopen(fn, 'rb');
if fid
	out = fread(fid, Inf, 'float64');
	fclose(fid);
else
	out = [];
end