function datwrite(x, fn, oldfs)
% datwrite(x, fn, oldfs)
% oldfs is optional

if exist('oldfs', 'var')
	x = resample(x, 32e3, oldfs);
end
if max(abs(x)) > 9.999
	warning('Data clipped during write to\nfile:%s', fn) %#ok<WNTAG>
	x = min(9.999, max(-9.999, x));
end
if ~strcmpi(fn(end - 3:end), '.dat')
	fn = [fn '.dat'];
end
fid = fopen(fn, 'wb');
fwrite(fid, x(:), 'double');
fclose(fid);