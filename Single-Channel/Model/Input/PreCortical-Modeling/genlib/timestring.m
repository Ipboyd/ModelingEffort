function varargout = timestring(t, p, showhours)
% out = timestring(t, p, showhours)
%    t - number of seconds (default = current time, or number of seconds since midnight)
%    p - number of digits after the decimal (default = 0)
%    showhours - bool for whether or not the hours place should be shown for times less than one hour (default = 0)


tocFlag = 1;
if nargin == 0 || isempty(t)
	try
		t = toc;
		tocFlag = 1;
	catch
		temp = clock;
		tocFlag = 0;
		t = 3600*temp(4) + 60*temp(5) + temp(6);
	end
end
if ~exist('p', 'var')
	p = 0;
elseif isempty(p)
	p = 0;
end
p = ceil(p);

if ~exist('showhours', 'var')
	showhours = 0;
end

t = round(t*10^p)/10^p;

h = floor(t/3600);
m = floor((t - h*3600)/60);
s = t - h*3600 - m*60;

if (h || showhours) && p
	out = sprintf(sprintf('%%02.0f:%%02.0f:%%0%0.0f.%0.0ff', 3 + p, p), h, m, s);
elseif ~(h || showhours) && p
	out = sprintf(sprintf('%%02.0f:%%0%0.0f.%0.0ff', 3 + p, p), m, s);
elseif (h || showhours) && ~p
	out = sprintf('%02.0f:%02.0f:%02.0f', h, m, s);
else
	out = sprintf('%02.0f:%02.0f', m, s);
end

if nargout == 1
	varargout{1} = out;
else
	if tocFlag
		fprintf('Elapsed time is %s\n', out)
	else
		fprintf('Current time is %s\n', out)
	end
end