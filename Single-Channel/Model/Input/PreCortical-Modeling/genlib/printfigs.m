function printfigs(handles)

check = '1';
pcmd = printopt;
if isempty(findstr(pcmd, 'PDFCreator'))
	check = input(sprintf('The printer is currently set to:\n\n%s\n\nDo you wish to proceed? ', pcmd), 's');
end

if strcmpi(check, '1') || strcmpi(check, 'y') || strcmpi(check, 'yes')
	if ~exist('handles', 'var') || isempty(handles) || isequal(handles, 0)
		handles = sort(get(0, 'Children'));
	end

	for ii = 1:length(handles)
		print(handles(ii));
		pause(2.0)
	end
else
	disp('Printing cancelled.')
end