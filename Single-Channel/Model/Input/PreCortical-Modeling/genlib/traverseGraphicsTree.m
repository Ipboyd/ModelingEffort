function traverseGraphicsTree(parent, condition, props)

children = get(parent, 'Children');
try
	if condition(1) == '%'
		doIt = eval(regexprep(condition(2:end), 'hObject', 'parent'));
	elseif strcmpi(get(parent, 'Type'), condition)
		doIt = true;
	else
		doIt = false;
	end
catch ERR
	disp(ERR.message)
	if ~isempty(children)
		disp('Dropping down to child level.')
	end
	doIt = 0;
end
if doIt
    for ii = 1:length(props)
        if ischar(props{ii})
            if props{ii}(1) == '%'
				props{ii} = eval(regexprep(props{ii}(2:end), 'hObject', 'parent'));
            end
        end
	end
	try
		set(parent, props{:})
	catch ERR
		disp(ERR.message)
	end
end
for ii = 1:length(children)
    traverseGraphicsTree(children(ii), condition, props)
end