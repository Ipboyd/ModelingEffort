

empty_struct = struct('animal_names',{all_data.subject},'n90','','p0','','p45','','p90','');

empty_struct(1).n90 = struct('is_Onset','','is_Offset','','is_Both','','Onset_Data','','Offset_Data','');



%empty_struct = struct('animal_names',{all_data.subject},'is_Onset','','is_Offset','','is_Both','','Onset_Data','','Offset_Data','');


%empty_struct(1).is_Onset{1} = 1
%empty_struct(1).is_Onset{2} = 0
