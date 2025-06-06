import re

import re
from collections import defaultdict

def extract_state_variables_from_block(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    in_block = False
    var_assignments = defaultdict(dict)

    # Match lines like: R2On_V(1,:) = R2On_E_L*ones(1,R2On_Npop);
    pattern = re.compile(r'^\s*(\w+)\((\d+),:?\)\s*=\s*(.+?);')

    

    for line in lines:
        if '% STATE_VARIABLES:' in line:
            in_block = True
            continue
        if in_block and '% MONITORS:' in line:
            break
        if in_block:
            match = pattern.match(line)
            if match:
                var_name = match.group(1)
                row_index = int(match.group(2)) - 1  # MATLAB is 1-indexed
                rhs_expr = match.group(3).strip()
                rhs_expr = rhs_expr.replace('ones(1,', 'ones(T,')
                var_assignments[var_name][row_index] = rhs_expr

    

    return var_assignments


def extract_state_update(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    in_block = False
    var_assignments = defaultdict(dict)

    # Match update block lines
    pattern = re.compile(r'^\s*(\w+)\(n,:?\)\s*=\s*(.+?);')

    for line in lines:
        if '% Update state variables:' in line:
            in_block = True
            continue
        if in_block and '% Conditional actions:' in line:
            break
        if in_block:
            match = pattern.match(line)
            if match:
                var_name = match.group(1)
                #lhs_var = match.group(1)
                rhs_expr = match.group(2).strip()

                var_name = var_name + '[t]'
                rhs_expr = re.sub(r'\(n-1,:?\)', '[t-1]', rhs_expr)

                var_assignments[var_name]['rhs'] = rhs_expr  # no index anymore

    return var_assignments

def extract_monitor_declarations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    in_block = False
    monitor_assignments = defaultdict(dict)

    # Matches: var = rhs;
    pattern = re.compile(r'^\s*(\w+)\s*=\s*(.+?);')

    for line in lines:
        if '% MONITORS:' in line:
            in_block = True
            continue
        if in_block and (line.strip().startswith('%') or line.strip() == ''):
            break
        if in_block:
            match = pattern.match(line)
            if match:
                var_name = match.group(1)
                rhs_expr = match.group(2).strip()
                #Will have to change likely for multichannel model and will need to use any(in test2b)
                #rhs_expr = rhs_expr.replace('ones(5,', 'ones(1,')
                monitor_assignments[var_name]['rhs'] = rhs_expr

    return monitor_assignments


