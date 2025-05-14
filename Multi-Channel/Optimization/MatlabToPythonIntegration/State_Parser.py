import re

def extract_state_variables_from_block(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    in_update_block = False
    state_vars = []

    # Pattern to extract LHS variable (e.g., On_V from On_V(n,:) = ...)
    pattern = re.compile(r'^\s*(\w+)\(n[^\)]*\)\s*=')

    for line in lines:
        if '% Update state variables:' in line:
            in_update_block = True
            continue
        if in_update_block and '% Conditional actions:' in line:
            break
        if in_update_block:
            match = pattern.match(line)
            if match:
                var_name = match.group(1)
                state_vars.append(var_name)

    return state_vars
