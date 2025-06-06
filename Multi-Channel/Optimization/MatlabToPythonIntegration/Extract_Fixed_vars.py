import re

def extract_fixed_variables_from_block(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    in_update_block = False
    fixed_vars = []

    # Variables to skip (e.g. input-related that you don't want to include)
    skip_params = ['self.Off_Off_IC_input', 'self.On_On_IC_input']

    # Pattern to extract simple assignments: var = expression;
    pattern = re.compile(r'^\s*(\w+)\s*=\s*(.+?);')

    for line in lines:
        if '% Fixed variables:' in line:
            in_update_block = True
            continue
        if in_update_block and '% Initial conditions:' in line:
            break
        if in_update_block:
            # Skip lines containing any of the skip_params
            if any(param in line for param in skip_params):
                continue
            match = pattern.match(line)
            if match:
                lhs = match.group(1)
                rhs = match.group(2)
                fixed_vars.append((lhs, rhs))

    return fixed_vars
