import re

def extract_fixed_variables_from_block(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    in_update_block = False
    fixed_vars = []

    # Pattern to extract LHS and RHS (e.g., On_R = 1/On_g_L;)
    pattern = re.compile(r'^\s*(\w+)\s*=\s*(.+?);')

    for line in lines:
        if '% Fixed variables:' in line:
            in_update_block = True
            continue
        if in_update_block and '% Initial conditions:' in line:
            break
        if in_update_block:
            match = pattern.match(line)
            if match:
                lhs = match.group(1)
                rhs = match.group(2)

                fixed_vars.append((lhs, rhs))

    return fixed_vars

