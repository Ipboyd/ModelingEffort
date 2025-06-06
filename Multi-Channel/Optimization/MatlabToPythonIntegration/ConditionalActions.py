import re

def extract_conditional_variables(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Match only lines assigning to a variable ending in "_test"
    pattern = re.compile(r'^\s*\w+_test\s*=\s*.*t\s*==\s*(\w+)\s*\+\s*(\w+)')
    conditional_vars = []

    for line in lines:
        match = pattern.search(line)
        if match:
            var1 = match.group(1)
            var2 = match.group(2)
            conditional_vars.append((var1, var2))

    return conditional_vars