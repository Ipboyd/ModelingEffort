import re

def extract_rhs_lhs(filename):
    lhs_rhs_pairs = []
    with open(filename, 'r') as f:
        for line in f:
            match = re.match(r'\s*(\w+_k1)\s*=\s*(.*?);', line)
            if match:
                lhs = match.group(1)
                rhs = match.group(2)
                lhs_rhs_pairs.append((lhs, rhs))
    return lhs_rhs_pairs

# Usage
#pairs = extract_rhs_lhs('solve_ode.m')
#for lhs, rhs in pairs:
#    print(f"LHS: {lhs} \nRHS: {rhs}\n")

