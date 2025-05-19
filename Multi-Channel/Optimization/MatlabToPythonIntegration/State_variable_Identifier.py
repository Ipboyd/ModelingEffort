import re

def add_self_prefix(rhs_expr, state_vars):
    """
    Add `self.` prefix to known state variables and parameters in a RHS expression.
    
    Parameters:
    - rhs_expr: the right-hand side string (e.g., 'On_V + On_g_ad * ...')
    - state_vars: list of variable names to be prefixed with `self.`
    """
    for var in sorted(state_vars, key=len, reverse=True):  # longest vars first to avoid partial matches
        pattern = r'\b' + re.escape(var) + r'\b'
        rhs_expr = re.sub(pattern, f'self.{var}', rhs_expr)
    return rhs_expr

def add_model_prefix(rhs_expr, state_vars):
    """
    Add `self.` prefix to known state variables and parameters in a RHS expression.
    
    Parameters:
    - rhs_expr: the right-hand side string (e.g., 'On_V + On_g_ad * ...')
    - state_vars: list of variable names to be prefixed with `self.`
    """
    for var in sorted(state_vars, key=len, reverse=True):  # longest vars first to avoid partial matches
        pattern = r'\b' + re.escape(var) + r'\b'
        rhs_expr = re.sub(pattern, f'model.{var}', rhs_expr)
    return rhs_expr


def Find_Var(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for count,line in enumerate(lines):
        if '% Update state variables:' in line:
            to_state = count
            break

    for count,line in enumerate(lines):
            if 'R2On_V(n,:) =' in line:
                break

    print(count)
    print(to_state)

    return count - to_state - 1 - 1 #Minus 1 for formatting and minus 1 for matlab to python indexing
