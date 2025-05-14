import re

def remove_discrete_time_indexing(rhs_expr):
    """
    Replace MATLAB-style discrete indexing like x(n-1,:) with x
    """
    # Match any var name followed by (n...), like x(n,:), x(n-1,:), etc.
    pattern = re.compile(r'(\b\w+)\(n[^\)]*\)')
    return pattern.sub(r'\1', rhs_expr)
