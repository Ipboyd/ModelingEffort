import re

def reformat_discrete_time_indexing(rhs_expr):
    """
    Replace MATLAB-style discrete indexing like x(n-1,:) with x
    """
    # Match any var name followed by (n...), like x(n,:), x(n-1,:), etc.
    pattern = re.compile(r'(\b\w+)\(n[^\)]*\)')
    return pattern.sub(r'\1[t-1]', rhs_expr)

def reformat_input_time_indexing(rhs_expr):
    """
    Replace MATLAB-style discrete indexing like x(k,:) with x
    """
    # Match any var name followed by (n...), like x(n,:), x(n-1,:), etc.
    pattern = re.compile(r'(\b\w+)\(k[^\)]*\)')
    return pattern.sub(r'\1[t]', rhs_expr)
