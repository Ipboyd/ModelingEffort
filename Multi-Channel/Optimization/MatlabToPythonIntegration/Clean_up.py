import re

def Clean_gen_code(generated_code):



    # 1. Replace element-wise MATLAB operators with PyTorch-compatible syntax
    generated_code = generated_code.replace('.*', '*')
    generated_code = generated_code.replace('./', '/')
    generated_code = generated_code.replace('.^', '**')  # MATLAB power op
    generated_code = generated_code.replace('^', '**')   # fallback catch-all

    # 2. Remove all lines that use randn (not differentiable)
    randn_pattern = re.compile(r'\+[^+]*?randn\([^\)]*\)')
    generated_code = randn_pattern.sub('', generated_code)

    # 3. Replace eye(A,B) -> torch.eye(A, B)
    generated_code = re.sub(r'\beye\(([^,]+),([^)]+)\)', r'torch.eye(\1, \2)', generated_code)

    # 4. Replace ones(1,N) -> torch.ones(N) or torch.ones(1, N)
    #generated_code = re.sub(r'\bones\(\s*1\s*,\s*([^)]+)\)', r'torch.ones(1, \1)', generated_code)
    generated_code = re.sub(r'\bones\(\s*1\s*,\s*([^)]+)\)', r'torch.ones(T, \1)', generated_code)

    # 5. Replace zeros(1,N) -> torch.zeros(1,N) or torch.zeros(2, N)
    #generated_code = re.sub(r'\bzeros\(([^,]+),([^)]+)\)', r'torch.zeros(\1, \2)', generated_code)
    generated_code = re.sub(r'\bzeros\(([^,]+),([^)]+)\)', r'torch.zeros(T, \2)', generated_code)

    # 6. Replace genPoissonInputs/Times with thier python function calls
    # Replace genPoissonInputs with genPoissonInputs.gen_poisson_inputs, unless preceded by 'import '
    generated_code = re.sub(r'(?<!import\s)genPoissonInputs(?=\s*\()', 'genPoissonInputs.gen_poisson_inputs', generated_code)

    # Replace genPoissonTimes with genPoissonTimes.gen_poisson_times, unless preceded by 'import '
    generated_code = re.sub(r'(?<!import\s)genPoissonTimes(?=\s*\()', 'genPoissonTimes.gen_poisson_times', generated_code)

    # 7. Remove unused parameters (self.tspan to self.verbose_flag)
    skip_params = ['self.downsample_factor', 'self.random_seed',
                   'self.solver', 'self.disk_flag',
                   'self.datafile', 'self.mex_flag', 'self.verbose_flag']

    lines = generated_code.split('\n')

    lines = [line for line in lines if not any(param in line for param in skip_params)]

    # 8. Remove On_On_IC_label and Off_Off_IC_label lines
    #lines = [line for line in lines if 'On_On_IC_label' not in line and 'Off_Off_IC_label' not in line]

    # Reassemble the modified code
    generated_code = '\n'.join(lines)

    return generated_code
