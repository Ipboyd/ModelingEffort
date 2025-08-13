import re

def Clean_gen_code(generated_code):



    # 1. Replace element-wise MATLAB operators with PyTorch-compatible syntax
    generated_code = generated_code.replace('.*', '*')
    generated_code = generated_code.replace('./', '/')
    generated_code = generated_code.replace('.^', '**')  # MATLAB power op
    generated_code = generated_code.replace('^', '**')   # fallback catch-all

    generated_code = re.sub(
    r'self\.Off_Off_IC_trial\s*=\s*torch\.tensor\([^,]+,\s*dtype=torch\.float32\)',
    'self.Off_Off_IC_trial = torch.tensor(trial_num, dtype=torch.float32)',
    generated_code
    )

    generated_code = re.sub(
    r'self\.On_On_IC_trial\s*=\s*torch\.tensor\([^,]+,\s*dtype=torch\.float32\)',
    'self.On_On_IC_trial = torch.tensor(trial_num, dtype=torch.float32)',
    generated_code
    )
    
    #generated_code = generated_code.replace('self.On_On_IC_trial = torch.tensor(1.0, dtype=torch.float32)', 'self.On_On_IC_trial = torch.tensor(trial_num, dtype=torch.float32)') 

    #generated_code = generated_code.replace('len(np.arange(self.tspan[0],self.tspan[1],self.dt))', 'len(torch.arange(self.tspan[0],self.tspan[1],self.dt, device=self.device))') 


    # 2. Remove all lines that use randn (not differentiable)
    randn_pattern = re.compile(r'\+[^+]*?randn\([^\)]*\)')
    generated_code = randn_pattern.sub('', generated_code)

    # 3. Replace eye(A,B) -> torch.eye(A, B)
    generated_code = re.sub(r'\beye\(([^,]+),([^)]+)\)', r'np.eye(\1, \2)', generated_code)

    # 4. Replace ones(1,N) -> torch.ones(N) or torch.ones(1, N)
    #generated_code = re.sub(r'\bones\(\s*1\s*,\s*([^)]+)\)', r'torch.ones(1, \1)', generated_code)
    #generated_code = re.sub(
    #r'(?<!\w)(-?\d*\.?\d+(?:e[+-]?\d+)?\s*\*\s*)?ones\(([^)]+)\)',
    #lambda m: f"{m.group(1) or ''}np.ones(({m.group(2)}))",
    #generated_code
    #)   

    # 5. Replace zeros(1,N) -> torch.zeros(1,N) or torch.zeros(2, N)
    #generated_code = re.sub(r'\bzeros\(([^,]+),([^)]+)\)', r'torch.zeros(\1, \2)', generated_code)
    #generated_code = re.sub(r'\bzeros\(([^,]+),([^)]+)\)', r'np.zeros((T, \2))', generated_code)
    #pattern = re.compile(r'\bzeros\(([^,]+),\s*([^)]+)\)')
    #parts   = generated_code.split('def main():', 1)   # split at first occurrence

    #if len(parts) == 2:                # we found a “def main():”
    #    before, after = parts
    #    before = pattern.sub(r'np.zeros((T, \2))', before)
    #    generated_code = 'def main():'.join([before, after])
    #else:                              # no “def main():” in the file
    #    generated_code = pattern.sub(r'np.zeros((T, \2))', generated_code)

    # 5b. Go ahead an make everything T? This just preallocated larger arrays. Make sure this functions as intended.
    #generated_code = re.sub(
    #    r'torch\.(ones|zeros)\(\s*\d+\s*,\s*([^)]+?)\s*\)',  # match torch.ones(int, something)
    #    r'torch.\1(T, \2)',  # replace with torch.ones(T, something)
    #    generated_code
    #)

    # 6. Replace genPoissonInputs/Times with thier python function calls
    # Replace genPoissonInputs with genPoissonInputs.gen_poisson_inputs, unless preceded by 'import '
    generated_code = re.sub(r'(?<!import\s)genPoissonInputs(?=\s*\()', 'genPoissonInputs.gen_poisson_inputs', generated_code)

    # Replace genPoissonTimes with genPoissonTimes.gen_poisson_times, unless preceded by 'import '
    generated_code = re.sub(r'(?<!import\s)genPoissonTimes(?=\s*\()', 'genPoissonTimes.gen_poisson_times', generated_code)


    #generated_code = generated_code.replace('self.On_On_IC_input = genPoissonInputs.gen_poisson_inputs(self.On_On_IC_trial,self.On_On_IC_locNum,self.On_On_IC_label,self.On_On_IC_t_ref,self.On_On_IC_t_ref_rel,self.On_On_IC_rec)', 'self.On_On_IC_input = genPoissonInputs.gen_poisson_inputs(self.On_On_IC_trial,self.On_On_IC_locNum,self.On_On_IC_label,self.On_On_IC_t_ref,self.On_On_IC_t_ref_rel,self.On_On_IC_rec,self.device)') 
    #generated_code = generated_code.replace('self.Off_Off_IC_input = genPoissonInputs.gen_poisson_inputs(self.Off_Off_IC_trial,self.Off_Off_IC_locNum,self.Off_Off_IC_label,self.Off_Off_IC_t_ref,self.Off_Off_IC_t_ref_rel,self.Off_Off_IC_rec)', 'self.Off_Off_IC_input = genPoissonInputs.gen_poisson_inputs(self.Off_Off_IC_trial,self.Off_Off_IC_locNum,self.Off_Off_IC_label,self.Off_Off_IC_t_ref,self.Off_Off_IC_t_ref_rel,self.Off_Off_IC_rec,self.device)') 
    #generated_code = generated_code.replace('self.R2On_R2On_iNoise_V3_token = genPoissonTimes.gen_poisson_times(self.R2On_Npop,self.R2On_R2On_iNoise_V3_dt,self.R2On_R2On_iNoise_V3_FR,self.R2On_R2On_iNoise_V3_sigma,self.R2On_R2On_iNoise_V3_simlen)','self.R2On_R2On_iNoise_V3_token = genPoissonTimes.gen_poisson_times(self.R2On_Npop,self.R2On_R2On_iNoise_V3_dt,self.R2On_R2On_iNoise_V3_FR,self.R2On_R2On_iNoise_V3_sigma,self.R2On_R2On_iNoise_V3_simlen,self.device)')

    # 7. Remove unused parameters (self.tspan to self.verbose_flag)
    skip_params = ['self.downsample_factor', 'self.random_seed',
                   'self.solver', 'self.disk_flag',
                   'self.datafile', 'self.mex_flag', 'self.verbose_flag','self.On_On_IC_trial','self.Off_Off_IC_trial']

    lines = generated_code.split('\n')

    lines = [line for line in lines if not any(param in line for param in skip_params)]

    generated_code = '\n'.join(lines)

    # 8. Fix formatting issue with tspan
    # Pattern to detect and fix torch.tensor(array(...)) syntax
    fixed_lines = []
    for line in lines:
        if "torch.tensor(array('d'," in line:
            line = re.sub(r"torch\.tensor\(array\('d',\s*", "torch.tensor(", line)
            line = re.sub(r"\)\s*,\s*dtype=", ", dtype=", line)

        if "tspan =" in line:
            line = "    tspan = np.array([0.1, 2980.1*scale_factor])" #De-hardcode these things eventually

        if "random_seed" in line:
            line = ""

        if "solver" in line:
            line = ""

        if "datafile" in line:
            line = ""

        if "R2On_R2On_iNoise_V3_simlen =" in line:
            line = "    R2On_R2On_iNoise_V3_simlen = tspan[1]*10"


        #if "genPoissonTimes.gen_poisson_times" in line:



        #if "R2On_R1On_PSC_gSYN =" in line and "dv_d" not in line:
        #    line = '    R2On_R1On_PSC_gSYN = p_Ron'

        #For the Npops make them ints
        line = re.sub(r'(\b\w*Npop\b)\s*=\s*([0-9]+)\.0\b', r'\1 = \2', line)

        fixed_lines.append(line)

    # Join back into a single string if needed
    #cleaned_code = '\n'.join(fixed_lines)

    # Reassemble the modified code
    generated_code = '\n'.join(fixed_lines)

    return generated_code
