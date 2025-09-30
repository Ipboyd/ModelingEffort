
import os
import Parser
import State_Parser
import FormatODEs_Ns
import State_variable_Identifier
import Extract_Fixed_vars
import Clean_up
import ConditionalActions
import add_device_to_tensors
import Update_Grads
import pdb
import textwrap
import numpy as np
import math
import re

#Recursion definition for tracking gradient paths
def recurse(cur_node, tracker,nodes,edges):

  tracker.append(cur_node)

  for k in edges:
    if cur_node == k.split('->', -1)[1]:
      recurse(k.split('->', -1)[0],tracker,nodes,edges)
    

  return tracker



def build_ODE(parameters,batch_size):

    #--------------------------------------------------------        Connect to DynaSim Solve File        -----------------------------------------------------------------------#

    home_dir = os.path.expanduser("~")  
    file_path = os.path.join(home_dir, "Documents", "GitHub", "ModelingEffort", 
                         "Single-Channel", "Model", "Model-Core", 
                         "Model-Main", "run", "1-channel-paper", "solve", "solve_ode_1_channel_paper.m") #This could be changed for TD and multichannel and what not.

    generated_code = ''


    #------------------------------------------------------         Parse File for Desired Variables        ---------------------------------------------------------------------#

    #For now, not going to set up the seperation between learnable and non-learnable parameters. At some point you might want to implement some way to select them.
    #TODO/ IMPLEMENT selection method.


    #Add the params that are set by calculations (i.e tau = RC)
    fixed_params = Extract_Fixed_vars.extract_fixed_variables_from_block(file_path)
    lhs_list, rhs_list = zip(*fixed_params)

    #Add monitors for conditional actions -> Stuff used in PSCs and tests
    monitor_response = State_Parser.extract_monitor_declarations(file_path)
    monitor_vars = list(monitor_response.keys())
    monitor_vals = [list(v.values())[0] for v in monitor_response.values()]

    #Add state Variables -> Variables like voltage and what not used in ODEs
    both_sides = State_Parser.extract_state_variables_from_block(file_path)
    state_vars = list(both_sides.keys())
    state_vals = [list(v.values())[0] for v in both_sides.values()]

    #Add update vars (state vars but reordered for update alignment w/ dynasim)
    both_sides2 = State_Parser.extract_state_update(file_path)
    update_vars = list(both_sides2.keys())
    update_vals = [list(v.values())[0] for v in both_sides2.values()]

    #Add the ODEs
    pairs = Parser.extract_rhs_lhs(file_path)

    #Add in test 3 conditionals \TODO do this with all conditionals
    statement_pairs = ConditionalActions.extract_conditional_variables(file_path)

    #-------------------------------------------------------------         Write the File        -------------------------------------------------------------------------------#

    #----Imports
    import_string = textwrap.dedent("""\
        import genPoissonTimes
        import genPoissonInputs
        import matplotlib.pyplot as plt
        import gc
        import scipy.io
        import numpy as np
        import Calc_output_grad
        import Update_params
    """)


    #----Declare forwards loop & Initialize variables
    forwards_loop_header = textwrap.dedent("""\
        def main(trial_number,ps,scale_factor):
    """)

    #Bring In params
    params = '\n    #Params\n'
    count_ps = 0
    for name, value in parameters.items():
        if "_gSYN" in name and "R2Off" not in name:
            #Initialize our parameters with the values we inherit from the learning
            params += f"    {name} = ps[{count_ps}]\n"
            count_ps += 1

        elif "fP" in name and "R2Off" not in name:

            params += f"    {name} = ps[{count_ps}]\n"
            count_ps += 1

        elif "t_ref" in name and "On_On_IC_t_ref" not in name and "Off_Off_IC_t_ref" not in name and "R2Off" not in name:
            #print('here')
            params += f"    {name} = ps[{count_ps}]\n"
            count_ps += 1

        elif "R2On_R2On_iNoise_V3_FR" in name:
            params += f"    {name} = ps[{count_ps}]\n"
            count_ps += 1

        elif "g_inc" in name and "R2Off" not in name:
            params += f"    {name} = ps[{count_ps}]\n"
            count_ps += 1

        elif "tau_ad" in name and "R2Off" not in name:
            params += f"    {name} = ps[{count_ps}]\n"
            count_ps += 1

        else:
            params += f"    {name} = {value}\n"

    
    #Initialize grads
    for name, value in parameters.items():
        if "_gSYN" in name and "R2Off" not in name:
            
            post_node = name.split('_',-1)[0]
            pre_node = name.split('_',-1)[1]
            
            #Vectorize the gradient initializations.
            params += f'    dGSYN{post_node}_{pre_node} = np.zeros(({batch_size}))\n'

    #| For debugging gradients (tracking gradients over time) keep the following code

    # for k in range(len(update_vars)):
    #     if "_V" in update_vars[k] and ("R" in update_vars[k] or "S" in update_vars[k]) and "Noise" not in update_vars[k] and "R2Off" not in update_vars[k]:
    #         var = update_vars[k]
    #         var_base = var[:-3]
    #         params += f'    dspike_d{var_base}_tracker = []\n'

    #Debug here

    #Bring in fixed params
    fixed_param_declaration = '\n    #Fixed Param Declaration\n'
    for k in range(len(lhs_list)):
        if (lhs_list[k] != 'On_On_IC_input' and lhs_list[k] != 'Off_Off_IC_input'):
            #if rhs_list[k].contains('ones'):
            np_version = re.sub(r'ones\(\s*1\s*,\s*([^)]+)\)',r'np.ones((1,\1))',rhs_list[k])
            fixed_param_declaration += f"    {lhs_list[k]} = {np_version}\n"            
            #else:
            #    fixed_param_declaration += f"    {lhs_list[k]} = {rhs_list[k]}\n"

       

    #Tspan reports the length of the stimulus in ms. np.arange is 0 index and exclusive which requires 1 more index compared to the matlab implemention
    T_and_Helper_declaration = '\n    T = len(np.arange(tspan[0],tspan[1]+(dt),dt))\n    helper = np.arange(tspan[0],tspan[1]+(dt),dt)\n'
   

    #| Outdated due to multicore parralelization

    #Bring in Spikes Holders
    # spike_holder_string = '\n    #Spikes Holders\n'
    # for k in range(len(monitor_vars)):
    #     if "V_spikes" in monitor_vars[k]:
    #         spike_holder_string += f"    {monitor_vars[k]} = []\n"
    
    generated_code = import_string + forwards_loop_header + params + fixed_param_declaration + T_and_Helper_declaration #+ spike_holder_string
    

    #----Initilize variables that get reset per trial

    #trial_loop_declaration = '\n    for trial_number in range(10):\n'

    #Bring in State vars
    state_vars_string = '\n    #State Variable Declaration\n'
    for k in range(len(state_vars)):
        state_vars_string += f"    {state_vars[k]} = np.ones(({batch_size},2)) * {State_variable_Identifier.replace_ones_zeros(state_vals[k])}\n"

    #Bring in Monitors
    monitor_string = '\n    #Monitor Declaration\n'
    for k in range(len(monitor_vars)):

        #print(monitor_vars[k])

        if "V_spikes" in monitor_vars[k]:
            monitor_string += f"    {monitor_vars[k]}_holder = []\n"
        elif 'syn' in monitor_vars[k]:
            monitor_string += f"    {monitor_vars[k]} = 0\n" #
        elif 'IC_' in monitor_vars[k]:
            monitor_string += f"    {monitor_vars[k]} = 0\n" #
        else:
            #If you move to multiple populations you will have to deal with the 1 here. It will need to be replaced by xx_Npop to be compatible with dynasim
            np_version_ones = re.sub(r'ones\s*\(\s*5\s*,\s*([^)]+)\s*\)',rf'np.ones(({batch_size}, 5, \1))',monitor_vals[k])
            np_version_second_check = re.sub(r'ones\s*\(\s*1\s*,\s*([^)]+)\s*\)',rf'np.ones(({batch_size}))',np_version_ones)
            np_version_zeros = re.sub(r'zeros\(\s*nsamp\s*,\s*([^)]+)\)',r'np.zeros((T,\1))',np_version_second_check)

            monitor_string += f"    {monitor_vars[k]} = {np_version_zeros}\n"

            #if 'tspike' in monitor_vars[k]:
            #    monitor_string += f"    {monitor_vars[k]} = {np_version_zeros} * np.ones(({batch_size},5,1))\n"
            #if 'buffer' in monitor_vars[k]:
            #    monitor_string += f"    {monitor_vars[k]} = {np_version_zeros} * np.ones(({batch_size},1,1))\n"


    #Declare Inputs
    inputs_header = "\n    #Delcare Inputs\n    On_On_IC_input = genPoissonInputs.gen_poisson_inputs(trial_number,On_On_IC_locNum,On_On_IC_label,On_On_IC_t_ref,On_On_IC_t_ref_rel,On_On_IC_rec,scale_factor)\n    Off_Off_IC_input = genPoissonInputs.gen_poisson_inputs(trial_number,Off_Off_IC_locNum,Off_Off_IC_label,Off_Off_IC_t_ref,Off_Off_IC_t_ref_rel,Off_Off_IC_rec,scale_factor)\n"

    #generated_code = generated_code + trial_loop_declaration + state_vars_string + monitor_string + inputs_header
    generated_code = generated_code + state_vars_string + monitor_string + inputs_header

    #----ODE Intermost loop
    
    ODE_loop_Declaration = '\n    for t in range(0,T):\n'

    #ODE declarations
    ode_string = '\n        #ODEs\n'
    for k in range(len(pairs)):
        rhs_ode = FormatODEs_Ns.reformat_input_time_indexing(FormatODEs_Ns.reformat_discrete_time_indexing(pairs[k][1]))
        rhs_ode_rpl = rhs_ode.replace("[t-1]", "[:,-1]")
        ode_string += f"        {pairs[k][0]} = {rhs_ode_rpl}\n"
        
   
    #Update Eulers
    update_eulers = '\n        #Update Eulers\n'

    for k in range(len(update_vars)):
        rep_val = update_vals[k].replace("[t-1]", "[:,-1]")
        update_eulers += f"        {update_vars[k][:-3]}[:,-2] = {update_vars[k][:-3]}[:,-1]\n"
        update_eulers += f"        {update_vars[k][:-3]}[:,-1] = {rep_val}\n"

    #Spiking Behavior
    test1_string = '\n        #Spiking and conditional actions\n'
    
    #Test 1 (Spiking activity)
    for k in range(len(update_vars)):
        if "_V[t]" in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            var_name = var[:-5]
            var_prev = var.replace("[t]", "[:,-2]")
            var_thresh = f"{var.replace('[t]', '_thresh')}"
            test1_string += f"        mask = (({var_base}[:,-1] >= {var_thresh}) & ({var_prev} < {var_thresh})).astype(np.int8).tolist()\n"
            test1_string += f"        {var_base}_spikes_holder.append(mask)\n"
            test1_string += f"        if np.any(mask):\n"   
            test1_string += f"            spikers = np.flatnonzero(mask)\n"
            #test1_string += f"            cols = ({var_name}_buffer_index[spikers].astype(np.int8)-1)\n"
            #test1_string += f"            print(mask)\n"
            #test1_string += f"            print(spikers)\n"
            #test1_string += f"            print({var_name}_buffer_index)\n"
            test1_string += f"            {var_name}_tspike[spikers, {var_name}_buffer_index[spikers].astype(np.int8)-1] = helper[t]\n"
            #test1_string += f"            print({var_name}_tspike)\n"
            #test1_string += f"            print(spikers)\n"             
            #test1_string += f"            print({var_name}_buffer_index)\n"
            test1_string += f"            {var_name}_buffer_index[spikers] = ({var_name}_buffer_index[spikers] % 5) + 1\n"
            

    #Test 2 (Voltage reset and adaptation) 
    test2_string = '\n            #Voltage reset and adaptation\n'
    for k in range(len(update_vars)):
        if "_V[t]" in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            var_name = var[:-5]
            var_thresh = f"{var.replace('[t]', '_thresh')}"
            var_reset = f"{var.replace('[t]', '_reset')}"
            var_adapt = f"{var.replace('V[t]', 'g_ad[t]')}"
            var_inc = f"{var.replace('V[t]', 'g_inc')}"
            test2_string += f"        mask = ({var_base}[:,-1] > {var_thresh})\n"#.astype(np.int8).tolist()\n"
            #test2_string += f"        {var_base}_test2a = {var_base}[-1] > {var_thresh}\n"
            test2_string += f"        if np.any(mask):\n"
            test2_string += f"            spikers = np.flatnonzero(mask) \n"
            test2_string += f"            {var[:-3]}[spikers,-2] = {var[:-3]}[spikers,-1] \n"
            test2_string += f"            {var[:-3]}[spikers,-1] = {var_reset} \n"
            test2_string += f"            {var_adapt[:-3]}[spikers,-2] = {var_adapt[:-3]}[spikers,-1]\n"
            test2_string += f"            {var_adapt[:-3]}[spikers,-1] = {var_adapt[:-3]}[spikers,-1] + {var_inc}\n"
            #test2_string += f"        print((helper[t] <= ({var_name}_tspike + {var_name}_t_ref)))\n"

            
            #If t_ref has a batch dimention then its dim needs to be extended

            if "R2Off" not in update_vars[k]:

                test2_string += f"        mask = np.any((helper[t] <= (np.squeeze({var_name}_tspike) + {var_name}_t_ref[:, None])), axis = 1)\n"#.astype(np.int8).tolist()\n"
            else:

                test2_string += f"        mask = np.any((helper[t] <= (np.squeeze({var_name}_tspike) + {var_name}_t_ref)), axis = 1)\n"#.astype(np.int8).tolist()\n"

            #test2_string += f"        {var_base}_test2b = np.any(helper[t] <= {var_name}_tspike + {var_name}_t_ref)\n"
            test2_string += f"        if np.any(mask):\n"
            test2_string += f"            spikers = np.flatnonzero(mask) \n"
            
            #test2_string += f"            print({var_name}_t_ref)\n"
            #test2_string += f"            print({var_name}_tspike)\n"

            #test2_string += f"            print(np.shape({var_name}_tspike))\n"
            #test2_string += f"            print(np.shape({var_name}_t_ref))\n"

            #test2_string += f"            print(np.shape(mask))\n"
            #test2_string += f"            print(np.shape({var_name}_tspike + {var_name}_t_ref))\n"
            #test2_string += f"            print(np.shape(np.array({var_name}_tspike)+np.array({var_name}_t_ref)))\n"

            test2_string += f"            {var[:-3]}[spikers,-2] = {var[:-3]}[spikers,-1]\n"
            test2_string += f"            {var[:-3]}[spikers,-1] = {var_reset}\n"

    #Test 3 (Update PSC vars) 
    test3_string = '\n            #Update PSC vars\n'

    for k in range(len(statement_pairs)):

        var_base = var[:-3]
        var = statement_pairs[k][1]
        var_x = f"{var.replace('delay', 'x')}"
        var_q = f"{var.replace('delay', 'q')}"
        var_F = f"{var.replace('delay', 'F')}"
        var_P = f"{var.replace('delay', 'P')}"
        var_fF = f"{var.replace('delay', 'fF')}"
        var_max = f"{var.replace('delay', 'maxF')}"
        var_fP = f"{var.replace('delay', 'fP')}"

        test3_string += f"        mask = np.any((helper[t] == ({statement_pairs[k][0]} + {statement_pairs[k][1]})),axis = 1)\n"#".astype(np.int8).tolist()\n"  
        #test3_string += f"        {var_base}_test3 = np.any(helper[t] == {statement_pairs[k][0]} + {statement_pairs[k][1]})\n"  
        test3_string += f"        if np.any(mask):\n"  
        test3_string += f"            spikers = np.flatnonzero(mask) \n"
        test3_string += f"            {var_x}[spikers,-2] = {var_x}[spikers,-1]\n"
        test3_string += f"            {var_q}[spikers,-2] = {var_F}[spikers,-1]\n"
        test3_string += f"            {var_F}[spikers,-2] = {var_F}[spikers,-1]\n"
        test3_string += f"            {var_P}[spikers,-2] = {var_P}[spikers,-1]\n"
        test3_string += f"            {var_x}[spikers,-1] = {var_x}[spikers,-1] + {var_q}[spikers,-1]\n"
        test3_string += f"            {var_q}[spikers,-1] = {var_F}[spikers,-1] * {var_P}[spikers,-1]\n"
        test3_string += f"            {var_F}[spikers,-1] = {var_F}[spikers,-1] + {var_fF}*({var_max}-{var_F}[spikers,-1])\n"

        if "R2Off" not in var_fP:

            test3_string += f"            {var_P}[spikers,-1] = {var_P}[spikers,-1] * (1 - {var_fP}[spikers])\n"

        else:

            test3_string += f"            {var_P}[spikers,-1] = {var_P}[spikers,-1] * (1 - {var_fP})\n"

    generated_code = generated_code + ODE_loop_Declaration + ode_string + update_eulers + test1_string + test2_string + test3_string
    

    return_statement = f"\n    return R2On_V_spikes_holder"

    generated_code = generated_code + return_statement

    #--------------------------------------------------------         Clean up and port to .py        --------------------------------------------------------------------------#
   
    generated_code = Clean_up.Clean_gen_code(generated_code)

    with open("generated_GA.py", "w") as f:
        f.write(generated_code)

    print("generated_GA.py has been created.")

    return generated_code
