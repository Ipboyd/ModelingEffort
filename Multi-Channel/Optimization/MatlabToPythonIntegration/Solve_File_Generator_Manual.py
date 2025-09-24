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



def build_ODE(parameters,batch_size,learning_mask):

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
    count_ps2 = 0
    for k in range(len(lhs_list)):
        if learning_mask[1] == 1:
            if "tau" in lhs_list[k] and 'R2Off' not in lhs_list[k]:
                    fixed_param_declaration += f"    {lhs_list[k]} = ps[{count_ps + count_ps2}]\n"   
                    count_ps2 += 1

            elif (lhs_list[k] != 'On_On_IC_input' and lhs_list[k] != 'Off_Off_IC_input'):
                #if rhs_list[k].contains('ones'):
                np_version = re.sub(r'ones\(\s*1\s*,\s*([^)]+)\)',r'np.ones((1,\1))',rhs_list[k])
                fixed_param_declaration += f"    {lhs_list[k]} = {np_version}\n"   

        elif (lhs_list[k] != 'On_On_IC_input' and lhs_list[k] != 'Off_Off_IC_input'):
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

    #Declare derivative holders for previous derivatives necessary for spiking derivative
    for k in range(len(update_vars)):
        if "_V" in update_vars[k] and  "Noise" not in update_vars[k] and "R2Off" not in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            var_name = var[:-5]

            monitor_string += f"    dv1d{var_base} = np.ones(({batch_size},2))\n"
            monitor_string += f"    dv2d{var_base} = np.ones(({batch_size},2))\n"

            #Adding intialization for gradients to latch on to
            monitor_string += f"    spikers_{var_name} = np.zeros(({batch_size})).astype(np.int8)\n"


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
            test1_string += f"            spikers_{var_name} = np.flatnonzero(mask)\n"
            #test1_string += f"            cols = ({var_name}_buffer_index[spikers].astype(np.int8)-1)\n"
            #test1_string += f"            print(mask)\n"
            #test1_string += f"            print(spikers)\n"
            #test1_string += f"            print({var_name}_buffer_index)\n"
            test1_string += f"            {var_name}_tspike[spikers_{var_name}, {var_name}_buffer_index[spikers_{var_name}].astype(np.int8)-1] = helper[t]\n"
            #test1_string += f"            print({var_name}_tspike)\n"
            #test1_string += f"            print(spikers)\n"             
            #test1_string += f"            print({var_name}_buffer_index)\n"
            test1_string += f"            {var_name}_buffer_index[spikers_{var_name}] = ({var_name}_buffer_index[spikers_{var_name}] % 5) + 1\n"
            

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
            test2_string += f"        mask = np.any((helper[t] <= ({var_name}_tspike + {var_name}_t_ref)), axis = 1)\n"#.astype(np.int8).tolist()\n"
            #test2_string += f"        {var_base}_test2b = np.any(helper[t] <= {var_name}_tspike + {var_name}_t_ref)\n"
            test2_string += f"        if np.any(mask):\n"
            test2_string += f"            spikers = np.flatnonzero(mask) \n"
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
        test3_string += f"            {var_P}[spikers,-1] = {var_P}[spikers,-1] * (1 - {var_fP})\n"

    generated_code = generated_code + ODE_loop_Declaration + ode_string + update_eulers + test1_string + test2_string + test3_string
    
    #----Gradient Calculations

    #Going to try to build in gradient to forwards loop
    #Build out gradients
    grad_string = '\n        #Grad Calculations\n'


    #Grab all of the valid voltage update ones. This is the spiking deriavte wrt the voltage


    #Current issue: This can go to infinity for some reason

    grad_string += '\n\n        #Surrogate Spike Related Derivates\n'

    for k in range(len(update_vars)):
        if "_V" in update_vars[k] and  "Noise" not in update_vars[k] and "R2Off" not in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            #grad_string += f'        print(\'here2\')\n' 

            grad_string += f'        #dspike_d{var_base} = (((10*np.exp(-(0.1)*({var_base}[:,-1] - {var_base}_thresh)))/(1+np.exp(-(0.1)*({var_base}[:,-1] - {var_base}_thresh)))**2))/500\n'  #Nominally 500
            #Incorrect dims
            #grad_string += f'        print(\'spikewrtV\')\n'
            #grad_string += f'        print(dspike_d{var_base})\n'


            #Instead of normalizing with a constant you could track the derivatives at every time step for each trial and then divide by the maximum (initialized to 1?)

            #grad_string += f'            dv_d{var_base}_tracker.append(dspike_d{var_base})\n'

    #implmenting spiking derivatives with all dependencies considered for voltage.
    for k in range(len(update_vars)):
        if "_V" in update_vars[k] and  "Noise" not in update_vars[k] and "R2Off" not in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            var_name = var[:-5]
            #Move the old derivative out of the way (but save it to be used in later calculation) -> 1 will become new derivatived while 0 retains old version of 1.
            grad_string += f'        dv1d{var_base}[:,0] = dv1d{var_base}[:,1]\n'
            grad_string += f'        dv1d{var_base}[:,1] = ((1+np.exp({var_base}[:,-1]-{var_base}_thresh))-({var_base}[:,-1]-{var_base}_reset*np.exp({var_base}[:,-1]-{var_base}_thresh)))/(1+np.exp({var_base}[:,-1]-{var_base}_thresh))**2\n'

            grad_string += f'        dv2d{var_base}[:,0] = dv2d{var_base}[:,1]\n'
            grad_string += f'        dv2d{var_base}[:,1] = np.squeeze(np.sum(1/(1+np.exp(-(helper[t]-({var_name}_tspike+{var_name}_t_ref)))),axis=1))\n'

            #grad_string += f'        print("dv2d{var_base}[:,1]")\n'
            #grad_string += f'        print(np.shape(dv2d{var_base}[:,1]))\n'

            #Not sure if V_prev is legitimate in this scenario or if I need to pull vprev from the true start of the previous eulers timestep

            #grad_string += f'        buff1 = (({var_name}_buffer_index[spikers_{var_name}] % 5) + 1).astype(np.int64)\n'
            #grad_string += f'        buff2 = ((({var_name}_buffer_index[spikers_{var_name}] - 1) % 5) + 1).astype(np.int64)\n'
            
            #grad_string += f'        print(np.shape({var_name}_tspike))\n'
            #grad_string += f'        print({var_name}_tspike)\n'

            #grad_string += f'        dukdv2{var_base} = (-np.squeeze(np.take_along_axis(np.squeeze({var_name}_tspike),buff1[:, None]-1,axis = 1)-np.take_along_axis(np.squeeze({var_name}_tspike),buff2[:, None]-1,axis = 1))*(-np.exp(-({var_base}[:,-1]-{var_base}_thresh)))*(1+np.exp(({var_base}[:,-2]-{var_base}_thresh))))/((1+np.exp(-({var_base}[:,-1]-{var_base}_thresh)))*(1+np.exp(({var_base}[:,-2]-{var_base}_thresh))))**2\n'
            
            #grad_string += f'        print(np.shape((np.max({var_name}_tspike,axis=1))))\n'
            #grad_string += f'        print(np.shape(np.partition({var_name}_tspike, -2, axis=1)[:, -2]))\n'

            #grad_string += f'        print(np.shape((np.max({var_name}_tspike,axis=1))-np.partition({var_name}_tspike, -2, axis=1)[:, -2]))\n'

            grad_string += f'        dukdv2{var_base} = (-np.squeeze((np.max({var_name}_tspike,axis=1)-np.partition({var_name}_tspike, -2, axis=1)[:, -2]))*(-np.exp(-({var_base}[:,-1]-{var_base}_thresh)))*(1+np.exp(({var_base}[:,-2]-{var_base}_thresh))))/((1+np.exp(-({var_base}[:,-1]-{var_base}_thresh)))*(1+np.exp(({var_base}[:,-2]-{var_base}_thresh))))**2\n'

        
            

            grad_string += f'        dspike_d{var_base} = dukdv2{var_base}*dv2d{var_base}[:,0]*dv1d{var_base}[:,0]\n'

            #grad_string += f'        print("dukdv2{var_base}")\n'
            #grad_string += f'        print(np.shape(({var_name}_tspike[:,-1]-{var_name}_tspike[:,-2])))\n'
    
    if learning_mask[1] == 1:
        grad_string += '\n\n        #Tau Related Derivates\n'
        #Tau related derivatives

        #Look through the ODEs and just extrac the voltage related traces
        for k in range(len(pairs)):
            rhs_ode = FormatODEs_Ns.reformat_input_time_indexing(FormatODEs_Ns.reformat_discrete_time_indexing(pairs[k][1]))
            rhs_ode_rpl = rhs_ode.replace("[t-1]", "[:,-1]")

            #LHS comparison
            if "_V_" in pairs[k][0]:
                ode_base = pairs[k][0][:-5]
                grad_string += f'        dv_dtau_{ode_base} = np.squeeze(-(dt*{rhs_ode_rpl}**2))\n' #Note the square works here because tau is listed as the last variables in the ODE string
                

            #ode_string += f"        {pairs[k][0]} = {rhs_ode_rpl}\n"

        #for k in range(len(update_vars)):
        #    if "_V" in update_vars[k] and ("R" in update_vars[k] or "S" in update_vars[k]) and "Noise" not in update_vars[k] and "R2Off" not in update_vars[k]:
        #        var = update_vars[k]
        #        var_base = var[:-3]
                #grad_string += f'        print(\'here2\')\n' 
                
                #Extract the correct line. Don't need post and pre cell here since the line just stays the same and tau is onyl effeced
        #        grad_string += f'        dv_dtau_{var_base} = np.squeeze(-(dt*{post_cell}_R*{post_cell}_{pre_cell}_PSC_s[:,-1]*{post_cell}_{pre_cell}_PSC_netcon*({post_cell}_V[:,-1]-{post_cell}_{pre_cell}_PSC_ESYN)/{post_cell}_tau**2)/15)\n' #Nominally 15

    #Search for all of the gsyns that we want to update
    #Write all of the partials of the voltage w.r.t the parameters
    grad_string += '\n\n        #PSC & Parameter Related Derivates\n'
    for name, value in parameters.items():
        if "_gSYN" in name and "R2Off" not in name:
            
            post_cell = name.split('_', -1)[0]
            pre_cell = name.split('_', -1)[1]

            #grad_string += f'            print(helper[t])\n'
            
            grad_string += f'        dv_d{name} = np.squeeze(-(dt*{post_cell}_R*{post_cell}_{pre_cell}_PSC_s[:,-1]*{post_cell}_{pre_cell}_PSC_netcon*({post_cell}_V[:,-1]-{post_cell}_{pre_cell}_PSC_ESYN)/{post_cell}_tau)/15)\n' #Nominally 15
            
            #grad_string += f'            dv_d{name} = -dt*{post_cell}_R*{post_cell}_{pre_cell}_PSC_netcon*({post_cell}_V[-1]-{post_cell}_{pre_cell}_PSC_ESYN)/{post_cell}_tau\n'
            #Incorrect dims
            #grad_string += f'        print(\'vwrtparam\')\n'
            #grad_string += f'        print(dv_d{name})\n'
            #if "R2On" in name and "R1On" in name:

            
            #grad_string += f'            print(dv_d{name})\n'

            #Make sure you are summing accross the correct dimention
            
            grad_string += f'        d{post_cell}_{pre_cell}_PSC_dUk = np.squeeze(-((dt*{post_cell}_{pre_cell}_PSC_scale*2*({post_cell}_{pre_cell}_PSC_x[:,-1]+{post_cell}_{pre_cell}_PSC_q[:,-1])/{post_cell}_{pre_cell}_PSC_tauR)*helper[t]*np.squeeze(np.sum(((({pre_cell}_tspike+{post_cell}_{pre_cell}_PSC_delay)-helper[t])*np.exp(-1*(({pre_cell}_tspike+{post_cell}_{pre_cell}_PSC_delay)-helper[t])**2)),axis=1)))/2500)\n' #Nominally 2500
            
            #Wrong Dims
            #grad_string += f'            d{post_cell}_{pre_cell}_PSC_dUk = -(dt*{post_cell}_{pre_cell}_PSC_scale*2/{post_cell}_{pre_cell}_PSC_tauR)*helper[t]*sum((({pre_cell}_tspike+{post_cell}_{pre_cell}_PSC_delay)-helper[t])*np.exp(-1*(({pre_cell}_tspike+{post_cell}_{pre_cell}_PSC_delay)-helper[t])**2))\n'
            #grad_string += f'        print(\'pscwrtspike\')\n'
            #grad_string += f'        print(np.squeeze(np.sum(((({pre_cell}_tspike+{post_cell}_{pre_cell}_PSC_delay)-helper[t])*np.exp(-1*(({pre_cell}_tspike+{post_cell}_{pre_cell}_PSC_delay)-helper[t])**2)),axis=1)))\n'
            #grad_string += f'        print(-(dt*{post_cell}_{pre_cell}_PSC_scale*2*({post_cell}_{pre_cell}_PSC_x[:,-1]+{post_cell}_{pre_cell}_PSC_q[:,-1])/{post_cell}_{pre_cell}_PSC_tauR)*helper[t])\n'
            #grad_string += f'            print(d{post_cell}_{pre_cell}_PSC_dUk)\n'
            grad_string += f'        dv_d{post_cell}_{pre_cell}_PSC = np.squeeze(-(dt*{post_cell}_R*{name}*{post_cell}_{pre_cell}_PSC_netcon*({post_cell}_V[:,-1]-{post_cell}_{pre_cell}_PSC_ESYN)/{post_cell}_tau)/10)\n' #Nominally 10
            #grad_string += f'            print(dv_d{post_cell}_{pre_cell}_PSC)\n'
            #Incorrect Dims
            #grad_string += f'        print(\'vwrtpsc\')\n'
            #grad_string += f'        print(dv_d{post_cell}_{pre_cell}_PSC)\n'
            #print('cells')
            #print(pre_cell)
            #print(post_cell)

            post_node = name.split('_',-1)[0]
            pre_node = name.split('_',-1)[1]


            

            #if "R1On_" in name and "_On" in name:
            #    grad_string += f'            if dspike_dR2On_V != 0:\n                voltage_derivative.append(dspike_dR2On_V*dv_dR2On_R1On_PSC*dR2On_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_On_PSC_gSYN)\n'
            #    grad_string += f'            if d{post_cell}_{pre_cell}_PSC_dUk != 0:\n                psc_derivative.append(d{post_cell}_{pre_cell}_PSC_dUk)\n'


    # grad_string += '\n\n            #Chcks\n'
    # for name, value in parameters.items():
    #     if "_gSYN" in name and "R2Off" not in name:
    #         post_cell = name.split('_', -1)[0]
    #         pre_cell = name.split('_', -1)[1]
    #         if "R2On_" in name and "_R1On" in name:
    #             grad_string += f'            if dspike_dR2On_V*dv_dR2On_R1On_PSC*dR2On_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_On_PSC_gSYN+dspike_dR2On_V*dv_dR2On_S2OnOff_PSC*dR2On_S2OnOff_PSC_dUk*dspike_dS2OnOff_V*dv_dS2OnOff_R1On_PSC*dS2OnOff_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_On_PSC_gSYN != 0:\n                voltage_derivative.append(dR2On_S2OnOff_PSC_dUk)\n'
    #             grad_string += f'            if dspike_dR2On_V*dv_dR2On_R1On_PSC_gSYN != 0:\n                psc_derivative.append(dv_dR2On_R1On_PSC_gSYN)\n'  


    

    #Put together Partials
    #Eventually it might be nice to automate this, however it is somewhat convoluted.
    #Just going to go ahead an automate it

    #Need to find a way to just get a list of all the possible paths are are upstream of R2on
    #Just going to write out nodes and edges and do that


    #************** Code Review Area #1 ****************

    #--------------------------------------------------------------------------#
    #                  #DFS algorithm for compiling derivatives                #
    #--------------------------------------------------------------------------#

    #Should I make this its own .py file?
    
    #List out all of the edges in the network (see 1)
    nodes = ['R2On','R1On','On','Off','S2OnOff','S1OnOff','R1Off','R2Off']
    edges = ['On->R1On','On->S1OnOff','Off->R1Off','Off->S1OnOff','S1OnOff->R1On','S1OnOff->R1Off','R1On->R2On','R1On->S2OnOff','R1Off->R2Off','R1Off->S2OnOff','S2OnOff->R2On','S2OnOff->R2Off']

    #Preform depth first search to get a list of all paths that lead to output node (ex. R2On)
    #
    #Inputs : reference node, all nodes, edges
    #Outptus : All of the nodes (str) in the order they are visited (Depth first)
    #
    all_paths = recurse('R2On',[],nodes,edges)

    #print('DFS output')
    #print(all_paths)

    #The DFS output only reports the order nodes are visited in NOT all the paths that I need to track for building the derivatives
    #The following portion Extracts all of the paths that lead to the first nodal entry (which should be the reference node)
    #Only paths that flow to the reference node are relevant in updating the gradients.
    #If all possible paths that flow to the reference node are reported then all possible relavent supaths will be definition be included.
    path_holder = []
    path = ''
    gate = 0
    gate2 = 0
    
    #Start by going through each node in the DFS
    for k in range(len(all_paths)-1):

        #Look through the edges:
        # Here lets designate downstream nodes as those the arrow points towards in edges so for ex. On->R1On  :  On = Upstream , R1On = Downstream
        for m in range(len(edges)):
            if edges[m].split('->', -1)[0] == all_paths[k+1] and edges[m].split('->', -1)[1] == all_paths[k]:
                path = edges[m] + '->' + path
                gate = 1    

        #If you have gone through all of them and the gate does not get flipped then append
        #This checks to see if we are at the end of a "primary path" which is a path the goes from the reference node to the last possilbe node. By doing this we include all node subpaths that reach the reference node.
        if gate == 0:
            #First append the total path but exclue the hanging ->
            path_holder.append(path[:-2])

            #Now we need to backtrack up the graph in the DFS search. We have found the last node in the DFS search (2) and we need to go back up to where a path branches (note it might go all the way back to the reference node)
            #Search back through the number of nodes that we have already gone through (k)
            for z in range(k):
                # Look at all of the edges to see if there is a branching connection
                for m in range(len(edges)):
                    if edges[m].split('->', -1)[0] == all_paths[k+1] and edges[m].split('->', -1)[1] == all_paths[k-(z+1)]:
                        
                        #Split the current path up into segments (-1 just means as many as possible)
                        path_segmented = path.split('->',-1)

                        #Now through the segemented path
                        for count, n in enumerate(path_segmented):
                            
                            #If the segement matches the dowsnstream node, that branched from the upstearm node detected
                            #then cut the path and update the path with the new node.
                            if n == edges[m].split('->', -1)[1]:

                                path = path.split('->',count+1)[count+1]
                                path = edges[m] + '->' + path

                                #This detects the edge case where all paths have been found. 
                                #This automatcailly stops the sequence and breaks out of the chain
                                if k == len(all_paths)-2 and gate2 == 0:
                                     gate2 = 1
                                     path_holder.append(path[:-2])
                                break

                        break

        gate = 0
    
    #print('list of all primary paths')
    #print(path_holder)
    
    #Now we have to build all of the derivatives in the file
    #The following code takes the primary paths and constructs all the gradients for all sub paths.
    grad_string += '\n        #Build derivs\n'

    #Iterate through all of the parameters. 
    #As it stands we are just looking at gSYN
    for name, value in parameters.items():
        #Right now ignore R2Off because it is not on the update path and calculating it will just take up space (TODO implement automatic collection for nodes on actual update path)
        if "gSYN" in name and "R2Off" not in name:
            
            #Here we are going to look through each path check to see if the parameter is on the path we are currently looking at
            cur_divs = []
            for cur_path in path_holder:
                
                #Grab the downstream (post node) and upstream (pre node)
                post_node = name.split('_',-1)[0]
                pre_node = name.split('_',-1)[1]

                #Path (Going to iterate through them just to qualify that the gsyn can only be updated by that synapse)
                path_segements = cur_path.split('->',-1)

                path_gate = 0
                check = 0

                #Loop through each of the path segements and see if it contains the parameter we are trying to update
                for ps in range(len(path_segements)):

                    #If the parameter exists on the path and has both the pre and post node then move to the next part (in this case the synaptic parameter is a connection between two nodes)
                    if path_segements[ps] == pre_node:

                        path_gate = 1

                    if path_gate == 1:
                    
                        path_gate = 0

                        if path_segements[ps+1] == post_node:
                            check = 1
                            path_gate = 0
                
                #If we have a valid connection then build the derivative 
                if check == 1:

                    deriv = ''

                    #all paths have to include the reference node at some pint.
                    #what happens here is that you look through your path for a given node if it happens that you find this node early then you are close to the reference node and you derivative
                    #will be compact. If you have something far away it will iterate through until the pre_der = pre_node (pre_node is set way up above). you will iterate until you hit the parameter edge and then break out.
                    for count_tar in range(int(len(path_segements)/2)):
                        pre_der = path_segements[-(count_tar*2 + 2)]
                        post_der = path_segements[-(count_tar*2 + 1)]

                        #3. Add the spiking derivate
                        deriv += 'dspike_d' + post_der + '_V*'

                        #If we are at the last relavent node add the derivative w.r.t. the parameter
                        if pre_der == pre_node:
                            deriv += 'dv_d' + post_der + '_' + pre_der + '_PSC_gSYN'
                            break
                        #Else keep thre chain going by adding the derivatve w.r.t the psc and then w.r.t spiking
                        else:
                            deriv += 'dv_d' + post_der + '_' + pre_der + '_PSC*' + 'd' + post_der + '_' + pre_der + '_PSC_dUk*'

                    cur_divs.append(deriv)




            
            #Construct the actual gradient that will be in the script
            grad_string += '        dGSYN' + post_node + '_' + pre_node +' += ' 
            
            #Check to make sure that each path you are putting in is unique and not a duplicate path (duplicate sub-paths are find)
            for un_divs in np.unique(cur_divs):
                grad_string += f'{un_divs}+'
            grad_string = grad_string[:-1]
            grad_string += '\n'

    generated_code = generated_code + grad_string


    #Build the Tau derivatvies

    #For each node
        #Find each subpath within each path
            #Compute and add the derivatives for each subpath
    if learning_mask[1] == 1:
        tau_grads = '\n\n        #Build Tau derivs\n'
        #Redefined Nodes here so that they are in the same order that they will be updated in in the gradient calculations.
        nodes2 = ['On','Off','R1On','R1Off','S1OnOff','R2On','S2OnOff']
        for k in nodes2: 

            #print(f'\n\n{k}\n\n')

            subpath_grads = []

            for m in path_holder:
                path_parts = m.split('->',-1)
                for count,z in enumerate(path_parts):
                    if z == k:

                        #Instead of printing m print the corresponding sub-path
                    
                        subpath = path_parts[count:]
                    
                        #Compensate for intermediate nodes
                        if len(subpath) > 1:
                            if subpath[0] == subpath[1]:
                                subpath = subpath[1:]
                    
                    
                        grad = ''



                        #rev_subpath = reversed(np.unique(subpath))

                        uniq, idx = np.unique(subpath, return_index=True)
                        rev_subpath = uniq[np.argsort(idx)][::-1]

                        #rev_subpath = np.unique(subpath)#[::-1]
                        #print(subpath)
                        #print(rev_subpath)

                        #print(len(subpath))

                        for count2,q in enumerate(rev_subpath):

                        

                            if count2 + 1 == len(rev_subpath):
                               #uk/v * v/tau
                               grad += f'dspike_d{q}_V*dv_dtau_{q}'
                            else:
                               #uk/v * v/psc * psc/uk
                               grad += f'dspike_d{q}_V*dv_d{q}_{rev_subpath[count2+1]}_PSC*d{q}_{rev_subpath[count2+1]}_PSC_dUk*'


                        subpath_grads.append(grad)

                        #print(subpath)
                        break

            #print(np.unique(subpath_grads))

            #Build the Tau derivatvies!

            grad_str_tau = ''
            for count3, g in enumerate(np.unique(subpath_grads)):
                if count3 + 1 == len(np.unique(subpath_grads)):
                    grad_str_tau += g
                else:
                    grad_str_tau += g + '+'

            tau_grads += f'        dTAU_{k} = {grad_str_tau}\n'

        generated_code = generated_code + tau_grads

    #************** Code Review Area #1 END! ****************


    #----Multi Core Return statement

    deriv_return = ''

    #for name, value in parameters.items():
    #    #Right now ignore R2Off because it is not on the update path and calculating it will just take up space (TODO implement automatic collection for nodes on actual update path)
    #    if "gSYN" in name and "R2Off" not in name:
    #        post_node = name.split('_',-1)[0]
    #        pre_node = name.split('_',-1)[1]
    #
    #        deriv_return += f'\n    print(max(dGSYN{post_node}_{pre_node}_tracker))\n'

    for k in range(len(update_vars)):
        if "_V" in update_vars[k] and ("R" in update_vars[k] or "S" in update_vars[k]) and "Noise" not in update_vars[k] and "R2Off" not in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            #deriv_return += f'    print(max(\'dspike_d{var_base}_tracker\'))\n'
            #deriv_return += f'    print(max(dspike_d{var_base}_tracker))\n'

    deriv_return2 = ''

    for name, value in parameters.items():
        #Right now ignore R2Off because it is not on the update path and calculating it will just take up space (TODO implement automatic collection for nodes on actual update path)
        if "gSYN" in name and "R2Off" not in name:
            post_node = name.split('_',-1)[0]
            pre_node = name.split('_',-1)[1]
            deriv_return2 += 'dGSYN' + post_node + '_' + pre_node +', '
    

    if learning_mask[1] == 1:
        for k in nodes[:-1]:
            deriv_return2 += f'dTAU_{k}, '
    
    deriv_return2 = '[' + deriv_return2[:-2] + ']'

    return_statement = f"\n{deriv_return}\n    return R2On_V_spikes_holder, {deriv_return2}"

    generated_code = generated_code + return_statement



    #----Post loop appending

    # #Append spikes
    # append_spikes = '\n        #Append Spikes\n'
        
    # for k in range(len(monitor_vars)):
    #     if "V_spikes" in monitor_vars[k]:    
    #         append_spikes += f'        {monitor_vars[k]}.append({monitor_vars[k]}_holder)\n'
    #         #append_spikes += f'        print(max({monitor_vars[k]}_holder))\n'

    # for k in range(len(update_vars)):
    #     if "_V" in update_vars[k] and ("R" in update_vars[k] or "S" in update_vars[k]) and "Noise" not in update_vars[k] and "R2Off" not in update_vars[k]:
    #         var = update_vars[k]
    #         var_base = var[:-3]
            
    #         #append_spikes += f'        print(max(dv_d{var_base}_tracker))\n' 
    #         #append_spikes += f'        print(min(dv_d{var_base}_tracker))\n'

    # # for name, value in parameters.items():
    # #     if "_gSYN" in name and "R2Off" not in name:
            
    # #         post_node = name.split('_',-1)[0]
    # #         pre_node = name.split('_',-1)[1]

    # #         if "R1On_" in name and "_On" in name:

    # #             append_spikes += f'    print(\'first few spikes\')\n'
    # #             append_spikes += f'    print(voltage_derivative[0:15])\n'
    # #             append_spikes += f'    print(psc_derivative[0:15])\n'

    # #             append_spikes += f'    print(\'maximums\')\n'
    # #             append_spikes += f'    print(max(voltage_derivative))\n'
    # #             append_spikes += f'    print(max(psc_derivative))\n'
    # #             append_spikes += f'    print(min(voltage_derivative))\n'
    # #             append_spikes += f'    print(min(psc_derivative))\n'
            

    # generated_code = generated_code + append_spikes

    
    #----Return Statement

    # return_statement = "\n    return R2On_V_spikes"

    # #Package gradients
    # count_p2 = 0
    # for name, value in parameters.items():
    #     if "gSYN" in name and "R2Off" not in name:

    #         post_node = name.split('_',-1)[0]
    #         pre_node = name.split('_',-1)[1]

    #         #print(count_p2)


    #         if count_p2 == 0:
    #             return_statement += f', [dGSYN{post_node}_{pre_node}'
    #         elif count_p2 == 9:
    #             return_statement += f', dGSYN{post_node}_{pre_node}]'
    #         else:
    #             return_statement += f', dGSYN{post_node}_{pre_node}'

    #         #print(return_statement)

    #         count_p2 += 1

    # return_statement += '\n'

    # generated_code = generated_code + return_statement

    #----Training Loop

    # training_loop = textwrap.dedent(f"""\
    # \ndef main():
        
        
    #     #Set epochs and parameter initialization
    #     num_epochs = 1
    #     p = np.array([1,1,1,1,1,1,1,1,1,1])*0.025

    #     #Initilze Adam Parameters
    #     m = np.zeros((10))
    #     v = np.zeros((10))
    #     beta1, beta2 = 0.999, 0.99995   #Nominally 0.92, 0.9995
    #     eps = 1e-6
    #     t = 0
    #     lr = 1e-3

    #     #Adjust length of training signal
    #     scale_factor = 1

    #     #Keep track for plotting
    #     losses = []
    #     param_tracker = []

    #     for epoch in range(num_epochs):

    #         #Track Parameters
    #         param_tracker.append(p)
            
    #         #Run forwards pass
    #         output, grads = forwards(p,scale_factor)  # forward pass

    #         #Extract gradients
    #         grad_holder2 = []
    #         for z in grads:
    #             grad_holder2.append(float(z[0][0]))

    #         #grads = [float(x) for x in grad_holder2] 

    #         #Calcualtes loss functions
    #         #---
    #         #Current functions:
    #         #    - Firing Rate L2 ("fr")
    #         #    - PSTH L2 ("PSTH")
    #         #    - Spike L2 Distance /WIP
    #         #    - van Rossum Distance (Spike Level) /WIP

    #         out_grad, loss = Calc_output_grad.calculate(output, grads, scale_factor, "spikeL2")

    #         #Calculate parameter updates using Adam Optimizer
    #         #---
    #         #Uses 2 terms to control the momementum of the learning
    #         #    -beta1 controlls short term momentum
    #         #    -beta2 contorlls long term dampening

    #         m, v, p, t = Update_params.adam_update(m, v, p, t, beta1, beta2, lr, eps, out_grad)

    #         losses.append(loss)
    #         print(f"Epoch {{epoch}}: L2 Loss = {{loss[0]}}: Vr Loss = {{loss[1]}}",flush=True) 
    #         #print(f"Epoch {{epoch}}: Loss = {{loss}}",flush=True) 

    #     return losses, output, param_tracker

       
    # """)

    # generated_code = generated_code + training_loop
        
    
    #--------------------------------------------------------         Clean up and port to .py        --------------------------------------------------------------------------#
   
    generated_code = Clean_up.Clean_gen_code(generated_code)

    with open("generated2.py", "w") as f:
        f.write(generated_code)

    print("generated2.py has been created.")

    return generated_code
