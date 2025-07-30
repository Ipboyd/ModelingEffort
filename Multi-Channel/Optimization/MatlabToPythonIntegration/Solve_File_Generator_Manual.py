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

#Recursion definition for tracking gradient paths
def recurse(cur_node, tracker,nodes,edges):

  tracker.append(cur_node)

  for k in edges:
    if cur_node == k.split('->', -1)[1]:
      recurse(k.split('->', -1)[0],tracker,nodes,edges)
    

  return tracker



def build_ODE(parameters):

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
        def forwards(ps,scale_factor):
    """)

    #Bring In params
    params = '\n    #Params\n'
    count_ps = 0
    for name, value in parameters.items():
        if "_gSYN" in name and "R2Off" not in name:
            params += f"    {name} = ps[{count_ps}]\n"
            #params += f"    {name} = ps[[0]]\n"
            #params += f"    print(ps[0][{count_ps}])\n"
            count_ps += 1
        else:
            params += f"    {name} = {value}\n"

    
    #Initialize grads
    for name, value in parameters.items():
        if "_gSYN" in name and "R2Off" not in name:
            
            post_node = name.split('_',-1)[0]
            pre_node = name.split('_',-1)[1]

            params += f'    dGSYN{post_node}_{pre_node} = 0\n'

    params += f'    psc_derivative = []\n'
    params += f'    voltage_derivative = []\n'

    #Bring in fixed params
    fixed_param_declaration = '\n    #Fixed Param Declaration\n'
    for k in range(len(lhs_list)):
        if (lhs_list[k] != 'On_On_IC_input' and lhs_list[k] != 'Off_Off_IC_input'):
            fixed_param_declaration += f"    {lhs_list[k]} = {rhs_list[k]}\n"

    #Bring in T, helper, and grad
    #Mind the 2*dt line
    #Tspan reports the length of the stimulus in ms. np.arange is 0 index and exclusive which requires 1 more index. 
    #7/28 Not using t-1 anymore? Can we just push the sim from 0 to T instea of 1 to T?
    T_and_Helper_declaration = '\n    T = len(np.arange(tspan[0],tspan[1]+(dt),dt))\n    helper = np.arange(tspan[0],tspan[1]+(dt),dt)\n'

    #Add Gradients
    for k in range(len(update_vars)):
        if "Off_V" in update_vars[k] or "On_V" in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            var_name = var[:-5]
            T_and_Helper_declaration += f'    grad_{var_base} = 0\n'

   
    #Bring in Spikes Holders
    spike_holder_string = '\n    #Spikes Holders\n'
    for k in range(len(monitor_vars)):
        if "V_spikes" in monitor_vars[k]:
            spike_holder_string += f"    {monitor_vars[k]} = []\n"
    
    generated_code = import_string + forwards_loop_header + params + fixed_param_declaration + T_and_Helper_declaration + spike_holder_string
    

    #----Initilize variables that get reset per trial

    trial_loop_declaration = '\n    for trial_number in range(10):\n'

    #Bring in State vars
    state_vars_string = '\n        #State Variable Declaration\n'
    for k in range(len(state_vars)):
        state_vars_string += f"        {state_vars[k]} = {State_variable_Identifier.replace_ones_zeros(state_vals[k])}\n"

    #Bring in Monitors
    monitor_string = '\n        #Monitor Declaration\n'
    for k in range(len(monitor_vars)):
        if "V_spikes" in monitor_vars[k]:
            monitor_string += f"        {monitor_vars[k]}_holder = []\n"
        else:  
            monitor_string += f"        {monitor_vars[k]} = {monitor_vals[k]}\n"

    #Declare Inputs
    inputs_header = "\n        #Delcare Inputs\n        On_On_IC_input = genPoissonInputs.gen_poisson_inputs(trial_number,On_On_IC_locNum,On_On_IC_label,On_On_IC_t_ref,On_On_IC_t_ref_rel,On_On_IC_rec,scale_factor)\n        Off_Off_IC_input = genPoissonInputs.gen_poisson_inputs(trial_number,Off_Off_IC_locNum,Off_Off_IC_label,Off_Off_IC_t_ref,Off_Off_IC_t_ref_rel,Off_Off_IC_rec,scale_factor)\n"

    generated_code = generated_code + trial_loop_declaration + state_vars_string + monitor_string + inputs_header


    #----ODE Intermost loop
    
    ODE_loop_Declaration = '\n        for t in range(0,T):\n'

    #ODE declarations
    ode_string = '\n            #ODEs\n'
    for k in range(len(pairs)):
        rhs_ode = FormatODEs_Ns.reformat_input_time_indexing(FormatODEs_Ns.reformat_discrete_time_indexing(pairs[k][1]))
        rhs_ode_rpl = rhs_ode.replace("[t-1]", "[-1]")
        ode_string += f"            {pairs[k][0]} = {rhs_ode_rpl}\n"
        
   
    #Update Eulers
    update_eulers = '\n            #Update Eulers\n'

    for k in range(len(update_vars)):
        rep_val = update_vals[k].replace("[t-1]", "[-1]")
        update_eulers += f"            {update_vars[k][:-3]}[-2] = {update_vars[k][:-3]}[-1]\n"
        update_eulers += f"            {update_vars[k][:-3]}[-1] = {rep_val}\n"

    #Spiking Behavior
    test1_string = '\n            #Spiking and conditional actions\n'
    
    #Test 1 (Spiking activity)
    for k in range(len(update_vars)):
        if "_V[t]" in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            var_name = var[:-5]
            var_prev = var.replace("[t]", "[-2]")
            var_thresh = f"{var.replace('[t]', '_thresh')}"
            test1_string += f"            {var_base}_spikes_holder.append(int((({var_base}[-1] >= {var_thresh}) and ({var_prev} < {var_thresh}))))\n"
            test1_string += f"            if {var_base}_spikes_holder[-1]:\n"       
            test1_string += f"                {var_name}_tspike[int({var_name}_buffer_index)-1] = helper[t]\n"
            test1_string += f"                {var_name}_buffer_index = ({var_name}_buffer_index % 5) + 1\n"
            

    #Test 2 (Voltage reset and adaptation) 
    test2_string = '\n                #Voltage reset and adaptation\n'
    for k in range(len(update_vars)):
        if "_V[t]" in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            var_name = var[:-5]
            var_thresh = f"{var.replace('[t]', '_thresh')}"
            var_reset = f"{var.replace('[t]', '_reset')}"
            var_adapt = f"{var.replace('V[t]', 'g_ad[t]')}"
            var_inc = f"{var.replace('V[t]', 'g_inc')}"
            test2_string += f"            {var_base}_test2a = {var_base}[-1] > {var_thresh}\n"
            test2_string += f"            if {var_base}_test2a:\n"
            test2_string += f"                {var[:-3]}[-2] = {var[:-3]}[-1] \n"
            test2_string += f"                {var[:-3]}[-1] = {var_reset} \n"
            test2_string += f"                {var_adapt[:-3]}[-2] = {var_adapt[:-3]}[-1]\n"
            test2_string += f"                {var_adapt[:-3]}[-1] = {var_adapt[:-3]}[-1] + {var_inc}\n"
            test2_string += f"            {var_base}_test2b = np.any(helper[t] <= {var_name}_tspike + {var_name}_t_ref)\n"
            test2_string += f"            if {var_base}_test2b:\n"
            test2_string += f"                {var[:-3]}[-2] = {var[:-3]}[-1]\n"
            test2_string += f"                {var[:-3]}[-1] = {var_reset}\n"

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

        test3_string += f"            {var_base}_test3 = np.any(helper[t] == {statement_pairs[k][0]} + {statement_pairs[k][1]})\n"  
        test3_string += f"            if {var_base}_test3:\n"  
        test3_string += f"                {var_x}[-2] = {var_x}[-1]\n"
        test3_string += f"                {var_q}[-2] = {var_F}[-1]\n"
        test3_string += f"                {var_F}[-2] = {var_F}[-1]\n"
        test3_string += f"                {var_P}[-2] = {var_P}[-1]\n"
        test3_string += f"                {var_x}[-1] = {var_x}[-1] + {var_q}[-1]\n"
        test3_string += f"                {var_q}[-1] = {var_F}[-1] * {var_P}[-1]\n"
        test3_string += f"                {var_F}[-1] = {var_F}[-1] + {var_fF}*({var_max}-{var_F}[-1])\n"
        test3_string += f"                {var_P}[-1] = {var_P}[-1] * (1 - {var_fP})\n"

    generated_code = generated_code + ODE_loop_Declaration + ode_string + update_eulers + test1_string + test2_string + test3_string
    
    #----Gradient Calculations

    #Going to try to build in gradient to forwards loop
    #Build out gradients
    grad_string = '\n            #Grad Calculations\n'


    #Grab all of the valid voltage update ones. This is the spiking deriavte wrt the voltage
    grad_string += '\n\n            #Surrogate Spike Related Derivates\n'
    for k in range(len(update_vars)):
        if "_V" in update_vars[k] and ("R" in update_vars[k] or "S" in update_vars[k]) and "Noise" not in update_vars[k] and "R2Off" not in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            grad_string += f'            dspike_d{var_base} = (((10*np.exp(-(0.1)*({var_base}[-1] - {var_base}_thresh)))/(1+np.exp(-(0.1)*({var_base}[-1] - {var_base}_thresh)))**2))/500\n' 
            #grad_string += f'            dv_d{var_base}_tracker.append(dspike_d{var_base})\n'

    #Search for all of the gsyns that we want to update
    #Write all of the partials of the voltage w.r.t the parameters
    grad_string += '\n\n            #PSC & Parameter Related Derivates\n'
    for name, value in parameters.items():
        if "_gSYN" in name and "R2Off" not in name:
            
            post_cell = name.split('_', -1)[0]
            pre_cell = name.split('_', -1)[1]

            #grad_string += f'            print(helper[t])\n'

            grad_string += f'            dv_d{name} = -(dt*{post_cell}_R*{post_cell}_{pre_cell}_PSC_s[-1]*{post_cell}_{pre_cell}_PSC_netcon*({post_cell}_V[-1]-{post_cell}_{pre_cell}_PSC_ESYN)/{post_cell}_tau)/15\n'
            #grad_string += f'            dv_d{name} = -dt*{post_cell}_R*{post_cell}_{pre_cell}_PSC_netcon*({post_cell}_V[-1]-{post_cell}_{pre_cell}_PSC_ESYN)/{post_cell}_tau\n'
            
            #if "R2On" in name and "R1On" in name:

            
            #grad_string += f'            print(dv_d{name})\n'
            
            grad_string += f'            d{post_cell}_{pre_cell}_PSC_dUk = -((dt*{post_cell}_{pre_cell}_PSC_scale*2*({post_cell}_{pre_cell}_PSC_x[-1]+{post_cell}_{pre_cell}_PSC_q[-1])/{post_cell}_{pre_cell}_PSC_tauR)*helper[t]*sum((({pre_cell}_tspike+{post_cell}_{pre_cell}_PSC_delay)-helper[t])*np.exp(-1*(({pre_cell}_tspike+{post_cell}_{pre_cell}_PSC_delay)-helper[t])**2)))/2500\n'
            #grad_string += f'            d{post_cell}_{pre_cell}_PSC_dUk = -(dt*{post_cell}_{pre_cell}_PSC_scale*2/{post_cell}_{pre_cell}_PSC_tauR)*helper[t]*sum((({pre_cell}_tspike+{post_cell}_{pre_cell}_PSC_delay)-helper[t])*np.exp(-1*(({pre_cell}_tspike+{post_cell}_{pre_cell}_PSC_delay)-helper[t])**2))\n'
            
            #grad_string += f'            print(d{post_cell}_{pre_cell}_PSC_dUk)\n'
            grad_string += f'            dv_d{post_cell}_{pre_cell}_PSC = -(dt*{post_cell}_R*{name}*{post_cell}_{pre_cell}_PSC_netcon*({post_cell}_V[-1]-{post_cell}_{pre_cell}_PSC_ESYN)/{post_cell}_tau)/10\n'
            #grad_string += f'            print(dv_d{post_cell}_{pre_cell}_PSC)\n'

            #print('cells')
            #print(pre_cell)
            #print(post_cell)

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
   
    nodes = ['R2On','R1On','On','Off','S2OnOff','S1OnOff','R1Off','R2Off']
    edges = ['On->R1On','On->S1OnOff','Off->R1Off','Off->S1OnOff','S1OnOff->R1On','S1OnOff->R1Off','R1On->R2On','R1On->S2OnOff','R1Off->R2Off','R1Off->S2OnOff','S2OnOff->R2On','S2OnOff->R2Off']

    #Preform depth first search to get a list of all paths
    all_paths = recurse('R2On',[],nodes,edges)
    print(all_paths)

    path_holder = []
    path = ''
    gate = 0
    gate2 = 0


    for k in range(len(all_paths)-1):

        #Compare with the eddges
        for m in range(len(edges)):
            if edges[m].split('->', -1)[0] == all_paths[k+1] and edges[m].split('->', -1)[1] == all_paths[k]:
                path = edges[m] + '->' + path
                gate = 1    

        #If you have gone through all of them and the gate does not get flipped then append
        #print(gate)
        if gate == 0:
            path_holder.append(path[:-2])
            for z in range(k):
                for m in range(len(edges)):
                    if edges[m].split('->', -1)[0] == all_paths[k+1] and edges[m].split('->', -1)[1] == all_paths[k-(z+1)]:

                        #1. Search backwards to find the node that it connects to. Done
                        #2. Search through the path and replace in place the new connection
          
                        path_segmented = path.split('->',-1)

                        for count, n in enumerate(path_segmented):
       
                            if n == edges[m].split('->', -1)[1]:

                                path = path.split('->',count+1)[count+1]
                                path = edges[m] + '->' + path
                                if k == len(all_paths)-2 and gate2 == 0:
                                     gate2 = 1
                                     path_holder.append(path[:-2])
                                break

                        break

        gate = 0

    
    grad_string += '\n            #Build derivs\n'
    #1 Iterate through all of the gsyns that we are looking at.
    for name, value in parameters.items():
        if "gSYN" in name and "R2Off" not in name:
            #2. Look through all of the paths and match the names to the paths

            cur_divs = []

            #print(name)

            for cur_path in path_holder:
                #Parameter name
                post_node = name.split('_',-1)[0]
                pre_node = name.split('_',-1)[1]

                #Path (Going to iterate through them just to qualify that the gsyn can only be updated by that synapse)
                path_segements = cur_path.split('->',-1)

                path_gate = 0
                check = 0

                #print(cur_path)

                for ps in range(len(path_segements)):

                    #print(path_segmented)

                    #print(path_segmented[ps])
                    #print(pre_node)

                    if path_segements[ps] == pre_node:

                        

                        path_gate = 1

                    if path_gate == 1:
                        


                        path_gate = 0


                        if path_segements[ps+1] == post_node:
                            check = 1
                            path_gate = 0

                if check == 1:


                    deriv = ''



                    for count_tar in range(int(len(path_segements)/2)):
                        pre_der = path_segements[-(count_tar*2 + 2)]
                        post_der = path_segements[-(count_tar*2 + 1)]

    
                        #3. Add the spiking derivate
                        deriv += 'dspike_d' + post_der + '_V*'

                        if pre_der == pre_node:
                            deriv += 'dv_d' + post_der + '_' + pre_der + '_PSC_gSYN'
                            break
                        else:
                            deriv += 'dv_d' + post_der + '_' + pre_der + '_PSC*' + 'd' + post_der + '_' + pre_der + '_PSC_dUk*'

                    cur_divs.append(deriv)

                    
            
            grad_string += '            dGSYN' + post_node + '_' + pre_node +' += ' 
            for un_divs in np.unique(cur_divs):
                grad_string += f'{un_divs}+'
            grad_string = grad_string[:-1]
            grad_string += '\n'
            #grad_string += '            dGSYN' + post_node + '_' + pre_node +' = '
            #
            # grad_string += f'            if dGSYNR1On_On != 0:\n                voltage_derivative.append(dspike_dR2On_V*dv_dR2On_R1On_PSC*dR2On_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_On_PSC_gSYN)\n'
            # grad_string += f'            if dGSYNR1On_On != 0:\n                psc_derivative.append(dspike_dR2On_V*dv_dR2On_S2OnOff_PSC*dR2On_S2OnOff_PSC_dUk*dspike_dS2OnOff_V*dv_dS2OnOff_R1On_PSC*dS2OnOff_R1On_PSC_dUk*dspike_dR1On_V*dv_dR1On_On_PSC_gSYN)\n'   

            # #print(cur_divs)

                            

                        




    generated_code = generated_code + grad_string


    #----Post loop appending

    #Append spikes
    append_spikes = '\n        #Append Spikes\n'
        
    for k in range(len(monitor_vars)):
        if "V_spikes" in monitor_vars[k]:    
            append_spikes += f'        {monitor_vars[k]}.append({monitor_vars[k]}_holder)\n'
            #append_spikes += f'        print(max({monitor_vars[k]}_holder))\n'

    for k in range(len(update_vars)):
        if "_V" in update_vars[k] and ("R" in update_vars[k] or "S" in update_vars[k]) and "Noise" not in update_vars[k] and "R2Off" not in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            
            #append_spikes += f'        print(max(dv_d{var_base}_tracker))\n' 
            #append_spikes += f'        print(min(dv_d{var_base}_tracker))\n'

    # for name, value in parameters.items():
    #     if "_gSYN" in name and "R2Off" not in name:
            
    #         post_node = name.split('_',-1)[0]
    #         pre_node = name.split('_',-1)[1]

    #         if "R1On_" in name and "_On" in name:

    #             append_spikes += f'    print(\'first few spikes\')\n'
    #             append_spikes += f'    print(voltage_derivative[0:15])\n'
    #             append_spikes += f'    print(psc_derivative[0:15])\n'

    #             append_spikes += f'    print(\'maximums\')\n'
    #             append_spikes += f'    print(max(voltage_derivative))\n'
    #             append_spikes += f'    print(max(psc_derivative))\n'
    #             append_spikes += f'    print(min(voltage_derivative))\n'
    #             append_spikes += f'    print(min(psc_derivative))\n'
            

    generated_code = generated_code + append_spikes

    
    #----Return Statement

    return_statement = "\n    return R2On_V_spikes"

    #Package gradients
    count_p2 = 0
    for name, value in parameters.items():
        if "gSYN" in name and "R2Off" not in name:

            post_node = name.split('_',-1)[0]
            pre_node = name.split('_',-1)[1]

            #print(count_p2)


            if count_p2 == 0:
                return_statement += f', [dGSYN{post_node}_{pre_node}'
            elif count_p2 == 9:
                return_statement += f', dGSYN{post_node}_{pre_node}]'
            else:
                return_statement += f', dGSYN{post_node}_{pre_node}'

            #print(return_statement)

            count_p2 += 1

    return_statement += '\n'

    generated_code = generated_code + return_statement

    #----Training Loop

    training_loop = textwrap.dedent(f"""\
    \ndef main():
        
        
        #Set epochs and parameter initialization
        num_epochs = 150
        p = np.array([1,1,1,1,1,1,1,1,1,1])*0.025

        #Initilze Adam Parameters
        m = np.zeros((10))
        v = np.zeros((10))
        beta1, beta2 = 0.92, 0.9995
        eps = 1e-6
        t = 0
        lr = 1e-3

        #Adjust length of training signal
        scale_factor = 1

        #Keep track for plotting
        losses = []
        param_tracker = []

        for epoch in range(num_epochs):

            #Track Parameters
            param_tracker.append(p)
            
            #Run forwards pass
            output, grads = forwards(p,scale_factor)  # forward pass

            #Extract gradients
            grad_holder2 = []
            for z in grads:
                grad_holder2.append(float(z[0][0]))

            #grads = [float(x) for x in grad_holder2] 

            #Calcualtes loss functions
            #---
            #Current functions:
            #    - Firing Rate L2 ("fr")
            #    - PSTH L2 ("PSTH")
            #    - Spike L2 Distance /WIP
            #    - van Rossmum Distance (Spike Level) /WIP

            out_grad, loss, vr_ex = Calc_output_grad.calculate(output, grads, scale_factor, "vanRossum")

            #Calculate parameter updates using Adam Optimizer
            #---
            #Uses 2 terms to control the momementum of the learning
            #    -beta1 controlls short term momentum
            #    -beta2 contorlls long term dampening

            m, v, p, t = Update_params.adam_update(m, v, p, t, beta1, beta2, lr, eps, out_grad)

            losses.append(loss)
            print(f"Epoch {{epoch}}: Loss = {{loss}}",flush=True) 

        return losses, output, param_tracker, vr_ex

       
    """)

    generated_code = generated_code + training_loop
        
    
    #--------------------------------------------------------         Clean up and port to .py        --------------------------------------------------------------------------#
   
    generated_code = Clean_up.Clean_gen_code(generated_code)


    with open("generated2.py", "w") as f:
        f.write(generated_code)

    print("generated2.py has been created.")

    return generated_code
