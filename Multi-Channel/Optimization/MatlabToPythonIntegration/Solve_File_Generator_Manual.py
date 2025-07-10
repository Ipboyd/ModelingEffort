import os
import Parser
import State_Parser
import FormatODEs_Ns
import State_variable_Identifier
import Extract_Fixed_vars
import Clean_up
import ConditionalActions
import add_device_to_tensors
import pdb
import textwrap


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

    """)


    #----Declare forwards loop & Initialize variables
    forwards_loop_header = textwrap.dedent("""\
        def forwards(p_Ron):
    """)

    #Bring In params
    params = '\n    #Params\n'
    for name, value in parameters.items():
        params += f"    {name} = {value}\n"

    #TEMP
    for name, value in parameters.items():
        if "_gSYN" in name and "R2Off" not in name:
            params += f"    dv_d{name}_tracker = []\n"

    #Bring in fixed params
    fixed_param_declaration = '\n    #Fixed Param Declaration\n'
    for k in range(len(lhs_list)):
        if (lhs_list[k] != 'On_On_IC_input' and lhs_list[k] != 'Off_Off_IC_input'):
            fixed_param_declaration += f"    {lhs_list[k]} = {rhs_list[k]}\n"

    #Bring in T, helper, and grad
    T_and_Helper_declaration = '\n    T = len(np.arange(tspan[0],tspan[1],dt))\n    helper = np.arange(tspan[0],tspan[1],dt)\n'

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
    inputs_header = "\n        #Delcare Inputs\n        On_On_IC_input = genPoissonInputs.gen_poisson_inputs(trial_number,On_On_IC_locNum,On_On_IC_label,On_On_IC_t_ref,On_On_IC_t_ref_rel,On_On_IC_rec)\n        Off_Off_IC_input = genPoissonInputs.gen_poisson_inputs(trial_number,Off_Off_IC_locNum,Off_Off_IC_label,Off_Off_IC_t_ref,Off_Off_IC_t_ref_rel,Off_Off_IC_rec)\n"

    generated_code = generated_code + trial_loop_declaration + state_vars_string + monitor_string + inputs_header


    #----ODE Intermost loop
    
    ODE_loop_Declaration = '\n        for t in range(1,T):\n'

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

    #Search for all of the gsyns that we want to update
    #Write all of the partials of the voltage w.r.t the parameters
    for name, value in parameters.items():
        if "_gSYN" in name and "R2Off" not in name:
            
            post_cell = name.split('_', -1)[0]
            pre_cell = name.split('_', -1)[1]

            grad_string += f'            dv_d{name} = -dt*{post_cell}_R*{post_cell}_{pre_cell}_PSC_s[-1]*{post_cell}_{pre_cell}_PSC_netcon[-1]*({post_cell}_V[-1]-{post_cell}_{pre_cell}_PSC_ESYN)/{post_cell}_tau\n'
            grad_string += f'            d{post_cell}_{pre_cell}_PSC_dUk = -(dt*{post_cell}_{pre_cell}_PSC_scale*2*({post_cell}_{pre_cell}_PSC_x[-1]+{post_cell}_{pre_cell}_PSC_q[-1])/{post_cell}_{pre_cell}_PSC_tauR)*helper[t]*sum((({pre_cell}_tspike+{post_cell}_{pre_cell}_PSC_delay)-helper[t])*np.exp(-1*(({pre_cell}_tspike+{post_cell}_{pre_cell}_PSC_delay)-helper[t])**2))\n'
            grad_string += f'            dv_d{post_cell}_{pre_cell}_PSC = -dt*{post_cell}_R*{name}*{post_cell}_{pre_cell}_PSC_netcon[-1]*({post_cell}_V[-1]-{post_cell}_{pre_cell}_PSC_ESYN)/{post_cell}_tau\n'
            
            
            #grad_string += f'            dv_d{name}_cur_grad = dv_d{name}*d{post_cell}_{pre_cell}_PSC_dUk'

            grad_string += f'            dv_d{name}_tracker.append(dv_d{post_cell}_{pre_cell}_PSC)\n'
            
            
            
            #var = update_vars[k]
            #var_base = var[:-3]
            #var_name = var[:-5]
            
            #Old Grads
            #grad_string += f'            Alpha_vt = -((({var_base}_reset*-1)+{var_base}[-1]))/(1+np.exp(6*(helper[t]-max({var_name}_tspike + {var_name}_t_ref)))) + {var_base}[-1]\n' 
            #grad_string += f'            dvt1_dp = dt*{var_name}_R*R2On_R1On_PSC_netcon*(((Alpha_vt+{var_base}_reset*-1)/(1+np.exp(6*(Alpha_vt+{var_base}_thresh*-1))))+{var_base}_reset-R2On_R1On_PSC_ESYN)/{var_name}_tau\n'
            #grad_string += f'            dudp = ((np.exp(-1*((dvt1_dp) - {var_base}_thresh)))/(1+np.exp(-1*((dvt1_dp) - {var_base}_thresh)))**2)\n'
            

            #Attempting to skip some gradients.
            #grad_string += f'            dudp_{var_base} = ((np.exp(-1*(({var_base}[-1]) - {var_base}_thresh)))/(1+np.exp(-1*(({var_base}[-1]) - {var_base}_thresh)))**2)\n'

            #grad_string += f'            grad_{var_base} += dudp_{var_base}\n'

    generated_code = generated_code + grad_string


    #----Post loop appending

    #Append spikes
    append_spikes = '\n        #Append Spikes\n'
        
    for k in range(len(monitor_vars)):
        if "V_spikes" in monitor_vars[k]:    
            append_spikes += f'        {monitor_vars[k]}.append({monitor_vars[k]}_holder)\n'
            #append_spikes += f'        print(max({monitor_vars[k]}_holder))\n'

    for name, value in parameters.items():
        if "_gSYN" in name and "R2Off" not in name:
            append_spikes += f'        print(max(dv_d{name}_tracker))\n'
            append_spikes += f'        print(min(dv_d{name}_tracker))\n'
            

    generated_code = generated_code + append_spikes

    
    #----Return Statement

    return_statement = "\n    return R2On_V_spikes"

    #Package gradients
    for k in range(len(update_vars)):
        if "Off_V" in update_vars[k] or "On_V" in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]

            if k == 0:
                return_statement += f', [grad_{var_base}'
            if k == 7:
                return_statement += f', grad_{var_base}]'
            else:
                return_statement += f', grad_{var_base}'

    return_statement += '\n'

    #generated_code = generated_code + return_statement

    #----Training Loop

    training_loop = textwrap.dedent(f"""\
    \ndef main():
        num_epochs = 1

        p = ones((1,1))*0.005   #Initial parameter value

        #Adam
        m = 0
        v = 0
        beta1, beta2 = 0.92, 0.9995
        eps = 1e-6
        t = 0
        lr = 1e-3

        matfile_path = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting"
        filename = f"{{matfile_path}}/goalPSTH.mat"
        data = scipy.io.loadmat(filename)


        #target_spikes = np.array(data['ans'][0])
        target_spikes = 10
    
        losses = []
        param_tracker = []

        for epoch in range(num_epochs):

            output, grads = forwards(p)  # forward pass
            
            param_tracker.append(p)

            print(f'parameter = {{p}}')

            print(np.shape(output))
            output = np.reshape(output,(1,34998*10))

            print('Avg Firing Rate')
            print(output.sum()/10/3)

            fr = output.sum()/10/3  #total spikes/num_trials/num_seconds

            #print(target_spikes)

            loss = (target_spikes-fr)**2


            out_grad = 2*(fr-target_spikes)*grads

            print('grad below')
            print(out_grad)

            p = Update_Grads.grads_update(grads,p)

            
                
                

            print('p below')
            print(p)

            #loss = ((binned_counts - target_spikes)**2).mean()
            losses.append(loss)
            

            print(f"Epoch {{epoch}}: Loss = {{loss.item()}}",flush=True) 

        return losses, output, param_tracker

       
    """)

    generated_code = generated_code + training_loop
        
    
    #--------------------------------------------------------         Clean up and port to .py        --------------------------------------------------------------------------#
   
    generated_code = Clean_up.Clean_gen_code(generated_code)


    with open("generated2.py", "w") as f:
        f.write(generated_code)

    print("generated2.py has been created.")

    return generated_code
