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

def build_ODE(parameters):


    #Create spiking handler class

    SurrogateSpiking_Class_declaration = """
import torch
import torch.nn as nn
import genPoissonTimes
import genPoissonInputs
import matplotlib.pyplot as plt
import pdb
from memory_profiler import profile
import gc


#torch.autograd.set_detect_anomaly(True)


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, prev, threshold):
        ctx.save_for_backward(input)
        #if((input >= threshold) and (prev < threshold)):
        #print(((input >= threshold) and (prev < threshold)).float())
        return ((input >= threshold) and (prev < threshold)).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (1.0 / (1.0 + torch.abs(input)) ** 2)
        return grad_input, None, None

"""

    # Header
    main_class_declaration = """

class LIF_ODE(nn.Module):
    def __init__(self):
        super().__init__()
        
        

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(self.device)
        #print(trial_num)

        # Learnable Parameters

"""

    

    #Need to separate out the learnable and non-learnable parameters
    #Right now we are just leraning gsyns so separate those out
    #There are several ways we could go about seperating things out
    #1. just look for gsyn or for the name of the parameters
    #2. accept an array of values that correspond to learnable/nonlearnable params that we take
        #a. Going to try #1 for now

    learnable_block = ''
    nonlearnable_block = '\n        # Non-learnable Parameters\n'



    for name, value in parameters.items():
        
        if 'npop' in name.lower():
            nonlearnable_block += f"        self.{name} = int({value})\n"
        else:
            if 'gsyn' in name.lower():
                learnable_block += f"        self.{name} = nn.Parameter(torch.tensor({value}, dtype=torch.float32))\n"
            elif 'label' in name.lower():
                nonlearnable_block += f"        self.{name} = {value}\n"
            else:
                nonlearnable_block += f"        self.{name} = torch.tensor({value}, dtype=torch.float32)\n"
        
    


    #Here we build the ODES
    #A couple of things (which I will check)
    #The state variables should update automatcially we shouldn't have to include that update
    #The shunting should be covered by the above block (Might need to add the shunting cooldown and what not)


    #1. Find the file path. Going to set to this to single channel model for now.
    home_dir = os.path.expanduser("~")  # Gets the current user's home directory
    file_path = os.path.join(home_dir, "Documents", "GitHub", "ModelingEffort", 
                         "Single-Channel", "Model", "Model-Core", 
                         "Model-Main", "run", "1-channel-paper", "solve", "solve_ode_1_channel_paper.m") #This could be changed for TD and multichannel and what not.


    fixed_param_declaration = '\n        T = len(torch.arange(self.tspan[0],self.tspan[1],self.dt, dtype=torch.float32))\n        #print(trial_num)\n\n        # Fixed Params\n'

    #print('made it here')

    #Add the calculable parameters
    fixed_params = Extract_Fixed_vars.extract_fixed_variables_from_block(file_path)
    
    #Get the left hadn and right hand side of these equations.
    lhs_list, rhs_list = zip(*fixed_params)

   
    for k in range(len(lhs_list)):
        if (lhs_list[k] != 'On_On_IC_input' or lhs_list[k] != 'Off_Off_IC_input'):
            fixed_param_declaration += f"        self.{lhs_list[k]} = {State_variable_Identifier.add_self_prefix(State_variable_Identifier.add_self_prefix(rhs_list[k],lhs_list),parameters.keys())}\n"
    
    #print(fixed_param_declaration)
    

    #Put in forwards header
    forwards_declaration = """
    @profile
    def forward(self):
        
        #State Variables
            
        T = len(torch.arange(self.tspan[0],self.tspan[1],self.dt, dtype=torch.float32))
        helper = torch.arange(self.tspan[0],self.tspan[1],self.dt, dtype=torch.float32)

        print('hereT')
        print(T)
        
"""
    monitor_string = '\n\n        #Monitors\n\n'
    

    #Add in the necessary monitors so that the conditional actions function
    monitor_response = State_Parser.extract_monitor_declarations(file_path)
    monitor_vars = list(monitor_response.keys())
    monitor_vals = [list(v.values())[0] for v in monitor_response.values()]

    for k in range(len(monitor_vars)):
        if "V_spikes" in monitor_vars[k]:
            monitor_string += f"        {monitor_vars[k]} = []\n"
        
        

    ode_string = f'''\n\n        #ODEs
\n        #spike_holderOn = torch.full((T-1,),0.0)
\n        #spike_holderOff = torch.full((T-1,),0.0)
\n        #spike_holderR1On = torch.full((T-1,),0.0)
\n        spike_holderR2On = torch.full((T-1,),0.0)
\n        #spike_holderR2Off = torch.full((T-1,),0.0)
\n        #spike_holderR1Off = torch.full((T-1,),0.0)
\n        #spike_holderS2OnOff = torch.full((T-1,),0.0)
\n        #spike_holderS1OnOff = torch.full((T-1,),0.0)
\n        for num_trials_count in range(2):
\n            #print('made it here')\n'''
    
    
    for k in range(len(monitor_vars)):
        if "V_spikes" in monitor_vars[k]:
            ode_string += f"            {monitor_vars[k]}_holder = []\n"
        else:  
            ode_string += f"            {monitor_vars[k]} = {State_variable_Identifier.add_self_prefix(monitor_vals[k],parameters.keys())}\n"

    
    

    #Add in the state variables
    #This code seems very redundent. Clean if state_vars or state_vals are not used somewhere
    #else at some point

    both_sides = State_Parser.extract_state_variables_from_block(file_path)


    state_vars = list(both_sides.keys())
    state_vals = [list(v.values())[0] for v in both_sides.values()]

    
    for k in range(len(state_vars)):
        ode_string += f"            {state_vars[k]} = {State_variable_Identifier.add_self_prefix(State_variable_Identifier.replace_ones_zeros(state_vals[k]),parameters.keys())}\n"

    ode_string += "\n\n       "


    #ode_string += '''     print('made it here3')\n'''


    ode_string = ode_string + '''
            
            #Delcare Inputs
            self.On_On_IC_input = genPoissonInputs.gen_poisson_inputs(num_trials_count,self.On_On_IC_locNum,self.On_On_IC_label,self.On_On_IC_t_ref,self.On_On_IC_t_ref_rel,self.On_On_IC_rec)
            self.Off_Off_IC_input = genPoissonInputs.gen_poisson_inputs(num_trials_count,self.Off_Off_IC_locNum,self.Off_Off_IC_label,self.Off_Off_IC_t_ref,self.Off_Off_IC_t_ref_rel,self.Off_Off_IC_rec)
\n            for t in range(1,T):
                #print('hello2')\n\n'''
    

    pairs = Parser.extract_rhs_lhs(file_path)

    


    #Loop through and fill in the ODES

    #Need to come up with an effecient way of formating these ODEs in the format that pytorch will accept

    #The following loop:
    #1. Writes the ODes to matlab
    #2. Goes through and remove all of the (n-1,:)
    #3. Adds the self. to the state vars
    for k in range(len(pairs)):
        rhs_ode = State_variable_Identifier.add_self_prefix(State_variable_Identifier.add_self_prefix(FormatODEs_Ns.reformat_input_time_indexing(FormatODEs_Ns.reformat_discrete_time_indexing(pairs[k][1])),parameters.keys()),lhs_list)
        rhs_ode_rpl = rhs_ode.replace("[t-1]", "[-1]")
        ode_string += f"                {pairs[k][0]} = {rhs_ode_rpl}\n"
        #if not State_variable_Identifier.add_self_prefix(State_variable_Identifier.add_self_prefix(FormatODEs_Ns.reformat_input_time_indexing(FormatODEs_Ns.reformat_discrete_time_indexing(pairs[k][1])),parameters.keys()),lhs_list).strip().isdigit() and State_variable_Identifier.add_self_prefix(State_variable_Identifier.add_self_prefix(FormatODEs_Ns.reformat_input_time_indexing(FormatODEs_Ns.reformat_discrete_time_indexing(pairs[k][1])),parameters.keys()),lhs_list).strip() != "0":
        #    ode_string += f"                print({pairs[k][0]}.grad_fn)\n                print({pairs[k][0]}.requires_grad)\n"
        #ode_string += f"                 print('hello')\n"  


    update_eulers = '\n\n                #Update Eulers\n'

    #Update Eulers
    both_sides2 = State_Parser.extract_state_update(file_path)

    update_vars = list(both_sides2.keys())
    update_vals = [list(v.values())[0] for v in both_sides2.values()]

    for k in range(len(update_vars)):

        rep_val = update_vals[k].replace("[t-1]", "[-1]")
        #update_eulers += f"                {update_vars[k][:-3]}.append(({State_variable_Identifier.add_self_prefix(rep_val,parameters.keys())}).view(()))\n"
        update_eulers += f"                {update_vars[k][:-3]}[-2] = {update_vars[k][:-3]}[-1]\n"
        update_eulers += f"                {update_vars[k][:-3]}[-1] = ({State_variable_Identifier.add_self_prefix(rep_val,parameters.keys())}).view(())\n"

    #Calculate spikes w/ surrogate function for forwards and backwards

    #The way S2OnOff_buffer_index is used is insane I think? It declares a set of ones and then picks one of them as an index
    #So any index you pick it will pick 1 and then use the first index. Maybe we meant to make this some sort of
    #hyperparameter at some point. But this is very unncessary it seems.

    #Will need to take another look at this for the multi channel model
    
    #This all feels very hard coded but I guess we will see.

    spiking_string = '\n\n                #Spiking and conditional actions\n\n'

    #Test 1 (Spike) -- directly linked to loss so surrogate is needed here

    for k in range(len(update_vars)):
        if "_V[t]" in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            var_name = var[:-5]
            var_prev = var.replace("[t]", "[-2]")
            var_thresh = f"self.{var.replace('[t]', '_thresh')}"

            

            spiking_string += f"                {var_base}_spikes_holder.append(SurrogateSpike.apply({var_base}[-1], {var_prev}, {var_thresh}))\n"
            spiking_string += f"                {var_base}_test = SurrogateSpike.apply({var_base}[-1], {var_prev}, {var_thresh})\n"
            #spiking_string += f"                print('test1')\n"
            spiking_string += f"                if {var_base}_test:\n"
            #if var_base == "S2OnOff_V":
            #    spiking_string += f"                   pdb.set_trace()\n"
            #spiking_string += f"                    print('test1')\n"                                             
            #spiking_string += f"                    print({var_base}.grad_fn)\n                    print({var_base}.requires_grad)\n                    print({var_base}_spikes.grad_fn)\n                    print({var_base}_spikes.requires_grad)\n"
            
            spiking_string += f"                    {var_name}_tspike[int({var_name}_buffer_index)-1] = helper[t]\n"
            spiking_string += f"                    {var_name}_buffer_index = ({var_name}_buffer_index % 5) + 1\n"

    #Test 2 (Voltage reset and adaptation and refractoriness?) 

    spiking_string += '\n\n'

    for k in range(len(update_vars)):
        if "_V[t]" in update_vars[k]:
            var = update_vars[k]
            var_base = var[:-3]
            var_name = var[:-5]
            var_thresh = f"self.{var.replace('[t]', '_thresh')}"
            var_reset = f"self.{var.replace('[t]', '_reset')}"
            var_adapt = f"{var.replace('V[t]', 'g_ad[t]')}"
            var_inc = f"self.{var.replace('V[t]', 'g_inc')}"

            spiking_string += f"                {var_base}_test2a = {var_base}[-1] > {var_thresh}\n"
            #spiking_string += f"               print({var_base}_test2a)\n"
            #spiking_string += f"               print('test2a')\n"
            spiking_string += f"                if {var_base}_test2a:\n"
            #spiking_string += f"                    {var[:-3]}.append({var_reset})\n"
            spiking_string += f"                    {var[:-3]}[-2] = {var[:-3]}[-1] \n"
            spiking_string += f"                    {var[:-3]}[-1] = {var_reset} \n"
            #spiking_string += f"                    {var_adapt[:-3]}.append({var_adapt[:-3]}[-1] + {var_inc})\n"
            spiking_string += f"                    {var_adapt[:-3]}[-2] = {var_adapt[:-3]}[-1]\n"
            spiking_string += f"                    {var_adapt[:-3]}[-1] = {var_adapt[:-3]}[-1] + {var_inc}\n"
            
            spiking_string += f"                {var_base}_test2b = torch.any(helper[t] <= {var_name}_tspike + self.{var_name}_t_ref)\n"
            #spiking_string += f"               print({var_base}_test2b)\n"
            #spiking_string += f"               print('test2b')\n" 
            spiking_string += f"                if {var_base}_test2b:\n"
            #spiking_string += f"                    {var[:-3]}.append({var_reset})\n"
            spiking_string += f"                    {var[:-3]}[-2] = {var[:-3]}[-1]\n"
            spiking_string += f"                    {var[:-3]}[-1] = {var_reset}\n"

    #Test 3 (Update PSC vars) 
    #(Probably how we should do things in the future)
    #(should honestly just be grabbing all of the lines and then reformatting them) \TODO

    statement_pairs = ConditionalActions.extract_conditional_variables(file_path)

    #print(statement_pairs[0][0])

    spiking_string += '\n\n'

    #print(statement_pairs)
    print('here5')

    for k in range(len(statement_pairs)):

        var_base = var[:-3]
        var = statement_pairs[k][1]
        var_x = f"{var.replace('delay', 'x')}"
        var_q = f"{var.replace('delay', 'q')}"
        var_F = f"{var.replace('delay', 'F')}"
        var_P = f"{var.replace('delay', 'P')}"
        var_fF = f"self.{var.replace('delay', 'fF')}"
        var_max = f"self.{var.replace('delay', 'maxF')}"
        var_fP = f"self.{var.replace('delay', 'fP')}"

        spiking_string += f"                {var_base}_test3 = torch.any(helper[t] == {statement_pairs[k][0]} + self.{statement_pairs[k][1]})\n"  
        #spiking_string += f"               print({var_base}_test3)\n"
        #spiking_string += f"               print('test3')\n"
        spiking_string += f"                if {var_base}_test3:\n"  
        #spiking_string += f"                    {var_x}.append({var_x}[-1] + {var_q}[-1])\n"
        #spiking_string += f"                    {var_q}.append({var_F}[-1] * {var_P}[-1])\n"
        #spiking_string += f"                    {var_F}.append({var_F}[-1] + {var_fF}*({var_max}-{var_F}[-1]))\n"
        #spiking_string += f"                    {var_P}.append({var_P}[-1] * (1 - {var_fP}))\n"
        spiking_string += f"                    {var_x}[-2] = {var_x}[-1]\n"
        spiking_string += f"                    {var_q}[-2] = {var_F}[-1]\n"
        spiking_string += f"                    {var_F}[-2] = {var_F}[-1]\n"
        spiking_string += f"                    {var_P}[-2] = {var_P}[-1]\n"
        spiking_string += f"                    {var_x}[-1] = {var_x}[-1] + {var_q}[-1]\n"
        spiking_string += f"                    {var_q}[-1] = {var_F}[-1] * {var_P}[-1]\n"
        spiking_string += f"                    {var_F}[-1] = {var_F}[-1] + {var_fF}*({var_max}-{var_F}[-1])\n"
        spiking_string += f"                    {var_P}[-1] = {var_P}[-1] * (1 - {var_fP})\n"


    #This needs to be fixed. Spikes lists need to be redeclared before each iteration. (change holder variable name)
    update_monitors = ''
        
    for k in range(len(monitor_vars)):
        if "R2On_V_spikes" in monitor_vars[k]:       
            update_monitors = update_monitors + f'            {monitor_vars[k]} = torch.stack({monitor_vars[k]}_holder, dim=0)\n'
            update_monitors = update_monitors + f'            print(len({monitor_vars[k]}))\n'
    
    

    return_statement = '''
    \n\n            #print(len(spike_holder))   
    \n\n            print(len(On_V_spikes))  
    \n\n            #spike_holder = torch.cat((spike_holder, On_V_spikes), dim=0)
    \n\n            #spike_holder = spike_holder.view(-1)
    \n\n            #R2On_V_spikes = R2On_V_spikes.view(-1)
    \n\n            #spike_holderOn = torch.cat((spike_holderOn, On_V_spikes), dim=0)
    \n\n            #spike_holderOff = torch.cat((spike_holderOff, Off_V_spikes), dim=0)
    \n\n            #spike_holderR1On = torch.cat((spike_holderR1On, R1On_V_spikes), dim=0)
    \n\n            spike_holderR2On = torch.cat((spike_holderR2On, R2On_V_spikes), dim=0)
    \n\n            #spike_holderS2OnOff = torch.cat((spike_holderS2OnOff, S2OnOff_V_spikes), dim=0)
    \n\n            #spike_holderS1OnOff = torch.cat((spike_holderS1OnOff, S1OnOff_V_spikes), dim=0)
    \n\n            #spike_holderR2Off = torch.cat((spike_holderR2Off, R2Off_V_spikes), dim=0)
    \n\n            #spike_holderR1Off = torch.cat((spike_holderR1Off, R1Off_V_spikes), dim=0)
    \n\n            print('made it here 5')
    \n\n        #print(max(self.On_On_IC_input))
    \n\n        #print(max(self.Off_Off_IC_input))
    \n\n        #return [On_V_spikes,Off_V_spikes,R1On_V_spikes,R1Off_V_spikes,R2On_V_spikes,S1OnOff_V_spikes,S2OnOff_V_spikes]
    \n\n        return spike_holderR2On
    \n\n        #return [spike_holderOn,spike_holderOff,spike_holderR1On,spike_holderR2On,spike_holderS2OnOff,spike_holderS1OnOff,spike_holderR2Off,spike_holderR1Off]\n\n'''

    # Combine full class
    generated_code = SurrogateSpiking_Class_declaration + main_class_declaration + learnable_block + nonlearnable_block + fixed_param_declaration + forwards_declaration + monitor_string + ode_string + update_eulers + spiking_string + update_monitors + return_statement



    

    #Put in out of function things

    

    #post_function_loop = '\n\ndef main():\n      model = LIF_ODE()\n      init_cond = ('

    # Initial conditions
    
    #init_str = ''

    #print(state_vals)

    #for k in range(len(state_vals)):
    #    if k < len(state_vals)-1:
    #        init_str += f"    {State_variable_Identifier.add_model_prefix(state_vals[k],parameters.keys())},\n"
    #    else:
    #        init_str += f"    {State_variable_Identifier.add_model_prefix(state_vals[k],parameters.keys())})\n"


    #Eventually need to make this so that we can tune hyperparameters from matlab or something

    #print(state_vars)

    #NOTE! Might need to change things if we do batching eventually??

    #Testing with 2 trials instead of 10 just to see if the backprop works

    training_loop = """

def main():
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    #print(torch.__version__)
    #print(torch.cuda.is_available())
    #print(torch.cuda.device_count())
    #print(torch.version.cuda)
    outputs = []
    #for k in range(10):

    #spike_saver = []

    model = LIF_ODE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.0, 0.999))
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.0)
    num_epochs = 2

    losses = []
    
    target_spikes = torch.tensor(50.0, dtype=torch.float32) #100/s
    
    spike_f = open("spikes.bin", "ab")


    for epoch in range(num_epochs):

        print(epoch)
        optimizer.zero_grad()

        from torchviz import make_dot

        output = model()  # forward pass

        

        #print("Forward pass ran successfully. Num Spikes")
        print('Avg Firing Rate')
        print(output.sum()/10/3)

        fr = output.sum()/10/3  #total spikes/num_trials/num_seconds

        loss = (fr - target_spikes)**2

        print('here')
        
        
        #dot = make_dot(loss)          # builds .dot graph
        #dot.render("lif_graph", format="pdf")  # opens in any PDF viewer

        #print("Loss requires grad:", loss.requires_grad)
        #print("Loss grad_fn:", loss.grad_fn)

        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print(f"{name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn}")
        
        #pdb.set_trace()

        print(type(output))                   # Tensor? List?
        print(output.requires_grad)
        print(output.grad_fn)

        print(fr.requires_grad)              # This is what goes into loss
        print(fr.grad_fn)


        #spikes_u8 = torch.stack(output).to(torch.uint8)

        #spikes_u8 = [thingy.detach().to(torch.uint8).cpu().numpy() for thingy in output]

        #spike_f.write(spikes_u8) 

        #spikes_bit.tofile("spikes.bin")

        #for tensor_holders in output:
        #    spike_f.write(tensor_holders.detach().to(torch.uint8).cpu().numpy())
            
        optimizer.zero_grad()

        loss.backward()

        del output 


        optimizer.step()
        gc.collect()
        #tracemalloc.clear_traces()
        
        print(f"Epoch {epoch}: Loss = {loss.item()}") 
        
        mid_ouputs = []

        #for k in range(len(output)):
        #    output_np_list = [tensor.detach().cpu().numpy() for tensor in output[k]]
        #    mid_ouputs.append(output_np_list)


        #outputs.append(mid_ouputs)


        

        #output_np_list = [tensor.detach().to(torch.uint8).cpu().numpy() for tensor in output]

        
        #outputs.append(output_np_list)

        

        losses.append(loss.item())

    #return outputs, losses
    #return losses

if __name__ == "__main__":
    main()
"""
    
    

    #generated_code = generated_code + post_function_loop + init_str + training_loop
    generated_code = generated_code + training_loop


    #Clean up a few things to make it pytorch compatible

    

    generated_code = Clean_up.Clean_gen_code(generated_code)

    #print(generated_code)

    #generated_code = add_device_to_tensors.Add_Device(generated_code)

    #print(generated_code)

    with open("generated.py", "w") as f:
        f.write(generated_code)

    print("generated.py has been created.")

    return generated_code



#\Todos

#Fill in init conditions for training loop & Fill in training loop


#Done

#Replace element wise operators (Replace ^ with ** as well)
#Remove randn statement from ODEs
#Remove self.tspan (Remove all unsed params up to verbose_flag)
#Remove on and off IC label
#Replace true with True & false with False
    #Does not exist?
#Do state variables need to be in init?
#Figure out genpoisson times/inputs
    #Just need to make sure we get access to the firing rate profiles when we actually try to run this


