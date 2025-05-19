import os
import Parser
import State_Parser
import FormatODEs_Ns
import State_variable_Identifier
import Extract_Fixed_vars
import Clean_up

def build_ODE(parameters):


    #Create spiking handler class

    SurrogateSpiking_Class_declaration = """
import torch
import torch.nn as nn
import numpy as np
import genPoissonTimes
import genPoissonInputs
import matplotlib.pyplot as plt


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        ctx.save_for_backward(input)
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output * (1.0 / (1.0 + torch.abs(input)) ** 2)
        return grad_input, None

"""

    # Header
    main_class_declaration = """

class LIF_ODE(nn.Module):
    def __init__(self):
        super().__init__()
        
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


    fixed_param_declaration = '\n        # Fixed Params\n'

    #print('made it here')

    #Add the calculable parameters
    fixed_params = Extract_Fixed_vars.extract_fixed_variables_from_block(file_path)
    
    #Get the left hadn and right hand side of these equations.
    lhs_list, rhs_list = zip(*fixed_params)

   
    for k in range(len(lhs_list)):
        fixed_param_declaration += f"        self.{lhs_list[k]} = {State_variable_Identifier.add_self_prefix(State_variable_Identifier.add_self_prefix(rhs_list[k],lhs_list),parameters.keys())}\n"
    
    #print(fixed_param_declaration)
    

    #Put in forwards header
    forwards_declaration = """

    def forward(self,t,state):
        
        #State Variables
        
        
"""
    
    #Add in the state variables
    #This code seems very redundent. Clean if state_vars or state_vals are not used somewhere
    #else at some point
    state_string = ''
    both_sides = State_Parser.extract_state_variables_from_block(file_path)


    state_vars = list(both_sides.keys())
    state_vals = [list(v.values())[0] for v in both_sides.values()]


    #state_vars, state_vals = zip(*both_sides)
    
    

    for k in range(len(state_vars)):
        state_string += f"        {state_vars[k]} = {state_vals[k]}\n"

    state_string += "\n\n        #ODEs\n\n        T = len(np.arange(self.tspan[0],self.dt,self.tspan[1]))\n\n"


    ode_string = '        for t in range(1,T):\n'
    pairs = Parser.extract_rhs_lhs(file_path)

    


    #Loop through and fill in the ODES

    #Need to come up with an effecient way of formating these ODEs in the format that pytorch will accept

    #The following loop:
    #1. Writes the ODes to matlab
    #2. Goes through and remove all of the (n-1,:)
    #3. Adds the self. to the state vars
    for k in range(len(pairs)):
        ode_string += f"            {pairs[k][0]} = {State_variable_Identifier.add_self_prefix(State_variable_Identifier.add_self_prefix(FormatODEs_Ns.reformat_input_time_indexing(FormatODEs_Ns.reformat_discrete_time_indexing(pairs[k][1])),parameters.keys()),lhs_list)}\n"  

    # Combine full class
    generated_code = SurrogateSpiking_Class_declaration + main_class_declaration + learnable_block + nonlearnable_block + fixed_param_declaration + forwards_declaration + state_string + ode_string

    #print(generated_code)

    #Put in out of function things

    

    post_function_loop = '\n\ndef main():\n      model = LIF_ODE()\n      init_cond = ('

    # Initial conditions
    
    init_str = ''

    print(state_vals)

    for k in range(len(state_vals)):
        if k < len(state_vals)-1:
            init_str += f"    {State_variable_Identifier.add_model_prefix(state_vals[k],parameters.keys())},\n"
        else:
            init_str += f"    {State_variable_Identifier.add_model_prefix(state_vals[k],parameters.keys())})\n"


    #Eventually need to make this so that we can tune hyperparameters from matlab or something

    print(state_vars)

    training_loop = f"""
    t = torch.linspace(0, 5, 100)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


    # Training loop
    for step in range(500):
        optimizer.zero_grad()
    
        pred_z = odeint(model, ninit_cond, t)  # pred_z shape: (T, state_dim)

        # Suppose R2On_V is stored as part of the full state vector at index `i_R2On_V`
        R2On_V_trace = pred_z[:, {State_variable_Identifier.Find_Var(file_path)}]  # shape: (T, N_R2On)

        # Compute average firing rate over time
        avg_fr = compute_avg_firing_rate(R2On_V_trace)

        print(avg_fr)

        # Define a target firing rate (e.g., 10 Hz)
        target_fr = torch.tensor(0.01)  # if time units are seconds and you're simulating 1000 steps

        loss = (avg_fr - target_fr).pow(2)
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 499:
            print(f"Step {{step}} | Loss: {{loss.item():.6f}} | Avg FR: {{avg_fr.item():.6f}}")

if __name__ == "__main__":
    main()



"""
    
    

    generated_code = generated_code + post_function_loop + init_str + training_loop

    #Clean up a few things to make it pytorch compatible

    

    generated_code = Clean_up.Clean_gen_code(generated_code)

    print(generated_code)

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


