import os
import Parser
import State_Parser

def build_ODE(parameters):

    #Create spiking handler class

    SurrogateSpiking_Class_declaration = """
import torch
import torch.nn as nn


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


    


    #Put in forwards header
    forwards_declaration = """

    def forward(self,t,state):
        
        #State Variables
        
        
"""
    
    #Add in the state variables
    state_string = '        '
    state_vars = State_Parser.extract_state_variables_from_block(file_path)
    for k in range(len(state_vars)):
        if  k == 0:
            state_string += f"{state_vars[k]}"
        else:
            state_string += f", {state_vars[k]}"

    state_string += " = state\n\n        #ODEs\n"

    ode_string = ''
    pairs = Parser.extract_rhs_lhs(file_path)

    #Loop through and fill in the ODES

    #Need to come up with an effecient way of formating these ODEs in the format that pytorch will accept
    for k in range(len(pairs)):
        ode_string += f"        {pairs[k][0]} = {pairs[k][1]}\n"

    # Combine full class
    generated_code = SurrogateSpiking_Class_declaration + main_class_declaration + learnable_block + nonlearnable_block + forwards_declaration + state_string + ode_string

    with open("generated.py", "w") as f:
        f.write(generated_code)

    print("generated.py has been created.")

    return generated_code




