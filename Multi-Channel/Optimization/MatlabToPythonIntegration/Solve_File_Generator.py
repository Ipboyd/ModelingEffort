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
    #



    # Combine full class
    generated_code = SurrogateSpiking_Class_declaration + main_class_declaration + learnable_block + nonlearnable_block

    return generated_code

#with open("generated.py", "w") as f:
#    f.write(generated_code)

#print("generated.py has been created.")
