import re

def Add_Device(generated_code):

    lines = generated_code.split('\n')

    # Patterns to match
    tensor_funcs = [
        r'torch\.tensor\((.*?)\)',     # torch.tensor(...)
        r'torch\.ones\((.*?)\)',       # torch.ones(...)
        r'torch\.zeros\((.*?)\)',      # torch.zeros(...)
        r'torch\.eye\((.*?)\)'        # torch.eye(...)
        
    ]

    # Add device=self.device if not already present
    updated_lines = []
    for line in lines:
        updated_line = line
        for pattern in tensor_funcs:
            matches = re.findall(pattern, line)
            for match in matches:
                # If 'device=' is not already in the match
                if "device=" not in match:
                    replacement = f"{match}, device=self.device"
                    updated_line = re.sub(pattern, lambda m: m.group(0).replace(match, replacement), updated_line)
        updated_lines.append(updated_line)

    generated_code = '\n'.join(updated_lines)
    return generated_code
