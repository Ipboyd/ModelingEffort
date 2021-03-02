classdef myRelu < nnet.layer.Layer
    % Example custom PReLU layer.

    properties (Learnable)
        % Layer learnable parameters
    end
    
    methods
        function layer = myRelu(name) 
            % layer = preluLayer(numChannels, name) creates a PReLU layer
            % for 2-D image input with numChannels channels and specifies 
            % the layer name.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "ReLU with custom slope";
            
        end
        
        function Z = predict(layer,X)
            % Z = predict(layer,X) forwards the input data X through the
            % layer and outputs the result Z.
            
            Z = max(X,0);
        end
    end
end