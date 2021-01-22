classdef negationLayer < nnet.layer.Layer
    % negates the activations of the previous layer.

    methods
        function layer = negationLayer(name) 
            % layer = weightedAdditionLayer(numInputs,name) creates a
            % weighted addition layer and specifies the number of inputs
            % and the layer name.

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "negates the activation of the previous layer";
        
        end
        
        function Z = predict(layer,X)
            % Z = predict(X) forwards the input data X through the layer and outputs the result Z.
            
            Z = -1*X;
            
        end
    end
end