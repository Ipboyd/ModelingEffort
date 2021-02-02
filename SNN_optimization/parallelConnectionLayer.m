classdef parallelConnectionLayer < nnet.layer.Layer
    % Custom layer with 1-to-1 connection between neurons in each layer.
    % Size of the presynaptic layer and post synaptic layer should be
    % equal.

    properties (Learnable)
        % Layer learnable parameters
            
        % Scaling coefficients
        W
        b
    end
    
    methods
        function layer = parallelConnectionLayer(input_shape,name) 
            % layer = weightedAdditionLayer(numInputs,name) creates a
            % weighted addition layer and specifies the number of inputs
            % and the layer name.

            % Set number of inputs.
            layer.Name = name;
            layer.Description = '1-to-1 connected layer';
            layer.W = rand([input_shape input_shape]);
            layer.b = rand([1 input_shape]);
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result
            %
            % Inputs:
            %         layer    -    Layer to forward propagate through
            %         X        -    Input data
            % Output:
            %         Z        -    Output of layer forward function
            % Layer forward function for prediction goes here
%             disp('W:')
%             disp(size(layer.W))
%             disp('X:')
%             disp(size(X))
%             disp('b:')
%             disp(size(layer.b))
            mask = eye(size(layer.W));
            Z = (mask.*layer.W)*X + layer.b;
        end
    end
end