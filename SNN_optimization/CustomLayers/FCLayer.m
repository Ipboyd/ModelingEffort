classdef FCpositiveWeights < nnet.layer.Layer
    % Custom fully connected layer, constrain weights to be positive.

    properties (Learnable)
        % Layer learnable parameters
        Weights
        Bias
    end
    
    methods
        function layer = FCpositiveWeights(prev_units,num_units,name,initWeights) 
            % layer = FCpositiveWeights(numInputs,name) creates a
            % fully connected layer with weights constrained to take on
            % positive values

            %a + (b-a).*
            % Set number of inputs.
            layer.Name = name;
            layer.Description = 'Fully connected, positive-weighted layer';
            
            if ~exist('initWeights','var')
                std = sqrt(2/(num_units+prev_units));
                layer.Weights = std*rand([num_units prev_units]);
            else
                assert(size(initWeights,1)==num_units) 
                assert(size(initWeights,2)==prev_units) 
                layer.Weights = initWeights;
            end
            
            layer.Bias = zeros([num_units 1]);
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
%             Z = log(1+exp(layer.Weights))*X + layer.Bias;
%             Z = max(0,layer.Weights)*X+layer.Bias;
            Z = layer.Weights*X+layer.Bias;
        end
        
    end
end