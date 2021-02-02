classdef FCInhibitoryInput < nnet.layer.Layer
    % Custom fully connected layer, constrain weights to be positive.

    properties (Learnable)
        % Layer learnable parameters
        Weights
        Bias
    end
    
    methods
        function layer = FCInhibitoryInput(prev_units,num_units,name,initWeights) 
            % layer = FCInhibitoryInput(numInputs,name) creates a
            % fully connected layer

            %a + (b-a).*
            % Set number of inputs.
            layer.Name = name;
            layer.Description = 'Fully connected, layer. Input is negated to simulate inhibition';
            
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
            W = layer.Weights;
            
%             geluW = 0.5*W*(1+tanh(sqrt(2/pi)*(W+0.044715*W.^3)));
%             Z = geluW*X+layer.Bias;
%             ELUW = W.*(W > 0) + 0.5*(exp(W)-1).*(W<0);
%             Z = -ELUW*X+layer.Bias;

            % simple relu
%             Z = -(max(0,W))*X+layer.Bias; %relu

            % soft relu
            softW = log(1+exp(W))-0.6931;
%             softW = log(1+exp(W));
            Z = -softW*X+layer.Bias;
            
            % basic forward eqn
%             Z = -W*X+layer.Bias;
        end

    end
end