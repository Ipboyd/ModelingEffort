classdef FCInhibitoryInput < nnet.layer.Layer
    % Custom fully connected layer, constrain weights to be positive.
    properties
        mask
        
    end
    
    properties (Learnable)
        % Layer learnable parameters
        Weights
        a
        b
    end
    
    methods
        function layer = FCInhibitoryInput(prev_units,num_units,name,initWeights,mask) 
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
            
%             layer.Bias = zeros([num_units 1]);
            
            % Mask on weights
            if ~exist('mask','var')
                layer.mask = (ones(4)-eye(4));
            else
                layer.mask = mask;
            end
            std = sqrt(2/(num_units+prev_units));
            layer.a = std*rand([prev_units, 1]);
            layer.b = rand([prev_units, 1])+0.5;
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
            W = layer.Weights .* layer.mask; %remove weights along diagnol
            W = max(0,W);
%             geluW = 0.5*W*(1+tanh(sqrt(2/pi)*(W+0.044715*W.^3)));
%             Z = geluW*X+layer.Bias;
%             ELUW = W.*(W > 0) + 0.5*(exp(W)-1).*(W<0);
%             Z = -ELUW*X+layer.Bias;

            % simple relu
%             Z = -(max(0,W))*X+layer.Bias; %relu
            A  = max(0,layer.a);
            B  = layer.b;
            Z = -W*(max(1E-3,(X-A)).^B);

            % soft relu
%             softW = log(1+exp(W))-0.6931;
%             softW = log(1+exp(W));
%             Z = -W*softW(X)+layer.Bias;
            
            % basic forward eqn
%             Z = -W*X;
        end

    end
end