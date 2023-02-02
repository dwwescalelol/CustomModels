classdef my_ClassificationPerceptron < handle
    %my_ClassificationPerceptron simple binary peceptron.

    properties
        X
        Y
        Weights         % - Weight bias.
        Bias            % - Node bias.
        LearningRate    % - Constant at which weights are updated.
        ClassNames      % - All unique class names.
        NumObservations % - Number of observations trained on.
        Function        % - Type of activation function.
    end

    methods
        function obj = my_ClassificationPerceptron(X,Y,LearningRate,Function)
            obj.X = X;
            obj.Y = Y;
            obj.ClassNames = unique(Y);
            obj.LearningRate = LearningRate;
            obj.Function = Function;

            % initialised as 0 insted of rand. 
            obj.Weights = zeros(1,width(X));
            obj.Bias = 0;
 
            obj.train(X,Y);
        end

        function train(obj, train_examples, train_labels)

            if width(train_examples)~=width(obj.Weights)
                error("Incorrect train data.")
            end
            obj.NumObservations = obj.NumObservations + height(train_examples);
                                   
            formatted_labels = obj.formatLabels(train_labels);

            for i=1:height(train_examples)
                % Get input and desired output
                input = train_examples(i,:);
                true_output = formatted_labels(i);
                
                % activation function
                if obj.Function == "Sigmoid"
                    output = obj.my_sigmoid(input);
                else
                    output = obj.my_relu(input);
                end
                
                % Update weights and bias
                obj.Weights = obj.Weights + obj.LearningRate*(true_output - output)*input;
                obj.Bias = obj.Bias + obj.LearningRate*(true_output - output);
            end
        end

        function output = my_sigmoid(obj, input)
            % activation function !
            % need to invert array for matrix multiplication.
            z = input*obj.Weights' + obj.Bias;
            % sigmoid
            output = 1/(1+exp(-z));            
        end

        function output = my_relu(obj, input)
            % activation function !
            % need to invert array for matrix multiplication.
            z = input*obj.Weights' + obj.Bias;
            % sigmoid
            output = max(0,z);
        end

        function prediction = predict(obj, test_examples)
 
            prediction = zeros(height(test_examples),1);
            for i=1:height(test_examples)
                output = obj.my_sigmoid(test_examples(i,:));

                prediction(i) = obj.functionToLabel(output);
            end          
        end

        function labelIndex = functionToLabel(obj, output)
            
            switch obj.Function
                case 'Sigmoid'
                    if output > 0.5
                        labelIndex = 1;
                    else
                        labelIndex = -1;
                    end
                otherwise
                    if output > 0
                        labelIndex = 1;
                    else
                        labelIndex = -1;
                    end
            end
        end
    end
    methods(Static)
        function train_values = formatLabels(labels)
            %formatLabels converts labels to either -1 or 1.
            if height(unique(labels)) > 2
                error("Data not binary.")
            end

            train_values = grp2idx(labels);
            % make labels binary 
            train_values(train_values==2) = -1;
        end


    end
end