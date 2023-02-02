classdef my_ClassificationEnsemble < my_ClassificationModel
    %my_ClassificationEnsemble Summary of this class goes here
    %   Detailed explanation goes here

    properties
        Trained             % - Trained learners.
        NumTrained          % - Number of trained learners in the ensemble.      
    end

    methods
        function obj = my_ClassificationEnsemble(X, Y, Method, Learners, ...
                NumLearningCycles, NFeatureToSample)
            % superclass
            obj@my_ClassificationModel(X,Y);

            % use factory to create all classifiers.
            f = my_ClassifierFactory(X, Y, Method, Learners, NumLearningCycles, NFeatureToSample);
            obj.Trained = f.makeClassifiers();
            obj.NumTrained = height(obj.Trained);
        end

        function [predictions, scores] = predict(obj, test_examples)
            % scores predicted just as shown in lab 8.
            
            % initialise scores to be correct size
            scores = zeros(height(test_examples),height(obj.ClassNames));

            % foreach classifier in ensamble predict
            for i=1:obj.NumTrained
                % make test examples have same features as classifier (ensamble)
                X = test_examples(:,obj.Trained{i}.PredictorIndexs);

                [~,scores_i] = obj.Trained{i}.predict(X);
                scores = scores + scores_i;
            end

            % average scores
            scores = scores / obj.NumTrained;

            % largest indicy represents most confident classifier guess.
            [~, ind] = max(scores,[],2);
            predictions = obj.ClassNames(ind);
        end
    end
end