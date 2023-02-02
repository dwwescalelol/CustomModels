classdef (Abstract) my_ClassificationModel <  matlab.mixin.Heterogeneous & handle
    %my_ClassificationModel Superclass for all clasification models made 
    %   by me <:). Stores values used across all clasifiers and enforces
    %   that all inherited classes must have the predict method.

    properties
        X               % - Training examples.
        Y               % - Taining labels.
        ClassNames      % - All unique class names.
        NumObservations % - Number of observations in training data.
        PredictorIndexs % - Indexs of predictors used for this ensemble.
    end

    methods
        function obj = my_ClassificationModel(X, Y)
            % Superclass constructor, can never be used by itself as class
            % is abstract. Can be used in inherited classes.
            obj.X = X;
            obj.Y = Y;
            obj.ClassNames = unique(Y);
            obj.NumObservations = height(Y);
            obj.PredictorIndexs = 1:width(X);
        end
    end

    methods (Abstract)
        predict(obj)    % - All models have a predict method.
    end
end