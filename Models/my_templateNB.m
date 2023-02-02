classdef my_templateNB
    %my_templateNB Creates a classification Naive Bayes template that
    %   stores all hyperparameters for my_ClassificationNaiveBayes. For 
    %   use with my_fitcensamble as argument for 'Lables'. Since 
    %   my_ClassificationNaiveBayes has no hyperparameters this is 
    %   an empty class <:(.

    properties
    end

    methods
        function obj = my_templateNB(varargin)
            p = inputParser;
            p.parse(varargin{:});
        end
    end
end