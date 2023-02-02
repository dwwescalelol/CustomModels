classdef my_templateKNN
    %my_templateKNN Summary Creates a classification KNN template that
    %   stores all hyperparameters for my_ClassificationKNN. For use with
    %   my_fitcensamble as argument for 'Lables'.

    properties
        NumNeighbors
    end

    methods
        function obj = my_templateKNN(varargin)
            p = inputParser;
            addParameter(p, 'NumNeighbors', 1);
            p.parse(varargin{:});

            obj.NumNeighbors = p.Results.NumNeighbors;
        end
    end
end