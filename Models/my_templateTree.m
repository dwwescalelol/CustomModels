classdef my_templateTree
    %my_templateTree Creates a classification Decision Tree template that
    %   stores all hyperparameters for my_ClassificationTree. For use with
    %   my_fitcensamble as argument for 'Lables'.

    properties
        MinParentSize       % - Minimum amount of data to be a parent.
        MaxNumSplits        % - Maximum total splits in tree.
    end

    methods
        function obj = my_templateTree(varargin)
            p = inputParser;
            addParameter(p, 'MinParentSize', 10);
            addParameter(p, 'MaxNumSplits', -1);
            p.parse(varargin{:});

            obj.MinParentSize = p.Results.MinParentSize;
            obj.MaxNumSplits = p.Results.MaxNumSplits;
        end
    end
end