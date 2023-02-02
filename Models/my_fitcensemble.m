function m = my_fitcensemble(train_examples, train_labels, varargin)
    %my_fitcensemble Summary of this function goes here
    %   Detailed explanation goes here
    
    p = inputParser;
    
    addParameter(p, 'Method', 'SoftVote');
    addParameter(p, 'Learners', my_templateKNN());
    % defaults as described in Lab 8.
    addParameter(p, 'NPredToSample', round(sqrt(width(train_examples))));
    addParameter(p, 'NumLearningCycles', 100);

    p.parse(varargin{:});
    
    % use the supplied parameters to create a new my_ClassificationTree
    % object:

    m = my_ClassificationEnsemble(train_examples, train_labels, ...
        p.Results.Method, p.Results.Learners, p.Results.NumLearningCycles, ...
        p.Results.NPredToSample);
end