function m = my_fitcperc(train_examples, train_labels, varargin)
    
    p = inputParser;
    addParameter(p, 'Function', 'Sigmoid');
    addParameter(p, 'LearningRate', 0.1);

    p.parse(varargin{:});
        
    m = my_ClassificationPerceptron(train_examples, train_labels, ...
        p.Results.LearningRate,p.Results.Function);
end