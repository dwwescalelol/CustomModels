% Description: create a model ready to perform DT classification
% from some training data
%
% Inputs:
% train_examples: a numeric array containing the training examples
% train_labels: a categorical array containing the associated
% labels (i.e., with the same ordering as train_examples)
% 
% Optionally, the user can also ask to set 'MinParentSize' (via an extra
% name-value pair) controlling the minimum number of training
% examples/labels that must be in a node for it to be considered for
% splitting (10 by default)
%
% Optionally, the user can also ask to set 'MaxNumSplits' (via an extra
% name-value pair) controlling the maximum number of splits in the
% resulting tree (number of training examples-1 by default, which is the
% theoretical maximum)
% 
% Optionally, the user can also ask to switch on 'Verbose' mode (via an
% extra name-value pair) causing the model to generate debug
% 
% Outputs:
% m: a my_ClassificationTree object holding the parameters of the resulting
% model
% 
% Notes:
% The only job here is to instantiate an object from the relevant
% class and hand it back; just providing equivalence with the Matlab
% implementation for familiarity/ease of use
%
function m = my_fitctree(train_examples, train_labels, varargin)

    p = inputParser;
    % take an extra name-value pair allowing us to set the minimum
    % parent size (10 by default):
    addParameter(p, 'MinParentSize', 10);
    % take an extra name-value pair allowing us to set the maximum
    % number of splits (number of training examples-1 by default):
    addParameter(p, 'MaxNumSplits', size(train_examples,1) - 1);

    p.parse(varargin{:});
    
    % use the supplied parameters to create a new my_ClassificationTree
    % object:
    
    m = my_ClassificationTree(train_examples, train_labels, ...
        p.Results.MinParentSize, p.Results.MaxNumSplits);
            
end