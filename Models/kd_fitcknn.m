% Description: create a model ready to perform k-NN classification from
% some training data, implements a k-dtree for faster searching
%
% Inputs:
% train_examples: a numeric array containing the training examples
% train_labels: a categorical array containing the associated
% labels (i.e., with the same ordering as train_examples)
% 
% Optionally, the user can also ask to set 'k' (via an extra
% name-value pair) controlling the number of nearest neighbours used during
% prediction (1 by default)
% 
% Optionally, the user can also ask to switch on 'Verbose' mode (via an
% extra name-value pair) causing the model to generate debug
%
% Outputs:
% m: a kd_ClassificationKNN object holding the parameters of the resulting
% model

function m = kd_fitcknn(train_examples, train_labels, varargin)

    % take an extra name-value pair allowing us to turn debug on:
    p = inputParser;
    addParameter(p, 'Verbose', false);
    % take an extra name-value pair allowing us to set the number of
    % nearest neighbours used during prediction (1 by default):
    addParameter(p, 'k', 1);

    p.parse(varargin{:});
    
    % use the supplied parameters to create a new kd_ClassificationKNN
    % object:
    
    m = kd_ClassificationKNN(train_examples, train_labels, ...
        p.Results.k, p.Results.Verbose);
            
end